"""Predict from temporal sequences and static tabular features with FuseMoE.

mTAND turns ECG, CXR and optional raw time series into sequences. The
encoder routes these through experts. The prediction head joins their final
positions in modality order, followed by tabular features and availability.

Run from the repository root with:

    python -m src.train --config config/fusemoe.yaml
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.data.collate import modality_available, sequence_available, tabular_available
from src.encoders.mtand import MTANDSequenceEncoder
from src.models.base import BaseMultimodalModel
from src.models.smp import (
    ActivationType,
    GatingType,
    OutputMode,
    RouterScore,
    RouterType,
    SparseMoE,
    SparseMoEConfig,
)

MODALITIES = ("tabular", "ecg", "cxr", "ts")


def _flat_to_seq(x: torch.Tensor, seq_len: int, d_model: int) -> torch.Tensor:
    """Restore a flattened batch to [sequence, batch, features]."""
    return x.reshape(x.shape[0], seq_len, d_model).permute(1, 0, 2).contiguous()


def _seq_to_flat(x: torch.Tensor) -> torch.Tensor:
    """Flatten the sequence and features for each batch row."""
    return x.permute(1, 0, 2).contiguous().reshape(x.shape[1], -1)


class ModalityTokenTypeEmbedding(nn.Module):
    """Add a learned vector for each modality to its sequence."""

    def __init__(self, num_modalities: int, d_model: int) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        self.num_modalities = int(num_modalities)
        self.d_model = int(d_model)
        self.token_type_embeddings = nn.Embedding(self.num_modalities, self.d_model)

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Add each modality's learned vector and return the updated sequences."""
        _validate_modality_list(x_list, self.num_modalities, self.d_model)
        out: list[torch.Tensor] = []
        for modality_index, x in enumerate(x_list):
            # Add the same modality vector at every position
            id_vec = self.token_type_embeddings.weight[modality_index]
            out.append(x + id_vec)
        return out


def _validate_modality_list(x_list: list[torch.Tensor], num_modalities: int, d_model: int) -> None:
    """Check modality count, three-dimensional shape and feature width."""
    if len(x_list) != num_modalities:
        raise ValueError(
            f"expected a list of {num_modalities} per-modality tensors, got {len(x_list)}.",
        )
    for modality_index, x in enumerate(x_list):
        if x.ndim != 3:
            raise ValueError(
                "expected sequence-first tensors [S, B, d], got shape "
                f"{tuple(x.shape)} for modality {modality_index}."
            )
        if x.shape[-1] != d_model:
            raise ValueError(
                f"expected last dim d={d_model}, got {x.shape[-1]} for modality "
                f"{modality_index}."
            )


class SequenceEmbeddingAugmentation(nn.Module):
    """Scale sequences, add shared learned positions and apply dropout."""

    def __init__(
        self, num_modalities: int, seq_len: int, d_model: int, *, embed_dropout: float = 0.0,
        use_embed_scale: bool = True, use_positional: bool = True,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")

        self.num_modalities = int(num_modalities)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.embed_scale = math.sqrt(self.d_model) if use_embed_scale else 1.0

        if use_positional:
            # Share learned positions and stop gradients for position 0
            self.positional: nn.Module = nn.Embedding(self.seq_len, self.d_model, padding_idx=0)
            nn.init.normal_(self.positional.weight, std=0.02)
            self.register_buffer("positions", torch.arange(self.seq_len), persistent=False)
        else:
            self.positional = None
            self.positions = None

        self.dropout = nn.Dropout(embed_dropout)

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Scale each sequence, add positions and apply dropout."""
        _validate_modality_list(x_list, self.num_modalities, self.d_model)
        if x_list[0].shape[0] != self.seq_len:
            raise ValueError(f"expected S={self.seq_len}, got {x_list[0].shape[0]}.")

        pos = None
        if self.positional is not None:
            # Apply the same positions to every batch row
            pos = self.positional(self.positions).unsqueeze(1)

        out: list[torch.Tensor] = []
        for x in x_list:
            x = self.embed_scale * x
            if pos is not None:
                x = x + pos
            out.append(self.dropout(x))
        return out


class LastSequencePositionConcat(nn.Module):
    """Join the final sequence position from each modality in list order."""

    def __init__(self, num_modalities: int, d_model: int) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        self.num_modalities = int(num_modalities)
        self.d_model = int(d_model)

    def forward(self, hidden_list: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate the last sequence position from each modality."""
        _validate_modality_list(hidden_list, self.num_modalities, self.d_model)
        return torch.cat([hidden[-1] for hidden in hidden_list], dim=1)


class ResidualProjectionHead(nn.Module):
    """Apply two linear layers, add the input and predict logits."""

    def __init__(self, input_dim: int, output_dim: int = 1, *, dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than 0.")
        if output_dim <= 0:
            raise ValueError("output_dim must be greater than 0.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.proj1 = nn.Linear(self.input_dim, self.input_dim)
        self.proj2 = nn.Linear(self.input_dim, self.input_dim)
        self.out_layer = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map [batch, input_dim] features to [batch, output_dim] logits."""
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D tensor [B, input_dim], got shape {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim input_dim={self.input_dim}, got {x.shape[-1]}.")
        projected = self.proj2(self.dropout(F.relu(self.proj1(x))))
        projected = projected + x
        return self.out_layer(projected)


class MultiModalSequenceMoEBlock(nn.Module):
    """Route flattened sequences, restore modality order and add the original inputs."""

    def __init__(
        self, num_modalities: int, seq_len: int, d_model: int, *, router_type: RouterType = "joint",
        num_experts: int = 6, hidden_size: int = 256, top_k: int = 2, disjoint_top_k: int = 1,
        gating: GatingType = "softmax", activation: ActivationType = "gelu", dropout: float = 0.0,
        loss_coef: float = 1e-2, pre_layer_norm: bool = False, res_dropout: float = 0.0,
        output_mode: OutputMode = "log_prob", router_score: RouterScore = "dot",
        use_shared_expert: bool = False, shared_expert_weight: float | None = None,
        modality_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if router_type == "combination":
            raise ValueError(
                "MultiModalSequenceMoEBlock does not support router_type='combination'. "
                "Use 'joint', "
                f"'permod', or 'disjoint'. Got {router_type!r}."
            )

        self.num_modalities = int(num_modalities)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.flat_size = self.seq_len * self.d_model
        self.loss_coef = float(loss_coef)
        # Sequence routing starts after tabular features
        if modality_names is None:
            modality_names = MODALITIES[1 : 1 + self.num_modalities]

        # Normalise each modality before routing when requested
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.d_model) if pre_layer_norm else nn.Identity()
            for _ in range(self.num_modalities)
        )
        self.res_dropout = nn.Dropout(res_dropout)
        config = SparseMoEConfig(
            input_size=self.num_modalities * self.flat_size,
            output_size=self.num_modalities * self.flat_size, num_experts=num_experts,
            hidden_size=hidden_size, top_k=top_k, disjoint_top_k=disjoint_top_k,
            router_type=router_type, gating=gating, num_modalities=self.num_modalities,
            dropout=dropout, activation=activation, output_mode=output_mode,
            router_score=router_score, use_shared_expert=use_shared_expert,
            shared_expert_weight=shared_expert_weight, modality_names=tuple(modality_names),
        )
        self.moe = SparseMoE(config)

    def forward(
        self, x_list: list[torch.Tensor], train: bool | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return updated sequences, the routing loss and gate weights."""
        _validate_modality_list(x_list, self.num_modalities, self.d_model)
        seq_len, _, _ = x_list[0].shape
        if seq_len != self.seq_len:
            raise ValueError(f"expected S={self.seq_len}, got {seq_len}.")

        residuals = x_list
        flat_list = []
        for modality_index, x in enumerate(x_list):
            normed = self.norms[modality_index](x)
            flat_list.append(_seq_to_flat(normed))

        output, aux_loss, gates, _ = self.moe(
            flat_list, train=self.training if train is None else train, loss_coef=self.loss_coef,
        )
        hidden_list: list[torch.Tensor] = []
        for modality_index in range(self.num_modalities):
            start = modality_index * self.flat_size
            chunk = output[:, start : start + self.flat_size]
            hidden = _flat_to_seq(chunk, self.seq_len, self.d_model)
            hidden_list.append(residuals[modality_index] + self.res_dropout(hidden))

        return hidden_list, aux_loss, gates


class MultiModalSequenceMoETransformerLayer(nn.Module):
    """Update sequences with attention, expert routing and a feed-forward network."""

    def __init__(
        self, num_modalities: int, seq_len: int, d_model: int, *, num_heads: int = 4,
        ffn_dim: int | None = None, attn_dropout: float = 0.0, relu_dropout: float = 0.0,
        res_dropout: float = 0.0, router_type: RouterType = "joint", num_experts: int = 6,
        hidden_size: int = 256, top_k: int = 2, disjoint_top_k: int = 1,
        gating: GatingType = "softmax", activation: ActivationType = "gelu", dropout: float = 0.0,
        loss_coef: float = 1e-2, output_mode: OutputMode = "log_prob",
        router_score: RouterScore = "dot", use_shared_expert: bool = False,
        shared_expert_weight: float | None = None, modality_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.num_modalities = int(num_modalities)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        ffn_dim = 4 * self.d_model if ffn_dim is None else int(ffn_dim)
        self.norm1 = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.num_modalities))
        self.self_attn = nn.ModuleList(
            nn.MultiheadAttention(self.d_model, num_heads, dropout=attn_dropout, batch_first=False)
            for _ in range(self.num_modalities)
        )
        self.norm3 = nn.ModuleList(nn.LayerNorm(self.d_model) for _ in range(self.num_modalities))
        self.fc1 = nn.ModuleList(
            nn.Linear(self.d_model, ffn_dim) for _ in range(self.num_modalities)
        )
        self.fc2 = nn.ModuleList(
            nn.Linear(ffn_dim, self.d_model) for _ in range(self.num_modalities)
        )
        self.relu_dropout = nn.Dropout(relu_dropout)
        self.res_dropout = nn.Dropout(res_dropout)
        self.moe_block = MultiModalSequenceMoEBlock(
            self.num_modalities, seq_len, d_model, router_type=router_type, num_experts=num_experts,
            hidden_size=hidden_size, top_k=top_k, disjoint_top_k=disjoint_top_k, gating=gating,
            activation=activation, dropout=dropout, loss_coef=loss_coef, pre_layer_norm=True,
            res_dropout=res_dropout, output_mode=output_mode, router_score=router_score,
            use_shared_expert=use_shared_expert, shared_expert_weight=shared_expert_weight,
            modality_names=modality_names,
        )

    def forward(
        self, x_list: list[torch.Tensor], train: bool | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return updated modality sequences, the routing loss and gate weights."""
        _validate_modality_list(x_list, self.num_modalities, self.d_model)

        # Add the attention output to the original sequence
        after_attn: list[torch.Tensor] = []
        for modality_index, x in enumerate(x_list):
            normed = self.norm1[modality_index](x)
            attn_out, _ = self.self_attn[modality_index](normed, normed, normed, need_weights=False)
            after_attn.append(x + self.res_dropout(attn_out))

        after_moe, aux_loss, gates = self.moe_block(after_attn, train=train)
        out_list: list[torch.Tensor] = []
        for modality_index, x in enumerate(after_moe):
            normed = self.norm3[modality_index](x)
            ff = self.fc2[modality_index](
                self.relu_dropout(F.relu(self.fc1[modality_index](normed))),
            )
            out_list.append(x + self.res_dropout(ff))

        return out_list, aux_loss, gates


class MultiModalSequenceMoEEncoder(nn.Module):
    """Prepare modality sequences, run encoder layers and normalize their outputs."""

    def __init__(
        self, num_modalities: int, seq_len: int, d_model: int, num_layers: int, *,
        apply_token_type: bool = True, apply_embedding_augmentation: bool = False,
        embed_dropout: float = 0.0, final_layer_norm: bool = True, num_heads: int = 4,
        ffn_dim: int | None = None, attn_dropout: float = 0.0, relu_dropout: float = 0.0,
        res_dropout: float = 0.0, router_type: RouterType = "joint", num_experts: int = 6,
        hidden_size: int = 256, top_k: int = 2, disjoint_top_k: int = 1,
        gating: GatingType = "softmax", activation: ActivationType = "gelu", dropout: float = 0.0,
        loss_coef: float = 1e-2, output_mode: OutputMode = "log_prob",
        router_score: RouterScore = "dot", use_shared_expert: bool = False,
        shared_expert_weight: float | None = None, modality_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be greater than 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")

        self.num_modalities = int(num_modalities)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.token_type = (
            ModalityTokenTypeEmbedding(self.num_modalities, self.d_model)
            if apply_token_type
            else None
        )
        # Apply scaling, positions and dropout after modality embeddings
        self.embedding_augmentation = (
            SequenceEmbeddingAugmentation(
                self.num_modalities, self.seq_len, self.d_model, embed_dropout=embed_dropout,
            )
            if apply_embedding_augmentation
            else None
        )
        self.layers = nn.ModuleList(
            MultiModalSequenceMoETransformerLayer(
                self.num_modalities, seq_len, d_model, num_heads=num_heads, ffn_dim=ffn_dim,
                attn_dropout=attn_dropout, relu_dropout=relu_dropout, res_dropout=res_dropout,
                router_type=router_type, num_experts=num_experts, hidden_size=hidden_size,
                top_k=top_k, disjoint_top_k=disjoint_top_k, gating=gating, activation=activation,
                dropout=dropout, loss_coef=loss_coef, output_mode=output_mode,
                router_score=router_score, use_shared_expert=use_shared_expert,
                shared_expert_weight=shared_expert_weight, modality_names=modality_names,
            )
            for _ in range(num_layers)
        )
        self.final_norms = nn.ModuleList(
            nn.LayerNorm(self.d_model) if final_layer_norm else nn.Identity()
            for _ in range(self.num_modalities)
        )

    def forward(
        self, x_list: list[torch.Tensor], train: bool | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return sequences, the summed routing loss and the last layer's gate weights."""
        _validate_modality_list(x_list, self.num_modalities, self.d_model)

        hidden_list = list(x_list)
        if self.token_type is not None:
            hidden_list = self.token_type(hidden_list)
        if self.embedding_augmentation is not None:
            hidden_list = self.embedding_augmentation(hidden_list)

        aux_losses: list[torch.Tensor] = []
        gates = None
        for layer in self.layers:
            hidden_list, aux_loss, gates = layer(hidden_list, train=train)
            aux_losses.append(aux_loss)

        total_aux_loss = torch.stack(aux_losses).sum()
        hidden_list = [self.final_norms[m](h) for m, h in enumerate(hidden_list)]
        return hidden_list, total_aux_loss, gates


class SparseMoEModel(BaseMultimodalModel):
    """Route temporal sequences through experts, then combine with tabular features."""

    supports_aux_output = True

    def __init__(
        self, tabular_dim: int, ecg_dim: int, cxr_dim: int, d_model: int, output_dim: int = 1,
        dropout: float = 0.0, num_experts: int = 6, moe_hidden_size: int = 256, top_k: int = 2,
        disjoint_top_k: int = 1, router_type: RouterType = "joint", gating: GatingType = "softmax",
        activation: ActivationType = "gelu", output_mode: OutputMode = "log_prob",
        loss_coef: float = 1e-2, use_modality_mask: bool = True, router_score: RouterScore = "dot",
        use_shared_expert: bool = False, shared_expert_weight: float | None = None,
        mtand_tt_max: int = 64, mtand_embed_time: int = 64, mtand_num_heads: int = 8,
        mtand_time_scale: float = 90.0, ts_modality_enabled: bool = False,
        ts_modality_concat_mask: bool = True, ts_dim: int | None = None, num_layers: int = 1,
        num_heads: int = 8, ffn_dim: int | None = None, embed_dropout: float = 0.0,
        feature_dims: dict[str, int] | None = None,
        modality_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if ts_modality_enabled and (ts_dim is None or int(ts_dim) <= 0):
            raise ValueError(
                "ts_modality.enabled requires a --record all bundle exposing ts_feats; "
                f"feature_dims has no positive 'ts' width (got ts_dim={ts_dim!r})."
            )

        super().__init__(
            tabular_dim=tabular_dim, ecg_dim=ecg_dim, cxr_dim=cxr_dim, d_model=d_model,
            output_dim=output_dim, dropout=dropout, feature_dims=feature_dims,
            modality_names=modality_names,
        )
        if output_dim != 1:
            raise ValueError("SparseMoEModel currently supports binary output_dim=1.")
        if router_type == "combination":
            raise ValueError(
                "fusemoe does not support router_type='combination'. The combination router "
                "is static-only: use fusion_method 'smp' (src/models/smp.py), or one of "
                "'joint', 'permod' or 'disjoint' here."
            )

        self.ts_modality_enabled = bool(ts_modality_enabled)
        self.ts_modality_concat_mask = bool(ts_modality_concat_mask)
        self.ts_dim: int | None = None
        if self.ts_modality_enabled:
            self.ts_dim = int(ts_dim)

        # Raw time series follow tabular, ECG and CXR when enabled
        self.active_modalities = (
            MODALITIES if self.ts_modality_enabled else tuple(self.modality_names)
        )
        # Every modality after tabular is routed as a sequence
        n_seq = len(self.active_modalities) - 1
        self.d_model = d_model
        self.output_dim = output_dim
        self.output_mode = output_mode
        self.loss_coef = loss_coef
        self.use_modality_mask = use_modality_mask
        self.router_type = router_type
        self.mtand_time_scale = float(mtand_time_scale)

        # Keep every reference position for the sequence encoder
        self.ecg_encoder = MTANDSequenceEncoder(
            input_dim=ecg_dim, d_model=d_model, embed_time=mtand_embed_time,
            num_heads=mtand_num_heads, tt_max=mtand_tt_max, pool="none", dropout=dropout,
        )
        self.cxr_encoder = MTANDSequenceEncoder(
            input_dim=cxr_dim, d_model=d_model, embed_time=mtand_embed_time,
            num_heads=mtand_num_heads, tt_max=mtand_tt_max, pool="none", dropout=dropout,
        )
        if self.ts_modality_enabled:
            self.ts_encoder = MTANDSequenceEncoder(
                input_dim=self.ts_dim, d_model=d_model, embed_time=mtand_embed_time,
                num_heads=mtand_num_heads, tt_max=mtand_tt_max, pool="none", dropout=dropout,
                concat_mask=self.ts_modality_concat_mask,
            )

        self.faithful_seq_len = int(mtand_tt_max)
        self.faithful_encoder = MultiModalSequenceMoEEncoder(
            num_modalities=n_seq, seq_len=self.faithful_seq_len, d_model=d_model,
            num_layers=num_layers, apply_token_type=True, apply_embedding_augmentation=True,
            embed_dropout=embed_dropout, final_layer_norm=True, num_heads=num_heads,
            ffn_dim=ffn_dim, attn_dropout=dropout, relu_dropout=dropout, res_dropout=dropout,
            router_type=router_type, num_experts=num_experts, hidden_size=moe_hidden_size,
            top_k=top_k, disjoint_top_k=disjoint_top_k, gating=gating, activation=activation,
            dropout=dropout, loss_coef=loss_coef, output_mode=output_mode,
            router_score=router_score, use_shared_expert=use_shared_expert,
            shared_expert_weight=shared_expert_weight,
            modality_names=tuple(self.active_modalities[1:]),
        )
        self.faithful_aggregate = LastSequencePositionConcat(n_seq, d_model)
        self.faithful_head = ResidualProjectionHead(
            n_seq * d_model + d_model + n_seq, output_dim, dropout=dropout,
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode tabular features and the enabled modalities' time-aligned sequences."""
        scale = self.mtand_time_scale
        encoded = {
            "tabular": self.tabular_encoder(batch["tab_feats"], batch.get("tab_lengths")),
            "ecg": self.ecg_encoder(
                batch["ecg_feats"], batch["ecg_time"] / scale, batch["ecg_lengths"],
            ),
            "cxr": self.cxr_encoder(
                batch["cxr_feats"], batch["cxr_time"] / scale, batch["cxr_lengths"],
            ),
        }
        if self.ts_modality_enabled:
            ts_feats = batch["ts_feats"]
            obs_mask = 1.0 - batch["ts_missing"].to(dtype=ts_feats.dtype)
            encoded["ts"] = self.ts_encoder(
                ts_feats, batch["ts_time"] / scale, batch["ts_lengths"], obs_mask=obs_mask,
            )
        return encoded

    def fuse(
        self, encoded: dict[str, torch.Tensor], modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Join final sequence positions, tabular features and availability, in that order."""
        tabular = encoded["tabular"]
        batch_size = tabular.shape[0]
        seq_len = self.faithful_seq_len
        d_model = self.d_model
        if modality_mask is None:
            modality_mask = tabular.new_ones(batch_size, len(self.active_modalities))
        mask = modality_mask.to(device=tabular.device, dtype=tabular.dtype)
        sequences: list[torch.Tensor] = []
        for seq_index, modality in enumerate(self.active_modalities[1:]):
            embedding = encoded[modality]
            if embedding.ndim != 2 or embedding.shape[1] != seq_len * d_model:
                raise ValueError(
                    f"{modality} embedding must have shape (B, {seq_len * d_model}), the "
                    f"flattened mTAND sequence, got {tuple(embedding.shape)}."
                )
            # Restore the sequence positions
            sequence = _flat_to_seq(embedding, seq_len, d_model)
            # Zero missing sequences before routing
            sequences.append(sequence * mask[:, seq_index + 1].view(1, batch_size, 1))

        hidden_list, aux_loss, gate_weights = self.faithful_encoder(sequences, train=self.training)
        pooled = self.faithful_aggregate(hidden_list)  # [B, n_seq * d]
        seq_mask = mask[:, 1:]  # [B, n_seq]
        return torch.cat([pooled, tabular, seq_mask], dim=1), aux_loss, gate_weights

    def predict(self, fused: torch.Tensor) -> torch.Tensor:
        """Convert the combined head input to one binary prediction logit per row."""
        return self.faithful_head(fused).squeeze(-1)

    def forward(
        self, batch: dict[str, torch.Tensor], return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode the batch and predict logits, optionally returning routing details."""
        encoded = self.encode(batch)
        modality_mask = self._modality_mask(batch, encoded)
        fused, aux_loss, gate_weights = self.fuse(encoded, modality_mask)
        logits = self.predict(fused)

        if not return_aux:
            return logits
        return {"logits": logits, "aux_loss": aux_loss, "gate_weights": gate_weights}

    def _modality_mask(
        self, batch: Mapping[str, torch.Tensor], encoded: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Read modality availability from the batch in the configured order."""
        reference = encoded[self.active_modalities[0]]
        device = reference.device
        if not self.use_modality_mask:
            return reference.new_ones(reference.shape[0], len(self.active_modalities))

        if self.ts_modality_enabled:
            ts_lengths = batch["ts_lengths"].to(device=device)
            available = [
                tabular_available(batch, device=device), sequence_available(batch, "ecg", device=device),
                sequence_available(batch, "cxr", device=device), ts_lengths.gt(0),
            ]
        else:
            available = [
                modality_available(batch, modality, device=device)
                for modality in self.active_modalities
            ]
        return torch.stack(available, dim=-1).to(dtype=reference.dtype)


def _mtand_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read mTAND settings and require unpooled sequence outputs."""
    mtand_cfg = cfg.get("mtand", {})
    if not isinstance(mtand_cfg, Mapping):
        mtand_cfg = {}
    enabled = bool(mtand_cfg.get("enabled", False))
    pool = str(mtand_cfg.get("pool", "none"))
    settings = {
        "mtand_tt_max": int(mtand_cfg.get("tt_max", 64)),
        "mtand_embed_time": int(mtand_cfg.get("embed_time", 64)),
        "mtand_num_heads": int(mtand_cfg.get("num_heads", 8)),
        "mtand_time_scale": float(mtand_cfg.get("time_scale", 90.0)),
    }

    if not enabled:
        raise ValueError(
            "fusemoe requires mtand.enabled: true, because experts route mTAND sequences "
            "inside the encoder. The static model without mTAND lives in src/models/smp.py: "
            "use fusion_method 'smp'."
        )
    if pool != "none":
        raise ValueError(
            "fusemoe requires mtand.pool: none so ECG, CXR and time series reach the encoder "
            f"as reference-grid sequences, got {pool!r}. Remove the key from the config or "
            "--set override."
        )
    return settings


def _ts_modality_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Read raw time-series settings, leaving the modality disabled when absent."""
    raw = cfg.get("ts_modality")
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"ts_modality must be a mapping with an 'enabled' bool; got {raw!r}.")
    return {
        "ts_modality_enabled": bool(raw.get("enabled", False)),
        "ts_modality_concat_mask": bool(raw.get("concat_mask", True)),
    }


def build_fusemoe_model_from_config(
    cfg: Mapping[str, Any], feature_dims: Mapping[str, int], output_dim: int = 1,
) -> SparseMoEModel:
    """Create the temporal model from model settings and dataset feature widths."""
    output_mode = str(cfg.get("output_mode", "log_prob"))
    router_type = str(cfg.get("router_type", "joint"))
    mtand_settings = _mtand_settings(cfg)
    ts_settings = _ts_modality_settings(cfg)
    modality_names = list(cfg.get("modality_names", MODALITIES[:3]))
    active_modalities = list(MODALITIES if ts_settings["ts_modality_enabled"] else modality_names)
    model_feature_dims = dict(feature_dims)
    ts_dim = feature_dims.get("ts")
    return SparseMoEModel(
        tabular_dim=(
            int(model_feature_dims["tabular"])
            if "tabular" in model_feature_dims
            else int(model_feature_dims.get("static", 0))
        ),
        ecg_dim=int(model_feature_dims["ecg"]) if "ecg" in model_feature_dims else None,
        cxr_dim=int(model_feature_dims["cxr"]) if "cxr" in model_feature_dims else None,
        ts_dim=int(ts_dim) if ts_dim is not None else None, d_model=int(cfg.get("d_model", 128)),
        output_dim=int(cfg.get("output_dim", output_dim)),
        dropout=float(cfg.get("dropout", 0.0)), num_experts=int(cfg.get("num_experts", 7)),
        moe_hidden_size=int(cfg.get("moe_hidden_size", 256)),
        top_k=int(cfg.get("top_k", 2)), disjoint_top_k=int(cfg.get("disjoint_top_k", 1)),
        router_type=router_type, gating=str(cfg.get("gating", "softmax")),
        activation=str(cfg.get("activation", "gelu")), output_mode=output_mode,
        loss_coef=float(cfg.get("loss_coef", 1e-2)),
        use_modality_mask=bool(cfg.get("use_modality_mask", True)),
        router_score=str(cfg.get("router_score", "dot")),
        use_shared_expert=bool(cfg.get("use_shared_expert", False)),
        shared_expert_weight=(
            None if cfg.get("shared_expert_weight") is None else float(cfg["shared_expert_weight"])
        ),
        **mtand_settings, **ts_settings,
        num_layers=int(cfg.get("num_layers", 1)), num_heads=int(cfg.get("num_heads", 8)),
        ffn_dim=(int(cfg["ffn_dim"]) if cfg.get("ffn_dim") is not None else None),
        embed_dropout=float(cfg.get("embed_dropout", 0.0)),
        feature_dims=model_feature_dims, modality_names=active_modalities,
    )
