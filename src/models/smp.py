"""Predict from static modality features by combining expert probabilities.

Encode each modality, select experts for each row and combine their predictions. An optional
shared expert also processes every row. The combined log probabilities give a binary logit.

Train the static model from the repository root:

    python -m src.train --config config/smp.yaml
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, NamedTuple

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F

from src.data.collate import modality_available
from src.models.base import BaseMultimodalModel

MODALITIES = ("tabular", "ecg", "cxr")

_PROTOTYPE_INIT_STD = 1.0
_LOG_PROB_MIX_EPS = 1e-6
_COSINE_TEMPERATURE_INIT = 1.0 / 0.07
_COSINE_TEMPERATURE_MAX = 100.0
_COSINE_NORM_EPS = 1e-6
_NOISE_EPSILON = 1e-2
_DOT_ROUTER_INIT_STD = 0.0

RouterType = Literal["joint", "permod", "disjoint", "combination"]
GatingType = Literal["softmax", "laplace", "gaussian"]
RouterScore = Literal["dot", "cosine"]
ActivationType = Literal["relu", "gelu"]
OutputMode = Literal["log_prob"]


@dataclass(frozen=True)
class SparseMoEConfig:
    """Settings for selecting experts and combining their log probabilities.

    The joint router accepts unequal modality widths through ``modality_input_sizes``.
    Combination routing can train expert selection to match the available modalities.
    """

    input_size: int
    output_size: int
    num_experts: int
    hidden_size: int = 256
    top_k: int = 2
    disjoint_top_k: int = 1
    router_type: RouterType = "joint"
    gating: GatingType = "softmax"
    num_modalities: int = 3
    dropout: float = 0.0
    activation: ActivationType = "gelu"
    output_mode: OutputMode = "log_prob"
    combination_router_loss_weight: float = 0.0
    combination_router_supervision: bool = True
    combination_supervise_full_modality: bool = False
    combination_balance_exclusion: bool = False
    router_score: RouterScore = "dot"
    use_shared_expert: bool = False
    shared_expert_weight: float | None = None
    modality_names: tuple[str, ...] = MODALITIES
    modality_input_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ExpertName:
    """Name an expert in training and evaluation summaries."""

    name: str


def _combination_expert_modalities(num_modalities: int) -> tuple[tuple[str, ...], ...]:
    """List non-empty modality combinations, largest groups first.

    Groups of the same size follow the order in ``MODALITIES``.
    """
    modalities = MODALITIES[:num_modalities]
    all_combinations: list[tuple[str, ...]] = []
    for size in range(num_modalities, 0, -1):
        all_combinations.extend(tuple(comb) for comb in combinations(modalities, size))
    return tuple(all_combinations)


def _combination_expert_names(num_experts: int, num_modalities: int) -> tuple[str, ...]:
    """Name combination experts first, then any extra experts.

    Extra experts use ``buffer_0``, ``buffer_1`` and so on. Names follow the gate column order
    used in exported summaries.
    """
    combination = tuple(
        "_".join(modality_group)
        for modality_group in _combination_expert_modalities(num_modalities)
    )
    buffers = tuple(f"buffer_{index}" for index in range(num_experts - len(combination)))
    return combination + buffers


class SparseDispatcher:
    """Send rows to selected experts and combine their weighted outputs.

    Positive values in ``[batch, num_experts]`` gates select experts and weight their outputs.
    Dispatch groups rows by expert. Combining outputs restores batch order and mixes probabilities.
    """

    def __init__(
        self, num_experts: int, gates: torch.Tensor, output_mode: OutputMode = "log_prob",
    ) -> None:
        if gates.ndim != 2:
            raise ValueError("gates must have shape (batch_size, num_experts).")
        if gates.shape[1] != num_experts:
            raise ValueError("gates second dimension must match num_experts.")
        if output_mode != "log_prob":
            raise ValueError("output_mode must be 'log_prob'.")

        nonzero = torch.nonzero(gates > 0, as_tuple=False)
        if nonzero.numel() == 0:
            raise ValueError("SparseDispatcher requires at least one non-zero gate.")

        # Group assignments by expert and retain the original batch indices
        sorted_order = torch.argsort(nonzero[:, 1])
        nonzero = nonzero[sorted_order]
        self.num_experts = num_experts
        self.gates = gates
        self.output_mode = output_mode
        self.expert_index = nonzero[:, 1]
        self.batch_index = nonzero[:, 0]
        self.part_sizes = (gates > 0).sum(dim=0).tolist()
        self.nonzero_gates = gates[self.batch_index, self.expert_index].unsqueeze(-1)

    def dispatch(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Group the selected input rows into one tensor per expert.

        Tensors follow expert order and have shape ``[expert_batch_size, input_size]``.
        """
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape (batch_size, input_dim).")
        if inputs.shape[0] != self.gates.shape[0]:
            raise ValueError("inputs batch size must match gates batch size.")

        expanded_inputs = inputs.index_select(0, self.batch_index)
        return tuple(torch.split(expanded_inputs, self.part_sizes, dim=0))

    def combine(
        self, expert_outputs: list[torch.Tensor], return_probs: bool = False,
    ) -> torch.Tensor:
        """Mix expert probabilities using their gate weights in batch order.

        Return ``[batch, output_size]`` log probabilities unless ``return_probs`` is set.
        Keep probabilities when a shared expert will be mixed in later.
        """
        if len(expert_outputs) != self.num_experts:
            raise ValueError("expert_outputs length must match num_experts.")
        if not expert_outputs:
            raise ValueError("expert_outputs must not be empty.")

        output_shapes = {tuple(output.shape[1:]) for output in expert_outputs}
        if len(output_shapes) != 1:
            raise ValueError("all expert outputs must have the same non-batch shape.")

        stitched = torch.cat(expert_outputs, dim=0)
        if stitched.shape[0] != self.nonzero_gates.shape[0]:
            raise ValueError("expert output rows must match dispatched rows.")

        stitched = stitched.exp()
        stitched = stitched * self.nonzero_gates.to(dtype=stitched.dtype)
        combined = stitched.new_zeros(self.gates.shape[0], stitched.shape[-1])
        combined = combined.index_add(0, self.batch_index, stitched)

        if not return_probs:
            # Keep probabilities positive before taking their logarithm
            eps = torch.finfo(combined.dtype).tiny
            combined = combined.clamp_min(eps).log()

        return combined


def _activation_layer(activation: ActivationType) -> nn.Module:
    """Create the requested expert activation function."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'relu' or 'gelu'.")


class SparseLogProbExpertMLP(nn.Module):
    """Apply two linear layers and return log probabilities."""

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, dropout: float = 0.0,
        activation: ActivationType = "gelu",
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation_layer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


class RouterLogits(NamedTuple):
    """Hold router scores before and after noise, along with the noise standard deviation.

    Router supervision uses ``clean_logits``. Expert selection and load estimates use
    ``noisy_logits``. During evaluation, both fields reference the same tensor.
    """

    clean_logits: torch.Tensor
    noisy_logits: torch.Tensor
    noise_stddev: torch.Tensor


class SparseMoE(nn.Module):
    """Select experts for each row and combine their probabilities.

    ``joint`` routes all modalities together. ``permod`` selects from the same experts separately
    for each modality. ``disjoint`` gives each modality its own experts. ``combination`` assigns
    experts to modality combinations and can train selection to match the available inputs.
    """

    def __init__(self, config: SparseMoEConfig) -> None:
        super().__init__()
        self._validate_config(config)

        self.config = config
        self.num_experts = config.num_experts
        self.output_size = config.output_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.top_k = config.top_k
        self.disjoint_top_k = config.disjoint_top_k
        self.router_type = config.router_type
        self.num_modalities = config.num_modalities
        self.gating = config.gating
        self.output_mode = config.output_mode
        self.modality_input_size = config.input_size // config.num_modalities
        if config.modality_input_sizes is None:
            self.modality_input_sizes = tuple(
                self.modality_input_size for _ in range(config.num_modalities)
            )
        else:
            self.modality_input_sizes = tuple(config.modality_input_sizes)
        self.combination_router_loss_weight = config.combination_router_loss_weight
        self.combination_router_supervision = config.combination_router_supervision
        self.combination_supervise_full_modality = config.combination_supervise_full_modality
        self.combination_balance_exclusion = config.combination_balance_exclusion
        self.router_score = config.router_score
        self.use_shared_expert = config.use_shared_expert
        self.shared_expert_weight = self._resolve_shared_expert_weight(config)

        # Extra experts have no modality combination or supervision target
        self.combination_expert_modalities = _combination_expert_modalities(config.num_modalities)
        self.combination_expert_indices = tuple(
            tuple(MODALITIES.index(modality) for modality in modality_group)
            for modality_group in self.combination_expert_modalities
        )
        self.register_buffer(
            "combination_binary_weights",
            torch.tensor([2**index for index in range(config.num_modalities)], dtype=torch.long),
            persistent=False,
        )
        code_to_expert = torch.full((2**config.num_modalities,), fill_value=-1, dtype=torch.long)
        for expert_index, indices in enumerate(self.combination_expert_indices):
            code = sum(2**index for index in indices)
            code_to_expert[code] = expert_index
        self.register_buffer("combination_code_to_expert", code_to_expert, persistent=False)

        if config.router_type == "joint":
            self.expert_structure = "shared"
            self.expert_subsets = tuple(ExpertName("shared") for _ in range(config.num_experts))
        elif config.router_type == "permod":
            self.expert_structure = "subset"
            self.expert_subsets = tuple(
                ExpertName(f"{modality}_expert_{expert_index}")
                for modality in config.modality_names
                for expert_index in range(config.num_experts)
            )
        elif config.router_type == "combination":
            self.expert_structure = "subset"
            self.expert_subsets = tuple(
                ExpertName(name)
                for name in _combination_expert_names(config.num_experts, config.num_modalities)
            )
        else:
            experts_per_modality = config.num_experts // config.num_modalities
            self.expert_structure = "subset"
            self.expert_subsets = tuple(
                ExpertName(f"{modality}_expert_{expert_index}")
                for modality in config.modality_names
                for expert_index in range(experts_per_modality)
            )

        if self.router_type in ("joint", "combination"):
            self.w_gate = self._new_gate_weight(config.input_size, config.num_experts)
            self.w_noise = nn.Parameter(torch.zeros(config.input_size, config.num_experts))
            self.experts = nn.ModuleList([
                SparseLogProbExpertMLP(
                    input_size=config.input_size, output_size=config.output_size,
                    hidden_size=config.hidden_size, dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_experts)
            ])
        elif self.router_type == "permod":
            self.w_gate = nn.ParameterList([
                self._new_gate_weight(self.modality_input_size, config.num_experts)
                for _ in range(config.num_modalities)
            ])
            self.w_noise = nn.ParameterList([
                nn.Parameter(torch.zeros(self.modality_input_size, config.num_experts))
                for _ in range(config.num_modalities)
            ])
            self.experts = nn.ModuleList([
                SparseLogProbExpertMLP(
                    input_size=self.modality_input_size, output_size=config.output_size,
                    hidden_size=config.hidden_size, dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.num_experts)
            ])
        else:
            experts_per_modality = config.num_experts // config.num_modalities
            self.w_gate = nn.ParameterList([
                self._new_gate_weight(self.modality_input_size, experts_per_modality)
                for _ in range(config.num_modalities)
            ])
            self.w_noise = nn.ParameterList([
                nn.Parameter(torch.zeros(self.modality_input_size, experts_per_modality))
                for _ in range(config.num_modalities)
            ])
            self.experts = nn.ModuleList([
                nn.ModuleList([
                    SparseLogProbExpertMLP(
                        input_size=self.modality_input_size, output_size=config.output_size,
                        hidden_size=config.hidden_size, dropout=config.dropout,
                        activation=config.activation,
                    )
                    for _ in range(experts_per_modality)
                ])
                for _ in range(config.num_modalities)
            ])

        self.backbone_experts = self._new_backbone_experts(config)
        self.router_log_temperature = self._new_router_temperature(config)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    @staticmethod
    def _resolve_shared_expert_weight(config: SparseMoEConfig) -> float:
        """Choose the shared expert's probability mixture weight.

        Default to ``1 / (1 + k)`` with ``disjoint_top_k`` for disjoint routing, otherwise ``top_k``.
        """
        if config.shared_expert_weight is not None:
            return float(config.shared_expert_weight)
        effective_top_k = (
            config.disjoint_top_k if config.router_type == "disjoint" else config.top_k
        )
        return 1.0 / (1.0 + effective_top_k)

    def _new_backbone_experts(self, config: SparseMoEConfig) -> nn.ModuleList | None:
        """Create shared experts that process every row when enabled.

        Joint and combination routing use one shared expert. Other routers use one per modality.
        Shared experts have no gate columns or balancing loss.
        """
        if not config.use_shared_expert:
            return None

        if config.router_type in ("joint", "combination"):
            widths = [config.input_size]
        else:
            widths = [self.modality_input_size for _ in range(config.num_modalities)]

        return nn.ModuleList([
            SparseLogProbExpertMLP(
                input_size=width, output_size=config.output_size,
                hidden_size=config.hidden_size, dropout=config.dropout,
                activation=config.activation,
            )
            for width in widths
        ])

    def _new_router_temperature(
        self, config: SparseMoEConfig,
    ) -> nn.Parameter | nn.ParameterList | None:
        """Create a learned cosine score scale for each router.

        Joint and combination routing share one scale. Other routers use one per modality.
        Dot scoring needs no scale. Initialisation is constant and consumes no random draws.
        """
        if config.router_score != "cosine":
            return None

        log_init = math.log(_COSINE_TEMPERATURE_INIT)
        if config.router_type in ("joint", "combination"):
            return nn.Parameter(torch.tensor(log_init))

        return nn.ParameterList(
            [nn.Parameter(torch.tensor(log_init)) for _ in range(config.num_modalities)],
        )

    def _cosine_logits(
        self, x: torch.Tensor, w_gate: torch.Tensor, modality_index: int | None,
    ) -> torch.Tensor:
        """Score experts by cosine similarity with a learned, capped scale.

        Limit small denominators so zero inputs keep finite gradients.
        """
        temperature = self.router_log_temperature
        if modality_index is not None:
            temperature = temperature[modality_index]
        temperature = temperature.clamp(max=math.log(_COSINE_TEMPERATURE_MAX)).exp()
        x_unit = x / x.norm(dim=-1, keepdim=True).clamp_min(_COSINE_NORM_EPS)
        w_unit = w_gate / w_gate.norm(dim=0, keepdim=True).clamp_min(_COSINE_NORM_EPS)
        return (x_unit @ w_unit) * temperature

    def _new_gate_weight(self, fan_in: int, num_experts: int) -> nn.Parameter:
        """Create router weights using the chosen starting standard deviation."""
        weight = torch.empty(fan_in, num_experts)
        nn.init.normal_(weight, std=self._gate_init_std())
        return nn.Parameter(weight)

    def _gate_init_std(self) -> float:
        """Choose the starting standard deviation for router weights.

        Dot-product weights start at zero. Cosine and distance routers use a standard deviation
        of 1. Both paths consume random draws during initialisation.
        """
        if self.gating != "softmax" or self.router_score == "cosine":
            return _PROTOTYPE_INIT_STD

        return _DOT_ROUTER_INIT_STD

    @staticmethod
    def _validate_config(config: SparseMoEConfig) -> None:
        """Check routing settings before allocating model parameters."""
        if config.input_size <= 0:
            raise ValueError("input_size must be greater than 0.")
        if config.output_size <= 0:
            raise ValueError("output_size must be greater than 0.")
        if config.num_experts <= 0:
            raise ValueError("num_experts must be greater than 0.")
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")
        if config.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if config.top_k > config.num_experts:
            raise ValueError("top_k must not exceed num_experts.")
        if config.disjoint_top_k <= 0:
            raise ValueError("disjoint_top_k must be greater than 0.")
        if config.router_type not in ("joint", "permod", "disjoint", "combination"):
            raise ValueError("router_type must be 'joint', 'permod', 'disjoint', or 'combination'.")
        if config.gating not in ("softmax", "laplace", "gaussian"):
            raise ValueError("gating must be 'softmax', 'laplace', or 'gaussian'.")
        if config.activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'.")
        if config.output_mode != "log_prob":
            raise ValueError("output_mode must be 'log_prob'.")
        if config.dropout < 0.0 or config.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if config.combination_router_loss_weight < 0.0:
            raise ValueError("combination_router_loss_weight must be non-negative.")
        if config.router_score not in ("dot", "cosine"):
            raise ValueError("router_score must be 'dot' or 'cosine'.")
        if config.router_score == "cosine" and config.gating != "softmax":
            raise ValueError(
                "router_score='cosine' requires gating='softmax'; under "
                "laplace/gaussian w_gate holds cdist prototypes, not a projection."
            )
        if config.shared_expert_weight is not None and not 0.0 < config.shared_expert_weight < 1.0:
            raise ValueError("shared_expert_weight must be in (0, 1); it is a mixture weight.")

        if config.modality_input_sizes is not None:
            sizes = tuple(config.modality_input_sizes)
            if len(sizes) != config.num_modalities:
                raise ValueError(
                    "modality_input_sizes must have length num_modalities "
                    f"({config.num_modalities}), got {len(sizes)}."
                )
            if any(size <= 0 for size in sizes):
                raise ValueError("every modality_input_sizes entry must be > 0.")
            if sum(sizes) != config.input_size:
                raise ValueError(
                    "input_size must equal sum(modality_input_sizes): "
                    f"{config.input_size} != {sum(sizes)}."
                )
            # Unequal modality widths require joint routing
            if len(set(sizes)) > 1 and config.router_type != "joint":
                raise ValueError(
                    f"router_type='{config.router_type}' requires equal per-modality widths; "
                    "unequal modality_input_sizes (un-pooled mTAND) are only supported by the "
                    "'joint' router, which routes over the concatenated modality vector."
                )

        if config.router_type in ("permod", "disjoint", "combination"):
            if config.num_modalities <= 0:
                raise ValueError("num_modalities must be greater than 0.")
            if len(config.modality_names) != config.num_modalities:
                raise ValueError("modality_names length must match num_modalities.")
            if config.input_size % config.num_modalities != 0:
                raise ValueError(
                    "input_size must be divisible by num_modalities for "
                    "permod/disjoint/combination routers."
                )

        if config.router_type == "combination":
            if config.num_modalities > len(MODALITIES):
                raise ValueError(
                    "combination router currently supports only the canonical "
                    f"{MODALITIES} modality set."
                )
            combination_experts = (2**config.num_modalities) - 1
            if config.num_experts < combination_experts:
                raise ValueError(
                    "router_type='combination' requires num_experts>="
                    f"{combination_experts} for {config.num_modalities} modalities: one "
                    "expert per non-empty modality combination. Experts beyond that are "
                    "unaddressed BUFFER experts, which Flex-MoE ships (16 experts for 4 "
                    "modalities, so 15 combinations + 1 buffer)."
                )

        if config.router_type == "disjoint":
            if config.num_experts % config.num_modalities != 0:
                raise ValueError(
                    "num_experts must be divisible by num_modalities for disjoint routers.",
                )
            experts_per_modality = config.num_experts // config.num_modalities
            if config.disjoint_top_k > experts_per_modality:
                raise ValueError("disjoint_top_k must not exceed num_experts // num_modalities.")

    def _modality_inputs(self, x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """Return one input tensor per modality in configured order.

        Accept a tensor list or split concatenated inputs by modality width.
        """
        if isinstance(x, torch.Tensor):
            if x.ndim != 2 or x.shape[1] != self.input_size:
                raise ValueError(f"input tensor must have shape (B, {self.input_size}).")
            return list(x.split(self.modality_input_sizes, dim=1))

        if len(x) != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modality tensors.")

        batch_size = x[0].shape[0]
        for modality_input, expected_width in zip(x, self.modality_input_sizes, strict=True):
            if (
                modality_input.ndim != 2
                or modality_input.shape[0] != batch_size
                or modality_input.shape[1] != expected_width
            ):
                raise ValueError(
                    "each modality tensor must have shape (B, width), with "
                    f"widths {self.modality_input_sizes} in modality order."
                )
        return x

    def _joint_input(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Join modality features into one tensor, preserving modality order."""
        if isinstance(x, torch.Tensor):
            if x.ndim != 2 or x.shape[1] != self.input_size:
                raise ValueError(f"input tensor must have shape (B, {self.input_size}).")
            return x

        return torch.cat(self._modality_inputs(x), dim=1)

    def cv_squared(self, values: torch.Tensor) -> torch.Tensor:
        """Measure relative variation as variance divided by squared mean.

        Compute in float with a small denominator offset. Zero or one value returns zero.
        """
        if values.numel() <= 1:
            return values.new_zeros(())
        eps = 1e-10
        values = values.float()
        return values.var(unbiased=False) / (values.mean().pow(2) + eps)

    def _balance_loss(
        self, gates: torch.Tensor, load: torch.Tensor, loss_coef: float,
    ) -> torch.Tensor:
        """Penalise uneven expert weights and loads, scaled by loss_coef."""
        return (self.cv_squared(gates.sum(dim=0)) + self.cv_squared(load)) * loss_coef

    def _balance_term(self, gates: torch.Tensor, row_load: torch.Tensor) -> torch.Tensor:
        """Measure uneven expert weights and loads after summing selected rows.

        Loads stay separate by row until the balancing groups are selected.
        """
        return self.cv_squared(gates.sum(dim=0)) + self.cv_squared(row_load.sum(dim=0))

    def _combination_balance_loss(
        self, gates: torch.Tensor, row_load: torch.Tensor, is_full_modality: torch.Tensor,
        loss_coef: float,
    ) -> torch.Tensor:
        """Balance rows with all modalities separately from incomplete rows.

        Complete rows use all experts. Incomplete rows exclude column 0, the expert for all
        modalities, and keep every other expert. Empty groups contribute zero. Add both losses
        before applying ``loss_coef``.
        """
        if bool(is_full_modality.all()):
            return self._balance_term(gates, row_load) * loss_coef

        partial = ~is_full_modality
        total = self._balance_term(gates[is_full_modality], row_load[is_full_modality])
        total = total + self._balance_term(gates[partial][:, 1:], row_load[partial][:, 1:])
        return total * loss_coef

    @staticmethod
    def _gates_to_load(gates: torch.Tensor) -> torch.Tensor:
        """Count the rows assigned to each expert."""
        return (gates > 0).sum(dim=0).to(dtype=gates.dtype)

    def _prob_in_top_k(
        self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """Estimate each expert's chance of selection after adding noise.

        The load-balancing loss can differentiate through this estimate.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + k
        threshold_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in).unsqueeze(1)
        is_in = noisy_values > threshold_if_in
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out).unsqueeze(1)
        normal = Normal(self.mean, self.std)
        noise_stddev = noise_stddev.clamp_min(1e-10)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def _get_logits(
        self, x: torch.Tensor, train: bool, modality_index: int | None = None,
    ) -> RouterLogits:
        """Return router scores before and after noise, and the noise standard deviation."""
        if modality_index is None:
            w_gate = self.w_gate
            w_noise = self.w_noise
        else:
            w_gate = self.w_gate[modality_index]
            w_noise = self.w_noise[modality_index]

        if self.gating == "softmax":
            if self.router_score == "cosine":
                clean_logits = self._cosine_logits(x, w_gate, modality_index)
            else:
                clean_logits = x @ w_gate
        elif self.gating == "laplace":
            clean_logits = -torch.cdist(x, w_gate.t())
        else:
            clean_logits = -torch.cdist(x, w_gate.t()).pow(2)

        noisy_logits = clean_logits
        noise_stddev = torch.ones_like(clean_logits)

        if train:
            raw_noise_stddev = x @ w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + _NOISE_EPSILON
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev

        return RouterLogits(clean_logits, noisy_logits, noise_stddev)

    def _top_k_gating(
        self, router_logits: RouterLogits, k: int, train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the top k experts and return their weights and total loads."""
        clean_logits = router_logits.clean_logits
        noisy_logits = router_logits.noisy_logits
        noise_stddev = router_logits.noise_stddev
        num_available_experts = noisy_logits.shape[1]

        # Keep one extra score for estimating load under routing noise
        top_logits, top_indices = noisy_logits.topk(min(k + 1, num_available_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]

        if self.gating == "softmax":
            top_k_gates = F.softmax(top_k_logits, dim=1)
        else:
            top_k_gates = torch.exp(
                top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True),
            )

        zeros = torch.zeros_like(noisy_logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if train and k < num_available_experts:
            load = self._prob_in_top_k(
                clean_values=clean_logits, noisy_values=noisy_logits, noise_stddev=noise_stddev,
                noisy_top_values=top_logits, k=k,
            ).sum(dim=0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def _combination_top_k_gating(
        self, router_logits: RouterLogits, k: int, train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the top k experts and return a load estimate for each row.

        Selection matches ``_top_k_gating``. Keeping loads separate lets the balancing loss
        group complete and incomplete rows.
        """
        clean_logits = router_logits.clean_logits
        noisy_logits = router_logits.noisy_logits
        noise_stddev = router_logits.noise_stddev
        num_available_experts = noisy_logits.shape[1]

        # Keep one extra score for estimating load under routing noise
        top_logits, top_indices = noisy_logits.topk(min(k + 1, num_available_experts), dim=1)
        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]

        if self.gating == "softmax":
            top_k_gates = F.softmax(top_k_logits, dim=1)
        else:
            top_k_gates = torch.exp(
                top_k_logits - torch.logsumexp(top_k_logits, dim=1, keepdim=True),
            )

        zeros = torch.zeros_like(noisy_logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if train and k < num_available_experts:
            # Keep loads separate by row until the balancing groups are selected
            row_load = self._prob_in_top_k(
                clean_values=clean_logits, noisy_values=noisy_logits, noise_stddev=noise_stddev,
                noisy_top_values=top_logits, k=k,
            )
        else:
            # Count assignments per row for the separate balancing groups
            row_load = (gates > 0).to(dtype=gates.dtype)

        return gates, row_load

    def noisy_top_k_gating(
        self, x: torch.Tensor | list[torch.Tensor], train: bool,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        """Select experts and return gate weights and loads.

        Training adds routing noise. Joint routing returns two tensors. Permod and disjoint
        return two lists in modality order. Combination routing is handled in ``forward``.
        """
        if self.router_type == "joint":
            embeddings = self._joint_input(x)
            router_logits = self._get_logits(embeddings, train=train)
            gates, load = self._top_k_gating(router_logits, k=self.top_k, train=train)
            return gates, load

        if self.router_type == "combination":
            raise RuntimeError(
                "combination router needs modality_mask and is handled in forward().",
            )

        modality_inputs = self._modality_inputs(x)
        all_gates: list[torch.Tensor] = []
        all_loads: list[torch.Tensor] = []

        for modality_index, modality_input in enumerate(modality_inputs):
            router_logits = self._get_logits(
                modality_input, train=train, modality_index=modality_index,
            )
            k = self.top_k if self.router_type == "permod" else self.disjoint_top_k
            gates, load = self._top_k_gating(router_logits, k=k, train=train)
            all_gates.append(gates)
            all_loads.append(load)

        return all_gates, all_loads

    def _observed_combination_indices(self, modality_mask: torch.Tensor) -> torch.Tensor:
        """Find each row's expert from its available modalities."""
        mask = modality_mask.to(device=self.combination_binary_weights.device, dtype=torch.bool)
        binary_weights = self.combination_binary_weights.to(device=mask.device)
        binary_codes = (mask.to(dtype=torch.long) * binary_weights).sum(dim=1)
        if torch.any(binary_codes == 0):
            raise ValueError("At least one modality must be observed for each row.")
        code_to_expert = self.combination_code_to_expert.to(device=mask.device)
        target = code_to_expert[binary_codes]
        if torch.any(target < 0):
            raise ValueError("Encountered an unknown modality-combination code.")
        return target

    def _combination_router_loss(
        self, clean_logits: torch.Tensor, modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Train router scores to select the expert for the available modalities.

        Skip rows with no modalities. Include complete rows only when
        ``combination_supervise_full_modality`` is enabled. Return zero when supervision is
        disabled, has zero weight or has no selected rows.
        """
        if not self.combination_router_supervision or self.combination_router_loss_weight == 0.0:
            return clean_logits.new_zeros(())

        # Explanation runs can hide every modality, leaving no expert to supervise
        observed = modality_mask.to(dtype=torch.bool).any(dim=1)
        modality_mask = modality_mask[observed]
        clean_logits = clean_logits[observed.to(device=clean_logits.device)]
        target = self._observed_combination_indices(modality_mask).to(device=clean_logits.device)
        row_mask = torch.ones_like(target, dtype=torch.bool)
        if not self.combination_supervise_full_modality:
            # Expert 0 represents all modalities
            row_mask = target.ne(0)

        if not bool(row_mask.any().item()):
            return clean_logits.new_zeros(())

        supervised_loss = F.cross_entropy(clean_logits[row_mask], target[row_mask])
        return self.combination_router_loss_weight * supervised_loss

    def _routed_combine(
        self, dispatcher: SparseDispatcher, expert_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Combine routed outputs, keeping probabilities when a shared expert will be mixed in."""
        return dispatcher.combine(expert_outputs, return_probs=self.backbone_experts is not None)

    def _add_shared_expert(
        self, routed: torch.Tensor, shared_input: torch.Tensor, backbone_index: int = 0,
    ) -> torch.Tensor:
        """Mix the shared expert's probabilities with the routed output.

        Take the logarithm after forming the weighted probability mixture. Return the routed
        output if no shared expert exists.
        """
        if self.backbone_experts is None:
            return routed

        backbone = self.backbone_experts[backbone_index](shared_input)
        weight = self.shared_expert_weight
        mixed = (1.0 - weight) * routed + weight * backbone.exp()
        return mixed.clamp_min(_LOG_PROB_MIX_EPS).log()

    def _forward_combination(
        self, x: torch.Tensor | list[torch.Tensor], modality_mask: torch.Tensor | None, train: bool,
        loss_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Route combined inputs and compute losses from the available modalities."""
        if modality_mask is None:
            raise ValueError("router_type='combination' requires modality_mask.")
        if modality_mask.ndim != 2 or modality_mask.shape[1] != self.num_modalities:
            raise ValueError("modality_mask must have shape (B, num_modalities) for combination.")

        embeddings = self._joint_input(x)
        if modality_mask.shape[0] != embeddings.shape[0]:
            raise ValueError("modality_mask batch size must match input batch size.")

        router_logits = self._get_logits(embeddings, train=train)
        if self.combination_balance_exclusion:
            # Split complete and incomplete rows before summing their loads
            gates, row_load = self._combination_top_k_gating(
                router_logits, k=self.top_k, train=train,
            )
            balance_loss = self._combination_balance_loss(
                gates, row_load, modality_mask.to(device=gates.device, dtype=torch.bool).all(dim=1),
                loss_coef,
            )
        else:
            gates, load = self._top_k_gating(router_logits, k=self.top_k, train=train)
            balance_loss = self._balance_loss(gates, load, loss_coef)
        router_loss = self._combination_router_loss(router_logits.clean_logits, modality_mask)
        aux_loss = balance_loss + router_loss
        dispatcher = SparseDispatcher(
            num_experts=self.num_experts, gates=gates, output_mode=self.output_mode,
        )
        expert_inputs = dispatcher.dispatch(embeddings)
        expert_outputs = [
            expert(expert_input)
            for expert, expert_input in zip(self.experts, expert_inputs, strict=True)
        ]
        output = self._add_shared_expert(
            self._routed_combine(dispatcher, expert_outputs), embeddings,
        )

        # Jointly routed outputs have no separate norm for each modality
        return output, aux_loss, gates, None

    def forward(
        self, x: torch.Tensor | list[torch.Tensor], train: bool | None = None,
        loss_coef: float = 1e-2, modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return combined outputs, routing loss, gate weights and output norms.

        Outputs have shape ``[batch, output_size]`` and the loss is a scalar. Gate weights have
        shape ``[batch, num_experts]`` except for ``permod``, which has
        ``num_modalities * num_experts`` columns in modality order. Combination gate columns
        list combination experts before extra experts.

        Permod and disjoint routing return one output norm per modality with shape
        ``[batch, num_modalities]``. Joint and combination routing return None for the norms.
        """
        if train is None:
            train = self.training

        if self.router_type == "combination":
            return self._forward_combination(
                x=x, modality_mask=modality_mask, train=train, loss_coef=loss_coef,
            )

        gates, load = self.noisy_top_k_gating(x=x, train=train)

        if self.router_type == "joint":
            if not isinstance(gates, torch.Tensor) or not isinstance(load, torch.Tensor):
                raise RuntimeError("joint router expected tensor gates and tensor load.")

            embeddings = self._joint_input(x)
            aux_loss = self._balance_loss(gates, load, loss_coef)
            dispatcher = SparseDispatcher(
                num_experts=self.num_experts, gates=gates, output_mode=self.output_mode,
            )
            expert_inputs = dispatcher.dispatch(embeddings)
            expert_outputs = [
                expert(expert_input)
                for expert, expert_input in zip(self.experts, expert_inputs, strict=True)
            ]
            output = self._add_shared_expert(
                self._routed_combine(dispatcher, expert_outputs), embeddings,
            )
            return output, aux_loss, gates, None

        if not isinstance(gates, list) or not isinstance(load, list):
            raise RuntimeError("permod/disjoint routers expected list gates and loads.")

        modality_inputs = self._modality_inputs(x)
        aux_loss = sum(
            self._balance_loss(modality_gates, modality_load, loss_coef)
            for modality_gates, modality_load in zip(gates, load, strict=True)
        )
        output: torch.Tensor | None = None

        if self.router_type == "permod":
            modality_norms: list[torch.Tensor] = []
            for index, (modality_input, modality_gates) in enumerate(
                zip(modality_inputs, gates, strict=True),
            ):
                dispatcher = SparseDispatcher(
                    num_experts=self.num_experts, gates=modality_gates,
                    output_mode=self.output_mode,
                )
                expert_inputs = dispatcher.dispatch(modality_input)
                expert_outputs = [
                    expert(expert_input)
                    for expert, expert_input in zip(self.experts, expert_inputs, strict=True)
                ]
                modality_output = self._add_shared_expert(
                    self._routed_combine(dispatcher, expert_outputs), modality_input,
                    backbone_index=index,
                )
                modality_norms.append(torch.linalg.vector_norm(modality_output, dim=-1))
                output = modality_output if output is None else output + modality_output

            if output is None:
                raise RuntimeError("permod sparse MoE produced no output.")

            # Export columns in modality order, then expert order
            gate_tensor = torch.stack(gates, dim=1).flatten(start_dim=1)
            router_output_norms = torch.stack(modality_norms, dim=1)
            return output, aux_loss, gate_tensor, router_output_norms

        experts_per_modality = self.num_experts // self.num_modalities
        modality_norms = []
        for modality_index, (modality_input, modality_gates) in enumerate(
            zip(modality_inputs, gates, strict=True),
        ):
            dispatcher = SparseDispatcher(
                num_experts=experts_per_modality, gates=modality_gates,
                output_mode=self.output_mode,
            )
            expert_inputs = dispatcher.dispatch(modality_input)
            expert_outputs = [
                expert(expert_input)
                for expert, expert_input in zip(
                    self.experts[modality_index], expert_inputs, strict=True,
                )
            ]
            modality_output = self._add_shared_expert(
                self._routed_combine(dispatcher, expert_outputs), modality_input,
                backbone_index=modality_index,
            )
            modality_norms.append(torch.linalg.vector_norm(modality_output, dim=-1))
            output = modality_output if output is None else output + modality_output

        if output is None:
            raise RuntimeError("disjoint sparse MoE produced no output.")

        # Export columns in modality order, then expert order
        gate_tensor = torch.stack(gates, dim=1).flatten(start_dim=1)
        router_output_norms = torch.stack(modality_norms, dim=1)
        return output, aux_loss, gate_tensor, router_output_norms


class SparseMoEModel(BaseMultimodalModel):
    """Encode static inputs, mix expert probabilities and predict binary logits.

    Subtract negative-class log probabilities from positive-class ones. Optional outputs include
    the routing loss, gate weights and output norms.
    """

    supports_aux_output = True

    def __init__(
        self, tabular_dim: int | None = None, ecg_dim: int | None = None,
        cxr_dim: int | None = None, d_model: int = 128, output_dim: int = 1, dropout: float = 0.0,
        num_experts: int = 6, moe_hidden_size: int = 256, moe_output_size: int = 2, top_k: int = 2,
        disjoint_top_k: int = 1, router_type: RouterType = "joint", gating: GatingType = "softmax",
        activation: ActivationType = "gelu", output_mode: OutputMode = "log_prob",
        loss_coef: float = 1e-2, use_modality_mask: bool = True,
        combination_router_loss_weight: float = 0.0, combination_router_supervision: bool = True,
        combination_supervise_full_modality: bool = False,
        combination_balance_exclusion: bool = False, router_score: RouterScore = "dot",
        use_shared_expert: bool = False, shared_expert_weight: float | None = None,
        feature_dims: dict[str, int] | None = None,
        modality_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            tabular_dim=tabular_dim, ecg_dim=ecg_dim, cxr_dim=cxr_dim, d_model=d_model,
            output_dim=output_dim, dropout=dropout, feature_dims=feature_dims,
            modality_names=modality_names,
        )
        if output_dim != 1:
            raise ValueError("SparseMoEModel currently supports binary output_dim=1.")
        if moe_output_size != 2:
            raise ValueError("output_mode='log_prob' requires moe_output_size=2 for binary logits.")

        self.d_model = d_model
        self.output_dim = output_dim
        self.output_mode = output_mode
        self.loss_coef = loss_coef
        self.use_modality_mask = use_modality_mask
        self.router_type = router_type
        self.active_modalities = tuple(self.modality_names)
        moe_input_size = len(self.active_modalities) * d_model
        config = SparseMoEConfig(
            input_size=moe_input_size, output_size=moe_output_size, num_experts=num_experts,
            hidden_size=moe_hidden_size, top_k=top_k, disjoint_top_k=disjoint_top_k,
            router_type=router_type, gating=gating, num_modalities=len(self.active_modalities),
            dropout=dropout, activation=activation, output_mode=output_mode,
            combination_router_loss_weight=combination_router_loss_weight,
            combination_router_supervision=combination_router_supervision,
            combination_supervise_full_modality=combination_supervise_full_modality,
            combination_balance_exclusion=combination_balance_exclusion, router_score=router_score,
            use_shared_expert=use_shared_expert, shared_expert_weight=shared_expert_weight,
            modality_names=tuple(self.active_modalities),
        )
        self.fusion = SparseMoE(config)

    def fuse(
        self, encoded: dict[str, torch.Tensor], modality_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply the availability mask, route encoded modalities and return routing results."""
        if modality_mask is None:
            reference = encoded[self.active_modalities[0]]
            modality_mask = reference.new_ones(reference.shape[0], len(self.active_modalities))

        modality_inputs = self._prepare_modality_inputs(encoded, modality_mask)
        return self.fusion(
            modality_inputs, train=self.training, loss_coef=self.loss_coef,
            modality_mask=modality_mask,
        )

    def predict(self, fused: torch.Tensor) -> torch.Tensor:
        """Return one binary prediction logit per row from the combined log probabilities."""
        if fused.shape[1] != 2:
            raise ValueError("log_prob mode expects fused output with shape [B, 2].")
        return fused[:, 1] - fused[:, 0]

    def forward(
        self, batch: dict[str, torch.Tensor], return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode the batch and predict logits, optionally returning routing details."""
        encoded = self.encode(batch)
        modality_mask = self._modality_mask(batch, encoded)
        fused, aux_loss, gate_weights, router_output_norms = self.fuse(encoded, modality_mask)
        logits = self.predict(fused)

        if not return_aux:
            return logits

        aux_output = {"logits": logits, "aux_loss": aux_loss, "gate_weights": gate_weights}
        if router_output_norms is not None:
            aux_output["router_output_norms"] = router_output_norms
        return aux_output

    def _prepare_modality_inputs(
        self, encoded: Mapping[str, torch.Tensor], modality_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return modality tensors in order with the availability mask applied."""
        reference = encoded[self.active_modalities[0]]
        if modality_mask.shape != (reference.shape[0], len(self.active_modalities)):
            raise ValueError(f"modality_mask must have shape (B, {len(self.active_modalities)}).")

        mask = modality_mask.to(device=reference.device, dtype=reference.dtype)
        expected_widths = self.fusion.modality_input_sizes
        prepared = []
        for index, modality in enumerate(self.active_modalities):
            embedding = encoded[modality]
            if embedding.ndim != 2 or embedding.shape[1] != expected_widths[index]:
                raise ValueError(
                    f"{modality} embedding must have shape (B, {expected_widths[index]}).",
                )

            prepared.append(embedding * mask[:, index : index + 1])

        return prepared

    def _modality_mask(
        self, batch: Mapping[str, torch.Tensor], encoded: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Read modality availability from the batch in the configured order."""
        if not self.use_modality_mask:
            reference = encoded[self.active_modalities[0]]
            return reference.new_ones(reference.shape[0], len(self.active_modalities))

        reference = encoded[self.active_modalities[0]]
        available = [
            modality_available(batch, modality, device=reference.device)
            for modality in self.active_modalities
        ]
        return torch.stack(available, dim=-1).to(dtype=reference.dtype)


def build_smp_model_from_config(
    cfg: Mapping[str, Any], feature_dims: Mapping[str, int], output_dim: int = 1,
) -> SparseMoEModel:
    """Create the static model from model settings and dataset feature widths."""
    output_mode = str(cfg.get("output_mode", "log_prob"))
    router_type = str(cfg.get("router_type", "joint"))
    modality_names = list(cfg.get("modality_names", MODALITIES))
    default_num_experts = (2 ** len(modality_names)) - 1 if router_type == "combination" else 6
    return SparseMoEModel(
        d_model=int(cfg.get("d_model", 128)), output_dim=int(cfg.get("output_dim", output_dim)),
        dropout=float(cfg.get("dropout", 0.0)),
        num_experts=int(cfg.get("num_experts", default_num_experts)),
        moe_hidden_size=int(cfg.get("moe_hidden_size", 256)), moe_output_size=2,
        top_k=int(cfg.get("top_k", 2)), disjoint_top_k=int(cfg.get("disjoint_top_k", 1)),
        router_type=router_type, gating=str(cfg.get("gating", "softmax")),
        activation=str(cfg.get("activation", "gelu")), output_mode=output_mode,
        loss_coef=float(cfg.get("loss_coef", 1e-2)),
        use_modality_mask=bool(cfg.get("use_modality_mask", True)),
        combination_router_loss_weight=float(cfg.get("combination_router_loss_weight", 0.0)),
        combination_router_supervision=bool(cfg.get("combination_router_supervision", True)),
        combination_supervise_full_modality=bool(
            cfg.get("combination_supervise_full_modality", False),
        ),
        combination_balance_exclusion=bool(cfg.get("combination_balance_exclusion", False)),
        router_score=str(cfg.get("router_score", "dot")),
        use_shared_expert=bool(cfg.get("use_shared_expert", False)),
        shared_expert_weight=(
            None if cfg.get("shared_expert_weight") is None else float(cfg["shared_expert_weight"])
        ),
        feature_dims=dict(feature_dims), modality_names=modality_names,
    )
