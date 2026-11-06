from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.base import BaseMultimodalModel


class ConcatModel(BaseMultimodalModel):
    """Join modality features in model order and predict logits with two linear layers."""

    def __init__(
        self, tabular_dim: int | None = None, ecg_dim: int | None = None,
        cxr_dim: int | None = None, d_model: int = 128, output_dim: int = 1,
        dropout: float = 0.0, feature_dims: dict[str, int] | None = None,
        modality_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            tabular_dim=tabular_dim, ecg_dim=ecg_dim, cxr_dim=cxr_dim,
            d_model=d_model, output_dim=output_dim, dropout=dropout,
            feature_dims=feature_dims, modality_names=modality_names,
        )
        fused_dim = d_model * len(self.modality_names)
        hidden_dim = max(d_model, fused_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), nn.Linear(fused_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )

    def fuse(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate encoded features in the model's modality order."""
        return torch.cat([encoded[modality] for modality in self.modality_names], dim=-1)

    def predict(self, fused: torch.Tensor) -> torch.Tensor:
        """Return logits shaped [batch] for one output, otherwise [batch, output_dim]."""
        logits = self.head(fused)
        if self.output_dim == 1:
            return logits.squeeze(-1)
        return logits


def build_concat_model_from_config(
    fusion_cfg: Mapping[str, Any], feature_dims: Mapping[str, int], output_dim: int = 1,
) -> ConcatModel:
    """Create a concat model from its settings and dataset feature widths."""
    return ConcatModel(
        d_model=int(fusion_cfg.get("d_model", 128)),
        output_dim=int(fusion_cfg.get("output_dim", output_dim)),
        dropout=float(fusion_cfg.get("dropout", 0.0)),
        feature_dims=dict(feature_dims),
        modality_names=list(fusion_cfg.get("modality_names", feature_dims.keys())),
    )
