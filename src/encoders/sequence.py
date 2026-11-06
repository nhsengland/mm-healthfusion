from __future__ import annotations

import torch
from torch import nn


class MaskedSequenceEncoder(nn.Module):
    """Project variable-length embedding sequences and pool valid records."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 3:
            raise ValueError("feats must have shape (B, T, F).")
        projected = self.projection(feats)
        time_index = torch.arange(projected.shape[1], device=projected.device).unsqueeze(0)
        valid = time_index < lengths.to(device=projected.device).unsqueeze(1)
        valid = valid.unsqueeze(-1).to(dtype=projected.dtype)
        pooled = (projected * valid).sum(dim=1)
        return pooled / valid.sum(dim=1).clamp_min(1.0)
