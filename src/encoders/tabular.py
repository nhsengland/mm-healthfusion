from __future__ import annotations

import torch
from torch import nn


class TabularEncoder(nn.Module):
    """Encode flat or sequence-shaped tabular features into a shared dimension."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(d_model, input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        tab_feats: torch.Tensor,
        tab_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tab_feats.ndim == 2:
            return self.network(tab_feats)
        if tab_feats.ndim != 3:
            raise ValueError("tab_feats must have shape (B, F) or (B, T, F).")

        encoded = self.network(tab_feats)
        if tab_lengths is None:
            return encoded.mean(dim=1)

        time_index = torch.arange(
            encoded.shape[1],
            device=encoded.device,
        ).unsqueeze(0)
        valid = time_index < tab_lengths.to(device=encoded.device).unsqueeze(1)
        valid = valid.unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * valid).sum(dim=1)
        normaliser = valid.sum(dim=1).clamp_min(1.0)
        return pooled / normaliser
