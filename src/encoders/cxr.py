from __future__ import annotations

from src.encoders.sequence import MaskedSequenceEncoder


class CXREncoder(MaskedSequenceEncoder):
    """Project variable-length CXR embeddings into a shared representation."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim=input_dim, d_model=d_model, dropout=dropout)
