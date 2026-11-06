from src.encoders.sequence import MaskedSequenceEncoder


class NotesEncoder(MaskedSequenceEncoder):
    """Project variable-length note embeddings into the shared representation."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(input_dim=input_dim, d_model=d_model, dropout=dropout)
