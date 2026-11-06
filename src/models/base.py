from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from src.data.datasets import modality_prefix
from src.encoders import CXREncoder, ECGEncoder, NotesEncoder, TabularEncoder

MODALITY_ORDER = (
    "tabular", "ecg", "cxr", "static", "ts_prev", "ts_curr", "ecg_curr", "cxr_prev", "cxr_curr",
    "notes_prev", "ts", "notes",
)
ENCODER_CLASSES = {
    "tabular": TabularEncoder, "ecg": ECGEncoder, "cxr": CXREncoder,
    "notes": NotesEncoder, "ts": TabularEncoder,
}


def canonical_modalities(modalities: Sequence[str]) -> tuple[str, ...]:
    """Check modality names and return them in the order defined by MODALITY_ORDER."""
    names = [str(name) for name in modalities]
    if not names:
        raise ValueError("A model needs at least one modality.")
    unknown = sorted(set(names) - set(MODALITY_ORDER))
    if unknown:
        raise ValueError(f"Unknown modalities {unknown}.")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate modalities are not supported: {names}.")
    selected = set(names)
    return tuple(name for name in MODALITY_ORDER if name in selected)


def encoder_name(modality: str) -> str:
    """Map static to tabular and remove previous or current visit suffixes from other names."""
    if modality == "static":
        return "tabular"
    return modality.removesuffix("_prev").removesuffix("_curr")


def require_positive_widths(feature_dims: Mapping[str, int]) -> None:
    """Check that each modality has a positive feature width."""
    for modality in canonical_modalities(feature_dims):
        width = int(feature_dims[modality])
        if width <= 0:
            raise ValueError(f"Modality {modality!r} has non-positive feature width {width}.")


def resolve_feature_dims(
    feature_dims: Mapping[str, int] | None = None, tabular_dim: int | None = None,
    ecg_dim: int | None = None, cxr_dim: int | None = None,
) -> dict[str, int]:
    """Validate feature widths and return them in model order.

    Use the individual tabular, ECG and CXR widths when ``feature_dims`` is not supplied.
    """
    if feature_dims is None:
        if None in (tabular_dim, ecg_dim, cxr_dim):
            raise ValueError("feature_dims or all legacy modality dimensions are required.")
        feature_dims = {"tabular": tabular_dim, "ecg": ecg_dim, "cxr": cxr_dim}
    dims = {name: int(feature_dims[name]) for name in canonical_modalities(feature_dims)}
    require_positive_widths(dims)
    return dims


class BaseMultimodalModel(nn.Module, ABC):
    """Encode each modality in model order, then combine features and predict outputs.

    Subclasses implement ``fuse`` to combine features and ``predict`` to produce outputs.
    """

    def __init__(
        self, tabular_dim: int | None = None, ecg_dim: int | None = None,
        cxr_dim: int | None = None, d_model: int = 128, output_dim: int = 1,
        dropout: float = 0.0, feature_dims: Mapping[str, int] | None = None,
        modality_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        dims = resolve_feature_dims(feature_dims, tabular_dim, ecg_dim, cxr_dim)
        requested = modality_names or tuple(dims)
        self.modality_names = list(canonical_modalities(requested))
        self.feature_dims = {name: dims[name] for name in self.modality_names}
        for modality in self.modality_names:
            family = encoder_name(modality)
            setattr(
                self, f"{modality}_encoder",
                ENCODER_CLASSES[family](self.feature_dims[modality], d_model, dropout=dropout),
            )
        self.output_dim = output_dim

    def modality_encoder(self, modality: str) -> nn.Module:
        """Return the encoder created for a modality."""
        return getattr(self, f"{modality}_encoder")

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode batch features using any supplied sequence lengths, keeping model order."""
        encoded = {}
        for modality in self.modality_names:
            prefix = modality_prefix(modality)
            feats = batch[f"{prefix}_feats"]
            lengths = batch.get(f"{prefix}_lengths")
            encoder = self.modality_encoder(modality)
            if isinstance(encoder, TabularEncoder) and feats.ndim == 2:
                # Treat flat tabular features as a sequence with one record per row
                if lengths is None:
                    lengths = torch.ones(feats.shape[0], dtype=torch.long, device=feats.device)
                feats = feats.unsqueeze(1)
            encoded[modality] = encoder(feats, lengths)
        return encoded

    @abstractmethod
    def fuse(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine the encoded modality features."""
        ...

    @abstractmethod
    def predict(self, fused: torch.Tensor) -> torch.Tensor:
        """Turn the combined features into model outputs."""
        ...

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch, combine its modality features and predict outputs."""
        return self.predict(self.fuse(self.encode(batch)))
