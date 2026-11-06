from __future__ import annotations

from collections.abc import Mapping

import torch

from src.data.datasets import TABULAR_MODALITIES, modality_prefix


def _resolve_batch_size(
    batch: Mapping[str, torch.Tensor],
    batch_size: int | None,
) -> int:
    if batch_size is not None:
        return int(batch_size)
    if "label" in batch:
        return int(batch["label"].numel())
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("batch_size is required when no batched tensors are present.")


def _resolve_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device | str | None,
) -> torch.device | str | None:
    if device is not None:
        return device
    if "label" in batch:
        return batch["label"].device
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def tabular_available(
    batch: Mapping[str, torch.Tensor],
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-sample tabular availability using missingness and lengths."""
    resolved_batch_size = _resolve_batch_size(batch, batch_size)
    resolved_device = _resolve_device(batch, device)

    if "tab_missing" in batch:
        missing = batch["tab_missing"]
        if resolved_device is not None:
            missing = missing.to(device=resolved_device)
        if missing.ndim == 1:
            return missing.eq(0)
        if missing.ndim == 2:
            return missing.eq(0).any(dim=1)
        if missing.ndim == 3:
            valid = missing.eq(0)
            if "tab_lengths" in batch:
                lengths = batch["tab_lengths"]
                if resolved_device is not None:
                    lengths = lengths.to(device=resolved_device)
                time_index = torch.arange(
                    missing.shape[1],
                    device=missing.device,
                ).unsqueeze(0)
                valid = valid & (time_index < lengths.unsqueeze(1)).unsqueeze(-1)
            return valid.flatten(start_dim=1).any(dim=1)
        raise ValueError("tab_missing must have shape (B,), (B, F), or (B, T, F).")

    if "tab_lengths" in batch:
        lengths = batch["tab_lengths"]
        if resolved_device is not None:
            lengths = lengths.to(device=resolved_device)
        return lengths.gt(0)

    if "tab_feats" in batch:
        feats = batch["tab_feats"]
        if resolved_device is not None:
            feats = feats.to(device=resolved_device)
        return torch.isfinite(feats).flatten(start_dim=1).all(dim=1) & feats.ne(
            0
        ).flatten(start_dim=1).any(dim=1)

    return torch.ones(resolved_batch_size, dtype=torch.bool, device=resolved_device)


def sequence_available(
    batch: Mapping[str, torch.Tensor],
    modality: str,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-sample sequence availability using missingness and lengths."""
    resolved_batch_size = _resolve_batch_size(batch, batch_size)
    resolved_device = _resolve_device(batch, device)
    available = torch.ones(
        resolved_batch_size,
        dtype=torch.bool,
        device=resolved_device,
    )

    missing_key = f"{modality}_missing"
    lengths_key = f"{modality}_lengths"
    feats_key = f"{modality}_feats"
    if missing_key in batch:
        missing = batch[missing_key]
        if resolved_device is not None:
            missing = missing.to(device=resolved_device)
        if missing.ndim == 1:
            available = available & missing.eq(0)
    if lengths_key in batch:
        lengths = batch[lengths_key]
        if resolved_device is not None:
            lengths = lengths.to(device=resolved_device)
        available = available & lengths.gt(0)
    if missing_key in batch or lengths_key in batch:
        return available

    if feats_key in batch:
        feats = batch[feats_key]
        if resolved_device is not None:
            feats = feats.to(device=resolved_device)
        return torch.isfinite(feats).flatten(start_dim=1).all(dim=1) & feats.ne(
            0
        ).flatten(start_dim=1).any(dim=1)

    return available


def ts_available(
    batch: Mapping[str, torch.Tensor],
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-sample whole-modality presence for the regular-TS sequence.

    Unlike ecg/cxr (which carry only lengths), ``ts_missing`` is a per-timestep,
    per-feature informative mask of shape ``(B, T, F)`` -- within-series missingness,
    not modality presence -- so it must not be folded into availability the way
    ``sequence_available`` folds a per-sample mask. Presence is taken from the series
    length (a stay has ts iff its series is non-empty), falling back to feats.
    """
    resolved_batch_size = _resolve_batch_size(batch, batch_size)
    resolved_device = _resolve_device(batch, device)

    if "ts_lengths" in batch:
        lengths = batch["ts_lengths"]
        if resolved_device is not None:
            lengths = lengths.to(device=resolved_device)
        return lengths.gt(0)

    if "ts_feats" in batch:
        feats = batch["ts_feats"]
        if resolved_device is not None:
            feats = feats.to(device=resolved_device)
        return torch.isfinite(feats).flatten(start_dim=1).all(dim=1) & feats.ne(
            0
        ).flatten(start_dim=1).any(dim=1)

    return torch.ones(resolved_batch_size, dtype=torch.bool, device=resolved_device)


def modality_available(
    batch: Mapping[str, torch.Tensor],
    modality: str,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-sample availability for a public modality name."""
    if modality in TABULAR_MODALITIES:
        return tabular_available(batch, batch_size, device)
    if modality == "ts":
        return ts_available(batch, batch_size, device)
    prefix = modality_prefix(modality)
    if (
        f"{prefix}_missing" in batch
        and batch[f"{prefix}_missing"].ndim > 1
        and f"{prefix}_lengths" in batch
    ):
        lengths = batch[f"{prefix}_lengths"]
        resolved_device = _resolve_device(batch, device)
        if resolved_device is not None:
            lengths = lengths.to(device=resolved_device)
        return lengths.gt(0)
    return sequence_available(batch, prefix, batch_size, device)


def batch_modality_availability(
    batch: Mapping[str, torch.Tensor],
    detach_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Return natural availability for the batch's configured modality order."""
    batch_size = _resolve_batch_size(batch, None)
    modalities = tuple(batch.get("modality_names", ()))
    if not modalities:
        modalities = ("tabular", "ecg", "cxr")
        if any(f"ts_{suffix}" in batch for suffix in ("feats", "lengths", "missing")):
            modalities = (*modalities, "ts")
    availability = {
        modality: modality_available(batch, modality, batch_size)
        for modality in modalities
    }
    if not detach_cpu:
        return availability
    return {
        modality: value.detach().cpu()
        for modality, value in availability.items()
    }


def _pad_1d_tensors(
    sequences: list[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad one-dimensional tensors to the batch maximum length."""
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.int64,
    )
    max_length = int(lengths.max().item()) if len(sequences) else 0
    padded = torch.zeros((len(sequences), max_length), dtype=dtype)

    for index, sequence in enumerate(sequences):
        length = int(sequence.shape[0])
        if length:
            padded[index, :length] = sequence.to(dtype=dtype)

    return padded, lengths


def _pad_2d_tensors(
    sequences: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    feature_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad two-dimensional tensors to the batch maximum sequence length."""
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.int64,
    )
    max_length = int(lengths.max().item()) if len(sequences) else 0
    inferred_feature_dim = feature_dim

    if inferred_feature_dim is None:
        inferred_feature_dim = 0
        for sequence in sequences:
            if sequence.ndim != 2:
                raise ValueError("Expected two-dimensional tensors for sequence padding.")
            if sequence.shape[1] > 0:
                inferred_feature_dim = int(sequence.shape[1])
                break

    padded = torch.zeros(
        (len(sequences), max_length, inferred_feature_dim),
        dtype=dtype,
    )

    for index, sequence in enumerate(sequences):
        if sequence.ndim != 2:
            raise ValueError("Expected two-dimensional tensors for sequence padding.")
        length = int(sequence.shape[0])
        if length:
            padded[index, :length] = sequence.to(dtype=dtype)

    return padded, lengths


def _stack_or_pad_tabular(
    sequences: list[torch.Tensor],
    missing: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Collate tabular features for either flat vectors or nested sequences."""
    first_tensor = sequences[0]
    if first_tensor.ndim == 1:
        return {
            "tab_feats": torch.stack(
                [tensor.to(dtype=torch.float32) for tensor in sequences]
            ),
            "tab_missing": torch.stack(
                [tensor.to(dtype=torch.int64) for tensor in missing]
            ),
        }
    if first_tensor.ndim != 2:
        raise ValueError("tab_feats must be one- or two-dimensional per sample.")

    padded_feats, tab_lengths = _pad_2d_tensors(sequences, dtype=torch.float32)
    padded_missing, _ = _pad_2d_tensors(
        missing,
        dtype=torch.int64,
        feature_dim=int(padded_feats.shape[-1]),
    )
    return {
        "tab_feats": padded_feats,
        "tab_missing": padded_missing,
        "tab_lengths": tab_lengths,
    }


def collate_base_batch(
    batch: list[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Collate samples for the shared multimodal batch contract."""
    if not batch:
        raise ValueError("batch must contain at least one sample.")
    modalities = tuple(batch[0].get("modality_names", ()))
    if not modalities:
        modalities = ("tabular", "ecg", "cxr")
        if "ts_feats" in batch[0]:
            modalities = (*modalities, "ts")

    identifier_columns = [
        column for column in ("hadm_id", "ed_stay_id") if column in batch[0]
    ]
    if not identifier_columns:
        raise ValueError("Samples must contain hadm_id or ed_stay_id.")
    for identifier_column in identifier_columns:
        if any(identifier_column not in sample for sample in batch):
            raise ValueError(f"Every sample must contain {identifier_column}.")

    collated = {
        **{
            identifier_column: torch.tensor(
                [sample[identifier_column] for sample in batch],
                dtype=torch.int64,
            )
            for identifier_column in identifier_columns
        },
        "label": torch.stack([sample["label"] for sample in batch]).to(
            dtype=torch.float32
        ),
        "modality_names": modalities,
    }

    for modality in modalities:
        prefix = modality_prefix(modality)
        if modality in TABULAR_MODALITIES:
            collated.update(
                _stack_or_pad_tabular(
                    [sample[f"{prefix}_feats"] for sample in batch],
                    [sample[f"{prefix}_missing"] for sample in batch],
                )
            )
            continue

        feats, lengths = _pad_2d_tensors(
            [sample[f"{prefix}_feats"] for sample in batch],
            dtype=torch.float32,
        )
        time, _ = _pad_1d_tensors(
            [sample[f"{prefix}_time"] for sample in batch],
            dtype=torch.float32,
        )
        missing_values = [sample[f"{prefix}_missing"] for sample in batch]
        if missing_values[0].ndim == 0:
            missing = torch.stack(missing_values).to(dtype=torch.int64)
        else:
            missing, _ = _pad_2d_tensors(
                missing_values,
                dtype=torch.int64,
                feature_dim=feats.shape[-1],
            )
        collated.update(
            {
                f"{prefix}_feats": feats,
                f"{prefix}_time": time,
                f"{prefix}_lengths": lengths,
                f"{prefix}_missing": missing,
            }
        )

    if "labels" in batch[0]:
        collated["labels"] = torch.stack([sample["labels"] for sample in batch]).to(
            dtype=torch.float32
        )
        collated["label_mask"] = torch.stack(
            [sample["label_mask"] for sample in batch]
        ).to(dtype=torch.float32)

    return collated


def collate_fusemoe_batch(
    batch: list[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Collate FuseMoE batches.

    This currently shares the baseline contract and gives us a named extension point
    for FuseMoE-specific routing or modality-mask fields.
    """
    return collate_base_batch(batch)


def collate_m3care_batch(
    batch: list[dict[str, torch.Tensor | int]],
) -> dict[str, torch.Tensor]:
    """Collate M3Care batches.

    This currently shares the baseline contract and gives us a named extension point
    for M3Care-specific temporal or graph fields.
    """
    return collate_base_batch(batch)


COLLATE_FNS = {
    "base": collate_base_batch,
    "concat": collate_base_batch,
    "fusemoe": collate_fusemoe_batch,
    "m3care": collate_m3care_batch,
}


def get_collate_fn(model_name: str):
    """Return the collate function for a model family."""
    try:
        return COLLATE_FNS[model_name.lower()]
    except KeyError as exc:
        available = ", ".join(sorted(COLLATE_FNS))
        raise ValueError(
            f"Unknown collate style {model_name!r}. Available styles: {available}."
        ) from exc


class BaseCollator:
    """Callable collator for the shared baseline batch contract."""

    def __call__(
        self,
        batch: list[dict[str, torch.Tensor | int]],
    ) -> dict[str, torch.Tensor]:
        return collate_base_batch(batch)


class FuseMoECollator:
    """Callable collator for FuseMoE batches."""

    def __call__(
        self,
        batch: list[dict[str, torch.Tensor | int]],
    ) -> dict[str, torch.Tensor]:
        return collate_fusemoe_batch(batch)


class M3CareCollator:
    """Callable collator for M3Care batches."""

    def __call__(
        self,
        batch: list[dict[str, torch.Tensor | int]],
    ) -> dict[str, torch.Tensor]:
        return collate_m3care_batch(batch)


MultimodalCollator = BaseCollator
collate_multimodal_batch = collate_base_batch
