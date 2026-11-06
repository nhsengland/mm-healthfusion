from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLIT_NAMES = ("train", "val", "test")
DETERIORATION_MODALITIES = ("tabular", "ecg", "cxr")
HOSPITALISATION_MODALITIES = (
    "static",
    "ts_prev",
    "ts_curr",
    "ecg_curr",
    "cxr_prev",
    "cxr_curr",
    "notes_prev",
)
TABULAR_MODALITIES = {"tabular", "static"}


def modality_prefix(modality: str) -> str:
    """Return the parquet/batch key prefix for a public modality name."""
    return "tab" if modality in TABULAR_MODALITIES else modality


def modality_columns(modality: str) -> tuple[str, ...]:
    """Return the parquet columns forming one modality record."""
    prefix = modality_prefix(modality)
    suffixes = (
        ("feats", "missing")
        if modality in TABULAR_MODALITIES
        else ("feats", "time", "missing")
    )
    return tuple(f"{prefix}_{suffix}" for suffix in suffixes)


def infer_outcome_columns(
    frame: pd.DataFrame,
    label_prefix: str = "deterioration_",
) -> list[str]:
    """Return task label columns from a split parquet frame."""
    return sorted(
        column for column in frame.columns if column.startswith(label_prefix)
    )


def resolve_split_path(bundle_dir: str | Path, split: str) -> Path:
    """Resolve the parquet path for a bundle split."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}.")

    bundle_dir = Path(bundle_dir)
    candidates = sorted(bundle_dir.glob(f"{split}_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No {split} parquet file found in {bundle_dir}."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Expected one {split} parquet file in {bundle_dir}, found {len(candidates)}."
        )
    return candidates[0]


def _infer_sequence_width(values: pd.Series) -> int:
    """Infer the trailing feature width from the first non-empty sequence."""
    for value in values:
        array = _normalise_array(value, dtype=np.float32)
        if array.size == 0:
            continue
        if array.ndim == 1:
            return int(array.shape[0])
        if array.ndim == 2:
            return int(array.shape[1])
        raise ValueError("Expected a one- or two-dimensional sequence.")
    return 0


def _normalise_array(
    value: object,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    """Convert list-like feature values into a numeric one- or two-dimensional array."""
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0,), dtype=dtype)

    if array.dtype != object:
        return np.asarray(array, dtype=dtype)

    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        if array.ndim != 1:
            raise ValueError("Expected a one- or two-dimensional sequence.") from None

        nested = []
        for item in array.tolist():
            item_array = np.asarray(item, dtype=dtype)
            if item_array.ndim == 0:
                item_array = item_array.reshape(1)
            nested.append(item_array)

        if not nested:
            return np.empty((0,), dtype=dtype)

        first_ndim = nested[0].ndim
        if any(item.ndim != first_ndim for item in nested):
            raise ValueError("Expected a consistent nested sequence structure.")
        if first_ndim != 1:
            raise ValueError("Expected a one- or two-dimensional sequence.")

        widths = {int(item.shape[0]) for item in nested}
        if len(widths) != 1:
            raise ValueError("Expected equal-width nested feature vectors.")

        return np.stack(nested, axis=0)


def _to_float_tensor(
    value: object,
    *,
    expected_width: int | None = None,
) -> torch.Tensor:
    """Convert a list-like value to a float32 tensor."""
    array = _normalise_array(value, dtype=np.float32)
    if array.size == 0:
        if expected_width is None:
            return torch.empty((0,), dtype=torch.float32)
        return torch.empty((0, expected_width), dtype=torch.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.array(array, dtype=np.float32, copy=True)
    return torch.from_numpy(array)


def _to_int_tensor(
    value: object,
    *,
    expected_width: int | None = None,
) -> torch.Tensor:
    """Convert a list-like value to an int64 tensor."""
    array = _normalise_array(value, dtype=np.int64)
    if array.size == 0:
        if expected_width is None:
            return torch.empty((0,), dtype=torch.int64)
        return torch.empty((0, expected_width), dtype=torch.int64)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.array(array, dtype=np.int64, copy=True)
    return torch.from_numpy(array)


def _build_label_tensor(
    value: object,
) -> torch.Tensor:
    """Convert a single-task label to a tensor."""
    label_value = 0.0 if pd.isna(value) else float(value)
    return torch.tensor(label_value, dtype=torch.float32)


def _build_mtl_tensors(
    values: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert MTL labels to safe tensors and validity masks."""
    label_array = values.to_numpy(dtype=np.float32, na_value=np.float32(0.0))
    mask_array = values.notna().to_numpy(dtype=np.float32)
    return (
        torch.tensor(label_array, dtype=torch.float32),
        torch.tensor(mask_array, dtype=torch.float32),
    )


def _safe_int(value: object, *, missing_value: int = -1) -> int:
    """Convert nullable scalar identifiers to integers with a sentinel for missing."""
    if pd.isna(value):
        return missing_value
    return int(value)


class MultimodalSplitDataset(Dataset):
    """Shared dataset for split parquet outputs with named modalities."""

    def __init__(
        self,
        parquet_path: str | Path,
        outcome: str | None = None,
        MTL: bool = False,
        *,
        label_prefix: str,
        modalities: Sequence[str],
        identifier_column: str = "ed_stay_id",
        source_identifier_column: str | None = None,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.frame = pd.read_parquet(self.parquet_path)
        self.label_prefix = str(label_prefix)
        self.modality_names = tuple(str(modality) for modality in modalities)
        self.identifier_column = identifier_column
        self.source_identifier_column = source_identifier_column or identifier_column
        self.outcome_columns = infer_outcome_columns(self.frame, self.label_prefix)

        required_columns = {self.source_identifier_column}
        for modality in self.modality_names:
            required_columns.update(modality_columns(modality))
        missing_columns = sorted(required_columns.difference(self.frame.columns))
        if missing_columns:
            raise ValueError(
                f"{self.parquet_path} is missing required columns: {missing_columns}."
            )
        if not self.outcome_columns:
            raise ValueError(
                f"{self.parquet_path} does not contain any {self.label_prefix} outcome columns."
            )

        if outcome is None:
            self.outcome = self.outcome_columns[0]
        elif outcome in self.outcome_columns:
            self.outcome = outcome
        else:
            raise ValueError(
                f"Unknown outcome {outcome!r}. Available outcomes: {self.outcome_columns}."
            )

        self.MTL = MTL
        if not self.MTL:
            self.frame = self.frame[self.frame[self.outcome].notna()].reset_index(
                drop=True
            )
            if self.frame.empty:
                raise ValueError(
                    f"{self.parquet_path} has no valid labels for outcome "
                    f"{self.outcome!r}."
                )

        self.tabular_is_sequential = (
            self._column_is_nested(self.frame["tab_feats"])
            if any(modality in TABULAR_MODALITIES for modality in self.modality_names)
            else False
        )
        self.modality_widths = {
            modality: _infer_sequence_width(
                self.frame[f"{modality_prefix(modality)}_feats"]
            )
            for modality in self.modality_names
        }
        self.modality_missing_widths = {
            modality: _infer_sequence_width(
                self.frame[f"{modality_prefix(modality)}_missing"]
            )
            for modality in self.modality_names
            if modality in TABULAR_MODALITIES
        }
        self.tabular_width = self.modality_widths.get("tabular", self.modality_widths.get("static", 0))
        self.tabular_missing_width = self.modality_missing_widths.get(
            "tabular",
            self.modality_missing_widths.get("static", 0),
        )
        self.ecg_width = self.modality_widths.get("ecg", 0)
        self.cxr_width = self.modality_widths.get("cxr", 0)
        self.ts_present = "ts" in self.modality_names
        self.ts_width = self.modality_widths.get("ts", 0)
        self._outcome_valid_mask = self.frame[self.outcome_columns].notna()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        row = self.frame.iloc[index]
        label = _build_label_tensor(row[self.outcome])
        sample = {
            self.identifier_column: _safe_int(row[self.source_identifier_column]),
            "label": label,
            "modality_names": self.modality_names,
        }

        for modality in self.modality_names:
            prefix = modality_prefix(modality)
            width = self.modality_widths[modality]
            if modality in TABULAR_MODALITIES:
                sample[f"{prefix}_feats"] = _to_float_tensor(
                    row[f"{prefix}_feats"],
                    expected_width=width if self.tabular_is_sequential else None,
                )
                sample[f"{prefix}_missing"] = _to_int_tensor(
                    row[f"{prefix}_missing"],
                    expected_width=(
                        self.modality_missing_widths[modality]
                        if self.tabular_is_sequential
                        else None
                    ),
                )
                continue

            sample[f"{prefix}_feats"] = _to_float_tensor(
                row[f"{prefix}_feats"],
                expected_width=width,
            )
            sample[f"{prefix}_time"] = _to_float_tensor(row[f"{prefix}_time"])
            missing_value = row[f"{prefix}_missing"]
            if np.asarray(missing_value).ndim == 0:
                sample[f"{prefix}_missing"] = torch.tensor(
                    _safe_int(missing_value),
                    dtype=torch.int64,
                )
            else:
                sample[f"{prefix}_missing"] = _to_int_tensor(
                    missing_value,
                    expected_width=width,
                )

        if self.MTL:
            labels, labels_mask = _build_mtl_tensors(row[self.outcome_columns])
            sample["labels"] = labels
            sample["label_mask"] = labels_mask

        return sample
    
    def filter_rows(self, mask: pd.Series) -> None:
        """Filter the dataset rows in place and refresh cached metadata."""
        self.frame = self.frame.loc[mask].reset_index(drop=True)

        if self.frame.empty:
            raise ValueError(
                f"{self.parquet_path} has no rows after dataset filtering."
            )

        self._outcome_valid_mask = self.frame[self.outcome_columns].notna()
        self.__dict__.pop("_permutation_strata", None)

    @staticmethod
    def _column_is_nested(values: pd.Series) -> bool:
        """Return True when a column stores list-of-lists style features."""
        for value in values:
            array = _normalise_array(value, dtype=np.float32)
            if array.size == 0:
                continue
            return array.ndim == 2
        return False

    @property
    def feature_dims(self) -> dict[str, int]:
        """Return per-modality trailing feature widths."""
        return dict(self.modality_widths)

    @property
    def outcome_valid_counts(self) -> dict[str, int]:
        """Return the number of valid labels per deterioration outcome."""
        return {
            column: int(self._outcome_valid_mask[column].sum())
            for column in self.outcome_columns
        }

    @property
    def outcome_missing_counts(self) -> dict[str, int]:
        """Return the number of missing labels per deterioration outcome."""
        return {
            column: int((~self._outcome_valid_mask[column]).sum())
            for column in self.outcome_columns
        }

    @property
    def valid_label_count(self) -> int:
        """Return the number of valid labels for the selected outcome."""
        return int(self._outcome_valid_mask[self.outcome].sum())


class DeteriorationDataset(MultimodalSplitDataset):
    """Dataset for split parquet outputs created by ``split_deterioration.py``."""

    def __init__(
        self,
        parquet_path: str | Path,
        outcome: str | None = None,
        MTL: bool = False,
        modalities: Sequence[str] | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        frame_columns = pq.read_schema(parquet_path).names
        default_modalities = list(DETERIORATION_MODALITIES)
        if modalities is None and "ts_feats" in frame_columns:
            default_modalities.append("ts")
        super().__init__(
            parquet_path,
            outcome=outcome,
            MTL=MTL,
            label_prefix="deterioration_",
            modalities=modalities or default_modalities,
        )


class HospitalisationDataset(MultimodalSplitDataset):
    """Hospitalisation dataset keyed publicly by admission identifier.

    ``ed_stay_id`` remains in samples for modality lookups, but exported results use
    the canonical ``hadm_id`` sourced from the parquet's ``ed_hadm_id`` column.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        outcome: str | None = None,
        MTL: bool = False,
        modalities: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            parquet_path,
            outcome=outcome,
            MTL=MTL,
            label_prefix="hospitalisation_",
            modalities=modalities or HOSPITALISATION_MODALITIES,
            identifier_column="hadm_id",
            source_identifier_column="ed_hadm_id",
        )
        if "ed_stay_id" not in self.frame:
            raise ValueError(f"{self.parquet_path} is missing required column: ed_stay_id.")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample = super().__getitem__(index)
        sample["ed_stay_id"] = _safe_int(self.frame.iloc[index]["ed_stay_id"])
        return sample


def _check_missing_matches_emptiness(frame: pd.DataFrame, modality: str) -> None:
    """Assert a sequence modality's ``*_missing`` flag agrees with feats emptiness.

    Coarse strata read availability from the ``*_missing`` scalar alone, but
    collate derives availability as ``missing == 0 AND length > 0`` (an empty
    ``*_feats`` yields length 0). Hospitalisation time-series missingness is a
    per-timestep mask, so only scalar missingness flags are checked here. If a
    scalar flag and its sequence emptiness disagree, a within-stratum swap could
    flip a row's collate-time availability, silently breaking the ablation.
    Fail loudly instead of relying on that never happening.
    """
    if modality == "ts":
        return
    prefix = modality_prefix(modality)
    missing_values = frame[f"{prefix}_missing"]
    mismatches = []
    for index, (missing_value, features) in enumerate(
        zip(missing_values, frame[f"{prefix}_feats"], strict=True)
    ):
        missing_array = np.asarray(missing_value)
        if missing_array.ndim != 0:
            continue
        is_missing = int(missing_array) == 1
        is_empty = _normalise_array(features, dtype=np.float32).shape[0] == 0
        if is_missing != is_empty:
            mismatches.append(index)

    if mismatches:
        raise ValueError(
            f"{prefix}_missing disagrees with {prefix}_feats emptiness for "
            f"{len(mismatches)} row(s); availability strata would not match collate-time "
            f"availability. Expected missing == 1 exactly when feats are empty."
        )


def _frame_modality_available(frame: pd.DataFrame, modality: str) -> np.ndarray:
    """Return row-wise availability using the same rules as the collate layer."""
    prefix = modality_prefix(modality)
    features = frame[f"{prefix}_feats"]
    missing = frame[f"{prefix}_missing"]
    available = []
    for feature_value, missing_value in zip(features, missing, strict=True):
        feature_array = _normalise_array(feature_value, dtype=np.float32)
        if modality in TABULAR_MODALITIES:
            missing_array = _normalise_array(missing_value, dtype=np.int64)
            if feature_array.ndim <= 1:
                available.append(bool(np.any(missing_array == 0)))
            else:
                available.append(
                    bool(np.any(missing_array[: feature_array.shape[0]] == 0))
                )
            continue

        if feature_array.shape[0] == 0:
            available.append(False)
            continue
        if modality == "ts":
            available.append(True)
            continue
        missing_array = np.asarray(missing_value)
        available.append(
            missing_array.ndim != 0 or int(missing_array) == 0
        )
    return np.asarray(available, dtype=bool)


def sample_missingness_strata(
    frame: pd.DataFrame,
    modalities: Sequence[str] | None = None,
) -> dict[tuple[bool, ...], np.ndarray]:
    """Group rows by the configured modalities' availability pattern.

    When no modality list is supplied, the outcome columns select the standard task schema.
    Callers wrapping a dataset pass its ``modality_names`` so custom subsets are represented
    exactly.
    """
    if modalities is None:
        modalities = (
            HOSPITALISATION_MODALITIES
            if any(column.startswith("hospitalisation_") for column in frame.columns)
            else DETERIORATION_MODALITIES
        )
    modalities = tuple(modalities)
    missing_columns = sorted(
        {
            column
            for modality in modalities
            for column in modality_columns(modality)
        }.difference(frame.columns)
    )
    if missing_columns:
        raise ValueError(
            f"Cannot build availability strata; missing modality columns: {missing_columns}."
        )
    for modality in modalities:
        _check_missing_matches_emptiness(frame, modality)

    strata: dict[tuple[bool, ...], list[int]] = {}
    availability = {
        modality: _frame_modality_available(frame, modality)
        for modality in modalities
    }
    for index, key in enumerate(
        zip(
            *(availability[modality].tolist() for modality in modalities),
            strict=True,
        )
    ):
        strata.setdefault(key, []).append(index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in strata.items()}


def stratified_permutation(
    strata: Mapping[tuple[bool, ...], np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a length-``n`` source index map that shuffles only within strata.

    ``perm[i]`` is the source row supplying target position ``i``. Positions are
    permuted among the members of their own stratum and left as identity across
    strata (and for singleton strata), so the availability pattern of every row is
    preserved while its within-stratum content donor is randomised.
    """
    perm = np.arange(n)
    for rows in strata.values():
        if rows.size > 1:
            perm[rows] = rows[rng.permutation(rows.size)]
    return perm


class PermutationDeteriorationDataset(MultimodalSplitDataset):
    """A base dataset with one modality-information permutation applied.

    Wraps an *already-built* task dataset and precomputes a single
    permuted draw: for each requested modality, that modality's column block is
    reindexed by an independent within-stratum permutation, so a patient keeps the
    same modality availability but receives another (same-missingness) patient's
    content. ``super().__init__`` is deliberately skipped -- copying ``base``'s
    already-built state (frame, width caches, ``_outcome_valid_mask``, outcome),
    which a within-column reorder leaves valid, lets ``__getitem__`` / ``__len__``
    be inherited verbatim with no duplicated sample-assembly logic.

    Labels and ``ed_stay_id`` are never permuted, so predictions stay aligned to
    the correct target. ``modalities=()`` yields an identity copy (the baseline).
    ``.permutation`` (per-modality source index maps) and ``.strata`` are exposed
    for testing and diagnostics.
    """

    def __init__(
        self,
        base: MultimodalSplitDataset,
        modalities: Sequence[str] = (),
        *,
        seed: int,
    ) -> None:
        unknown = [m for m in modalities if m not in base.modality_names]
        if unknown:
            raise ValueError(
                f"Unknown modality names {unknown}; active modalities are "
                f"{list(base.modality_names)}."
            )

        strata = getattr(base, "_permutation_strata", None)
        if strata is None:
            strata = sample_missingness_strata(base.frame, base.modality_names)
            base._permutation_strata = strata

        self.__dict__.update(base.__dict__)
        self.strata = strata
        row_count = len(base.frame)

        modalities_sorted = sorted(modalities)
        seed_seq = np.random.SeedSequence(int(seed))
        child_seeds = seed_seq.spawn(len(modalities_sorted))
        self.permutation = {
            modality: stratified_permutation(
                self.strata,
                row_count,
                np.random.default_rng(child_seed),
            )
            for modality, child_seed in zip(
                modalities_sorted,
                child_seeds,
                strict=True,
            )
        }

        frame = base.frame.copy()
        for modality, perm in self.permutation.items():
            for column in modality_columns(modality):
                # Positional numpy reindex preserves each column's dtype (object
                # feature/time lists stay object; int missing flags stay int).
                frame[column] = base.frame[column].to_numpy()[perm]
        self.frame = frame


def load_dataset_split(
    bundle_dir: str | Path,
    split: str,
    outcome: str | None = None,
    MTL: bool = False,
    dataset_cls: type[MultimodalSplitDataset] = DeteriorationDataset,
    modalities: Sequence[str] | None = None,
) -> MultimodalSplitDataset:
    """Build a dataset from a split name inside a split output directory."""
    return dataset_cls(
        parquet_path=resolve_split_path(bundle_dir, split),
        outcome=outcome,
        MTL=MTL,
        modalities=modalities,
    )


def load_dataset_from_parquet(
    parquet_path: str | Path,
    outcome: str | None = None,
    MTL: bool = False,
    dataset_cls: type[MultimodalSplitDataset] = DeteriorationDataset,
    modalities: Sequence[str] | None = None,
) -> MultimodalSplitDataset:
    """Build a dataset from an explicit split parquet path."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Split parquet file does not exist: {parquet_path}.")
    if parquet_path.suffix != ".parquet":
        raise ValueError(f"Expected a .parquet file, got {parquet_path}.")
    return dataset_cls(
        parquet_path=parquet_path,
        outcome=outcome,
        MTL=MTL,
        modalities=modalities,
    )
