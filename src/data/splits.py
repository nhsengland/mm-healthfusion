from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.collate import get_collate_fn
from src.data.datasets import (
    DeteriorationDataset,
    HospitalisationDataset,
    MultimodalSplitDataset,
    PermutationDeteriorationDataset,
    SPLIT_NAMES,
    load_dataset_from_parquet,
    load_dataset_split,
    resolve_split_path,
)
import numpy as np
import pandas as pd

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


def _section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"cfg[{key!r}] must be a mapping.")
    return value


def _data_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(cfg, "data")


def _model_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(cfg, "model")


def _get_first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _split_paths(data_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    paths = _get_first(
        data_cfg.get("paths"),
        data_cfg.get("split_paths"),
        data_cfg.get("splits"),
        default={},
    )
    if paths is None:
        return {}
    if not isinstance(paths, Mapping):
        raise TypeError("data.paths, data.split_paths or data.splits must be a mapping.")
    return paths


def _resolve_split_source(
    data_cfg: Mapping[str, Any],
    split: str,
) -> tuple[str, Path]:
    paths = _split_paths(data_cfg)
    explicit_path = _get_first(paths.get(split), data_cfg.get(f"{split}_path"))
    if explicit_path is not None:
        return "file", Path(explicit_path)

    split_dir = _configured_split_dir(data_cfg)
    if split_dir is None:
        raise ValueError(
            "Provide explicit data split paths or one of data.split_dir, "
            "data.bundle_dir or data.data_dir."
        )
    return "directory", Path(split_dir)


def _configured_split_dir(data_cfg: Mapping[str, Any]) -> Any:
    return _get_first(
        data_cfg.get("split_dir"),
        data_cfg.get("bundle_dir"),
        data_cfg.get("data_dir"),
    )


def resolve_split_dir(cfg: Mapping[str, Any]) -> Path | None:
    """Return the configured split directory, or None when none is configured."""
    split_dir = _configured_split_dir(_data_cfg(cfg))
    return None if split_dir is None else Path(split_dir)


def resolve_split_parquet(cfg: Mapping[str, Any], split: str) -> Path | None:
    """Return the parquet ``get_datasets`` loads for a split, or None when none is configured."""
    try:
        source_kind, source_path = _resolve_split_source(_data_cfg(cfg), split)
    except ValueError:
        return None
    if source_kind == "file":
        return source_path
    return resolve_split_path(source_path, split)


def find_split_summary(split_dir: str | Path) -> Path | None:
    """Return the first split summary JSON in a split directory, skipping macOS ``._`` files."""
    matches = [
        path for path in sorted(Path(split_dir).glob("*_split_summary.json"))
        if not path.name.startswith("._")
    ]
    return matches[0] if matches else None


def _collate_style(cfg: Mapping[str, Any]) -> str:
    data_cfg = _data_cfg(cfg)
    model_cfg = _model_cfg(cfg)
    return str(
        _get_first(
            data_cfg.get("collate_style"),
            model_cfg.get("collate_style"),
            model_cfg.get("name"),
            model_cfg.get("model"),
            model_cfg.get("fusion_method"),
            default="base",
        )
    )


def _dataset_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = _data_cfg(cfg)
    return {
        "outcome": _get_first(data_cfg.get("outcome"), cfg.get("outcome")),
        "MTL": bool(
            _get_first(
                data_cfg.get("MTL"),
                cfg.get("MTL"),
                default=False,
            )
        ),
        "modalities": data_cfg.get("modalities"),
    }

def _resolve_complete_set_modalities(
    cfg: Mapping[str, Any],
    dataset_cls: type[MultimodalSplitDataset],
) -> list[str]:
    """Resolve modalities used to define a complete case.

    data.complete_set_modalities takes precedence. When omitted, all
    modalities in model.modality_names are used.
    """
    data_cfg = _data_cfg(cfg)
    model_cfg = _model_cfg(cfg)

    if dataset_cls is HospitalisationDataset:
        supported = set(HOSPITALISATION_MODALITIES)
    elif dataset_cls is DeteriorationDataset:
        supported = set(DETERIORATION_MODALITIES) | {"ts"}
    else:
        raise TypeError(
            f"Complete-case filtering is only supported for DeteriorationDataset and HospitalisationDataset,"
            f"got {dataset_cls!r}."
        )

    configured = data_cfg.get("complete_set_modalities")

    if configured is None:
        default_modalities = (
            HOSPITALISATION_MODALITIES
            if dataset_cls is HospitalisationDataset
            else DETERIORATION_MODALITIES
        )
        configured = list(
            _get_first(
                data_cfg.get("modalities"),
                model_cfg.get("modality_names"),
                default=default_modalities,
            )
        )

    if isinstance(configured, str):
        configured = [configured]

    if not isinstance(configured, Sequence) or isinstance(
        configured,
        (str, bytes),
    ):
        raise TypeError(
            "data.complete_set_modalities must be a sequence of modality names."
        )

    modalities = [
        str(modality).strip().lower()
        for modality in configured
    ]

    if not modalities:
        raise ValueError(
            "complete_set_modalities cannot be empty when "
            "use_complete_set=true."
        )

    unknown = sorted(set(modalities) - supported)
    if unknown:
        raise ValueError(
            f"Unknown modalities in complete_set_modalities: {unknown}. "
            f"Supported modalities are {sorted(supported)}."
        )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(modalities))

def _filter_complete_cases(
    dataset: MultimodalSplitDataset,
    modalities: Sequence[str],
) -> MultimodalSplitDataset:
    """Filter a dataset to samples containing all requested modalities.

    Availability follows the same definitions used by the existing collate/
    modality-availability utilities:

    - tabular, static: at least one tabular entry has tab_missing == 0
    - ecg: ecg_missing == 0 and ecg_feats sequence is non-empty
    - ecg_curr: ecg_curr_missing == 0 and ecg_curr_feats is non-empty
    - cxr: cxr_missing == 0 and cxr_feats is non-empty
    - cxr_prev: cxr_prev_missing == 0 and cxr_prev_feats is non-empty
    - cxr_curr: cxr_curr_missing == 0 and cxr_curr_feats is non-empty
    - ts, ts_prev, ts_curr: sequence (ts_feats, ts_prev_feats, ts_curr_feats) is non-empty,
      i.e., has at least one timestep
    - notes_prev: notes_prev_missing == 0 and notes_prev_feats is non-empty
    """
    frame = dataset.frame.copy()

    complete_mask = pd.Series(
        True,
        index=frame.index,
        dtype=bool,
    )

    for modality in modalities:
        if modality in {"static", "tabular"}:
            available = frame["tab_missing"].apply(
                lambda value: bool(
                    np.any(np.asarray(value, dtype=np.int64) == 0)
                )
            )

        elif modality in {"ts", "ts_prev", "ts_curr"}:
            feats_key = f"{modality}_feats"
            if feats_key not in frame.columns:
                raise ValueError(
                    f"Complete-case filtering requested {modality!r}, "
                    f"but {feats_key!r} is absent."
                )
            available = frame[feats_key].apply(
                lambda value: np.asarray(value).shape[0] > 0
            )

        else:
            prefix = modality
            available = (
                frame[f"{prefix}_missing"].astype(int).eq(0)
                & frame[f"{prefix}_feats"].apply(
                    lambda value: np.asarray(value).shape[0] > 0
                )
            )

        complete_mask &= available

    original_count = len(dataset)

    dataset.filter_rows(complete_mask)

    filtered_count = len(dataset)

    if filtered_count == 0:
        raise ValueError(
            f"Complete-case filtering removed all samples. "
            f"Requested modalities: {list(modalities)}."
        )

    print(
        f"Complete-case filtering: retained {filtered_count:,} / "
        f"{original_count:,} samples "
        f"({100.0 * filtered_count / original_count:.2f}%) "
        f"for modalities={list(modalities)}"
    )

    return dataset


def _dataset_cls(cfg: Mapping[str, Any]) -> type[MultimodalSplitDataset]:
    data_cfg = _data_cfg(cfg)
    task = str(data_cfg.get("task", data_cfg.get("dataset", "deterioration"))).lower()
    if task == "deterioration":
        return DeteriorationDataset
    if task == "hospitalisation":
        return HospitalisationDataset
    raise ValueError("data.task must be either 'deterioration' or 'hospitalisation'.")


def get_datasets(
    cfg: Mapping[str, Any],
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, MultimodalSplitDataset]:
    """Build datasets for the requested split names."""
    data_cfg = _data_cfg(cfg)
    dataset_kwargs = _dataset_kwargs(cfg)
    dataset_cls = _dataset_cls(cfg)
    datasets: dict[str, MultimodalSplitDataset] = {}

    use_complete_set = bool(
        data_cfg.get("use_complete_set", False)
    )

    complete_set_modalities: list[str] | None = None
    if use_complete_set:
        # if dataset_cls is not DeteriorationDataset:
        #     raise NotImplementedError(
        #         "data.use_complete_set is only supported for deterioration datasets."
        #     )
        complete_set_modalities = _resolve_complete_set_modalities(
            cfg,
            dataset_cls,
        )
    for split in splits:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}.")
        source_kind, source_path = _resolve_split_source(data_cfg, split)
        if source_kind == "file":
            dataset = load_dataset_from_parquet(
                source_path,
                dataset_cls=dataset_cls,
                **dataset_kwargs,
            )
        else:
            dataset = load_dataset_split(
                source_path,
                split,
                dataset_cls=dataset_cls,
                **dataset_kwargs,
            )

        if complete_set_modalities is not None:
            dataset = _filter_complete_cases(
                dataset,
                complete_set_modalities,
            )
        datasets[split] = dataset

    return datasets


def _build_loader(
    dataset: MultimodalSplitDataset,
    cfg: Mapping[str, Any],
    split: str,
) -> DataLoader:
    """Construct one split's DataLoader from the shared ``data`` config knobs.

    Shared by :func:`get_dataloaders` and :func:`get_permuted_test_dataloader` so a
    permuted test loader uses byte-identical batch size / collate / worker settings
    as the real one -- only the underlying dataset differs.
    """
    data_cfg = _data_cfg(cfg)
    num_workers = int(data_cfg.get("num_workers", 0))
    persistent_workers = bool(
        data_cfg.get("persistent_workers", num_workers > 0)
    ) and num_workers > 0
    is_train = split == "train"

    generator = None
    seed = data_cfg.get("seed")
    if is_train and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=bool(data_cfg.get("shuffle_train", True)) if is_train else False,
        num_workers=num_workers,
        collate_fn=get_collate_fn(_collate_style(cfg)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=bool(data_cfg.get("drop_last", False)) if is_train else False,
        persistent_workers=persistent_workers,
        generator=generator,
    )


def get_dataloaders(
    cfg: Mapping[str, Any],
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, DataLoader]:
    """Build model-aware dataloaders for train, validation and test splits."""
    datasets = get_datasets(cfg, splits=splits)
    return {
        split: _build_loader(dataset, cfg, split)
        for split, dataset in datasets.items()
    }


def get_permuted_test_dataloader(
    cfg: Mapping[str, Any],
    modalities: Sequence[str],
    *,
    seed: int,
    base: MultimodalSplitDataset | None = None,
) -> DataLoader:
    """Build a test loader whose modality *content* is permuted within strata.

    Availability is preserved exactly (see
    :class:`~src.data.datasets.PermutationDeteriorationDataset`); only the listed
    modalities' per-patient records are shuffled among same-missingness patients.
    ``modalities=()`` reproduces the unperturbed test loader (the baseline draw).

    Pass a pre-built ``base`` test dataset to reuse it across many permutation
    draws (avoids re-reading the parquet each draw); when omitted it is loaded from
    ``cfg``.
    """
    if base is None:
        base = get_datasets(cfg, splits=("test",))["test"]
    dataset = PermutationDeteriorationDataset(base, modalities, seed=seed)
    return _build_loader(dataset, cfg, "test")
