"""Create train, validation and test files from a deterioration bundle.

Example::

    python -m src.preprocessing.split_deterioration \
        --bundle deterioration_allStay_allRecord_completeCohort
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.prep_utils import (  # noqa: E402
    apply_ohe_to_frame,
    apply_tabular_normalisation,
    fit_embedding_normalisation_stats,
    fit_ohe_columns,
    parse_time_values,
    resolve_bundle_dir,
    validate_embedding_array,
    validate_unique_keys,
)

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DETERIORATION_OUTPUT_DIR = OUTPUTS_DIR / "deterioration" / "3.1_ohe"
TRAIN_FOLDS = tuple(range(18))
VALIDATION_FOLD = 18
TEST_FOLD = 19
DATE_COLUMNS = ["general_intime", "general_90min", "general_ecg_time"]
KEY_REQUIRED_COLUMNS = [
    "benchmark_row_id",
    "subject_id",
    "ed_stay_id",
    "ed_hadm_id",
    "strat_fold",
    "general_intime",
]
SPLIT_NAMES = ("train", "val", "test")
TABULAR_PREFIXES = (
    "demographics_",
    "biometrics_",
    "vitals_",
    "labvalues_",
)
# Statistical summary suffixes are always treated as numeric columns
TABULAR_STAT_SUFFIXES = (
    "_mean",
    "_median",
    "_min",
    "_max",
    "_std",
    "_first",
    "_last",
    "_rate_change",
    "_coeff",
)
# Keep the five triage levels and represent missing values separately when --ohe is used
VITALS_ACUITY_COLUMN = "vitals_acuity"
VITALS_ACUITY_KEEP_LEVELS = (0, 1, 2, 3, 4)
VITALS_ACUITY_MISSING_COLUMN = "vitals_acuity_missing"
# These indicators are already encoded and pass through without scaling or another OHE step
PASSTHROUGH_BINARY_PREFIXES = ("demographics_ethnicity_", "vitals_acuity_")
PASSTHROUGH_BINARY_NAMES = ("demographics_gender",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train, validation and test parquet files from deterioration "
            "keys, tabular features, and ECG/CXR embeddings."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help=(
            "Bundle folder written by prepare_deterioration.py, for example "
            "deterioration_allStay_allRecord_completeCohort. A bare name is resolved "
            f"under {DETERIORATION_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Parent directory for split parquet files and the split summary JSON. "
            "By default, outputs are written to a bundle-named subfolder inside "
            "the input deterioration bundle directory."
        ),
    )
    parser.add_argument(
        "--include-all-val-test-stays",
        action="store_true",
        default=False,
        help=(
            "Keep all validation and test stays per patient. By default, only the "
            "earliest stay per patient is kept within validation and test."
        ),
    )
    parser.add_argument(
        "--normalise",
        "-n",
        action="store_true",
        default=False,
        help=(
            "Min-max normalise tabular columns and embedding sequences to [0, 1] "
            "using the training split and write *_normalised.parquet files."
        ),
    )
    parser.add_argument(
        "--ohe",
        action="store_true",
        default=False,
        help=(
            "One-hot encode categorical tabular columns using the training split "
            "categories and append _ohe to the output file suffix."
        ),
    )
    return parser.parse_args()


def build_bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "bundle_dir": bundle_dir,
        "keys": bundle_dir / f"{bundle_dir.name}_keys.csv",
        "tabular": bundle_dir / f"{bundle_dir.name}_tabular.csv",
        "ecg_embeddings": bundle_dir / f"{bundle_dir.name}_ecg_embeddings.pkl",
        "cxr_embeddings": bundle_dir / f"{bundle_dir.name}_cxr_embeddings.pkl",
        # Present only for --record all bundles
        "ts": bundle_dir / f"{bundle_dir.name}_ts.pkl",
        # The bundle summary carries the ordered time-series column names
        "summary": bundle_dir / f"{bundle_dir.name}_summary.json",
    }


def read_ts_columns(summary_path: Path) -> list[str] | None:
    """Return the recorded time-series feature names, if available."""
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as handle:
        producer_summary = json.load(handle)
    ts_columns = producer_summary.get("ts_columns")
    if not ts_columns:
        return None
    return list(ts_columns)


def build_split_output_dir(
    bundle_dir: Path,
    output_dir: Path,
    output_suffix: str,
) -> Path:
    split_dir_name = f"{bundle_dir.name}{output_suffix}"
    if output_dir == bundle_dir:
        return bundle_dir / split_dir_name
    return output_dir / split_dir_name

def read_key_table(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    parse_dates = [column for column in DATE_COLUMNS if column in header.columns]
    key_table = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    missing_columns = [
        column
        for column in KEY_REQUIRED_COLUMNS
        if column not in key_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "Key table is missing required columns: "
            f"{missing_columns}."
        )
    validate_unique_keys(key_table, ["ed_stay_id"], "Deterioration key table")
    return key_table


def read_tabular_table(path: Path) -> pd.DataFrame:
    tabular = pd.read_csv(path, low_memory=False)
    if "benchmark_row_id" not in tabular.columns:
        raise ValueError("Tabular CSV must contain benchmark_row_id.")
    validate_unique_keys(tabular, ["benchmark_row_id"], "Deterioration tabular data")
    return tabular


def expand_vitals_acuity(tabular: pd.DataFrame) -> pd.DataFrame:
    """Replace triage acuity with fixed one-hot columns and a missing flag."""
    if VITALS_ACUITY_COLUMN not in tabular.columns:
        return tabular
    tabular = tabular.copy()
    values = pd.to_numeric(tabular[VITALS_ACUITY_COLUMN], errors="coerce")
    level = values.round()
    tabular = tabular.drop(columns=[VITALS_ACUITY_COLUMN])
    for keep_level in VITALS_ACUITY_KEEP_LEVELS:
        tabular[f"{VITALS_ACUITY_COLUMN}_{keep_level}"] = (level == keep_level).astype("int8")
    tabular[VITALS_ACUITY_MISSING_COLUMN] = values.isna().astype("int8")
    return tabular


def read_embedding_array(path: Path, modality: str) -> np.ndarray:
    if path.suffix != ".pkl":
        raise ValueError(
            f"{modality} embeddings must be provided as a .pkl file, got {path.suffix}."
        )
    embeddings = pd.read_pickle(path)
    validate_embedding_array(modality, embeddings)
    return embeddings


def normalise_strat_folds(key_table: pd.DataFrame) -> pd.Series:
    folds = pd.to_numeric(key_table["strat_fold"], errors="coerce")
    if folds.isna().any():
        raise ValueError("strat_fold contains missing or non-numeric values.")
    if not folds.mod(1).eq(0).all():
        raise ValueError("strat_fold must contain integer fold identifiers only.")
    folds = folds.astype(int)
    expected_folds = set(TRAIN_FOLDS) | {VALIDATION_FOLD, TEST_FOLD}
    unexpected_folds = sorted(set(folds) - expected_folds)
    if unexpected_folds:
        raise ValueError(
            "strat_fold contains values outside the expected 0-19 range: "
            f"{unexpected_folds}."
        )
    return folds


def assign_split_labels(strat_folds: pd.Series) -> pd.Series:
    split_labels = pd.Series(index=strat_folds.index, dtype="object")
    split_labels.loc[strat_folds.isin(TRAIN_FOLDS)] = "train"
    split_labels.loc[strat_folds.eq(VALIDATION_FOLD)] = "val"
    split_labels.loc[strat_folds.eq(TEST_FOLD)] = "test"
    return split_labels


def sort_stays(key_table: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in [
            "subject_id",
            "general_intime",
            "general_ecg_time",
            "benchmark_row_id",
            "ed_stay_id",
        ]
        if column in key_table.columns
    ]
    return key_table.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def keep_first_stay_per_patient(key_table: pd.DataFrame) -> pd.DataFrame:
    if key_table.empty:
        return key_table.copy()
    key_table = sort_stays(key_table)
    return key_table.drop_duplicates("subject_id", keep="first").reset_index(drop=True)


def build_split_tables(
    key_table: pd.DataFrame, include_all_val_test_stays: bool = False
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    assigned = key_table.copy().reset_index(drop=True)
    assigned["_source_row"] = assigned.index
    assigned["strat_fold"] = normalise_strat_folds(assigned)
    assigned["split"] = assign_split_labels(assigned["strat_fold"])

    split_tables = {
        split_name: sort_stays(assigned.loc[assigned["split"].eq(split_name)].copy())
        for split_name in SPLIT_NAMES
    }

    if not include_all_val_test_stays:
        split_tables["val"] = keep_first_stay_per_patient(split_tables["val"])
        split_tables["test"] = keep_first_stay_per_patient(split_tables["test"])

    selected_source_rows = {
        int(row_id)
        for split_table in split_tables.values()
        for row_id in split_table["_source_row"].tolist()
    }
    assigned["is_selected"] = assigned["_source_row"].isin(selected_source_rows)
    return assigned, split_tables


def read_reference_keys(bundle_dir: Path) -> tuple[pd.DataFrame | None, Path | None]:
    """Load all-cohort keys when splitting a complete cohort."""
    if "completeCohort" not in bundle_dir.name:
        return None, None
    reference_dir = bundle_dir.with_name(bundle_dir.name.replace("completeCohort", "allCohort"))
    paths = build_bundle_paths(reference_dir)
    if not paths["keys"].exists():
        raise ValueError(
            f"Missing allCohort reference: {paths['keys']}. "
            "Prepare the matching allCohort bundle in the same parent directory."
        )
    own_paths = build_bundle_paths(bundle_dir)
    for summary_path in (own_paths["summary"], paths["summary"]):
        if not summary_path.exists():
            raise ValueError(f"Missing bundle summary: {summary_path}. Regenerate the bundle.")
    with own_paths["summary"].open() as handle:
        own = json.load(handle)
    with paths["summary"].open() as handle:
        reference = json.load(handle)
    for field in ("patient_stay", "record", "test_mode", "test_row_limit"):
        if own.get(field) != reference.get(field):
            raise ValueError(f"Reference allCohort has incompatible {field}.")
    if reference.get("cohort") != "all":
        raise ValueError("Reference bundle must be allCohort.")
    return read_key_table(paths["keys"]), paths["keys"]


def build_complete_split_tables(
    key_table: pd.DataFrame,
    reference_keys: pd.DataFrame,
    include_all_val_test_stays: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Apply the all-cohort split policy to complete stays."""
    for frame in (key_table, reference_keys):
        validate_unique_keys(frame, ["ed_stay_id"], "Cohort keys")
    complete = reference_keys.loc[
        reference_keys["has_ecg_embedding"] & reference_keys["has_cxr_embedding"]
    ]
    if set(key_table["ed_stay_id"]) != set(complete["ed_stay_id"]):
        raise ValueError(
            "Complete cohort differs from the reference ECG/CXR subset. "
            "Regenerate the complete bundle."
        )
    columns = [
        "subject_id",
        "benchmark_row_id",
        "strat_fold",
        "general_intime",
        "has_ecg_embedding",
        "has_cxr_embedding",
    ]
    left = key_table.set_index("ed_stay_id").sort_index()
    right = complete.set_index("ed_stay_id").sort_index()
    if not left[columns].eq(right[columns]).all(axis=None):
        raise ValueError(
            "Complete cohort keys disagree with the allCohort reference. "
            "Regenerate matching bundles."
        )
    _, reference_splits = build_split_tables(reference_keys, include_all_val_test_stays)
    assigned, _ = build_split_tables(key_table, include_all_val_test_stays=True)
    splits = {
        name: sort_stays(assigned.loc[assigned["ed_stay_id"].isin(rows["ed_stay_id"])].copy())
        for name, rows in reference_splits.items()
    }
    selected = {row for frame in splits.values() for row in frame["_source_row"]}
    assigned["is_selected"] = assigned["_source_row"].isin(selected)
    return assigned, splits


def get_selected_tabular_columns(tabular: pd.DataFrame) -> list[str]:
    selected = [
        column
        for column in tabular.columns
        if column.startswith(TABULAR_PREFIXES)
    ]
    if not selected:
        raise ValueError(
            "Tabular CSV does not contain any columns starting with "
            f"{TABULAR_PREFIXES}."
        )
    return sorted(selected)


def get_numeric_tabular_columns(
    tabular: pd.DataFrame, selected_columns: list[str]
) -> list[str]:
    numeric: list[str] = []
    for column in selected_columns:
        # Summary columns are numeric even when a small sample has few unique values
        if column.startswith("labvalues_") or column.endswith(TABULAR_STAT_SUFFIXES):
            numeric.append(column)
            continue
        values = pd.to_numeric(tabular[column], errors="coerce")
        unique_values = set(values.dropna().unique().tolist())
        if not unique_values:
            continue
        if unique_values.issubset({0, 1}):
            continue
        if len(unique_values) < 10:
            continue
        numeric.append(column)
    return sorted(numeric)


def get_sequence_tabular_columns(
    tabular: pd.DataFrame, selected_columns: list[str]
) -> list[str]:
    sequence_columns: list[str] = []
    for column in selected_columns:
        non_null = tabular[column].dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            sequence_columns.append(column)
    return sorted(sequence_columns)


def get_passthrough_binary_columns(selected_columns: list[str]) -> list[str]:
    """Return indicator columns that should pass through unchanged."""
    return [
        column
        for column in selected_columns
        if column in PASSTHROUGH_BINARY_NAMES
        or column.startswith(PASSTHROUGH_BINARY_PREFIXES)
    ]


def coerce_tabular_numeric_frame(
    split_frame: pd.DataFrame,
    tabular_columns: list[str],
    sequence_columns: list[str],
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    for column in tabular_columns:
        if column in sequence_columns:
            continue
        split_frame[column] = pd.to_numeric(
            split_frame[column], errors="coerce"
        ).astype("float32")
    return split_frame


def fit_tabular_imputation_stats(
    train_frame: pd.DataFrame,
    tabular_columns: list[str],
    sequence_columns: list[str],
) -> tuple[pd.Series, dict[str, float], dict[str, int]]:
    scalar_columns = [
        column
        for column in tabular_columns
        if column not in sequence_columns
    ]
    scalar_medians = train_frame[scalar_columns].median(skipna=True)
    sequence_medians: dict[str, float] = {}
    sequence_lengths: dict[str, int] = {}
    for column in sequence_columns:
        sequences = [
            value
            for value in train_frame[column].dropna().tolist()
            if isinstance(value, (list, np.ndarray))
        ]
        lengths = [len(sequence) for sequence in sequences]
        if lengths:
            mode_length = int(pd.Series(lengths).mode().iloc[0])
        else:
            mode_length = 0
        flat_values: list[float] = []
        for sequence in sequences:
            array = np.asarray(sequence, dtype="float32")
            if array.size == 0:
                continue
            flat_values.extend(array[np.isfinite(array)].tolist())
        median_value = float(np.nan)
        if flat_values:
            median_value = float(np.nanmedian(np.asarray(flat_values, dtype="float32")))
        sequence_medians[column] = median_value
        sequence_lengths[column] = mode_length
    return scalar_medians, sequence_medians, sequence_lengths


def apply_tabular_imputation(
    split_frame: pd.DataFrame,
    tabular_columns: list[str],
    sequence_columns: list[str],
    scalar_medians: pd.Series,
    sequence_medians: dict[str, float],
    sequence_lengths: dict[str, int],
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    mask_rows: list[list[object]] = [[] for _ in range(len(split_frame))]
    for column in tabular_columns:
        if column in sequence_columns:
            median_value = sequence_medians[column]
            expected_length = sequence_lengths[column]
            column_values = split_frame[column].tolist()
            imputed_values: list[list[float]] = []
            for row_index, value in enumerate(column_values):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    filled = [median_value] * expected_length
                    mask = [1] * expected_length
                else:
                    sequence = np.asarray(value, dtype="float32").tolist()
                    filled = []
                    mask = []
                    for item in sequence:
                        if item is None or (isinstance(item, float) and np.isnan(item)):
                            filled.append(median_value)
                            mask.append(1)
                        else:
                            filled.append(float(item))
                            mask.append(0)
                imputed_values.append(filled)
                mask_rows[row_index].append(mask)
            split_frame[column] = imputed_values
            continue
        missing_flags = split_frame[column].isna().to_numpy(dtype="int8").tolist()
        split_frame[column] = split_frame[column].fillna(
            scalar_medians[column]
        ).astype("float32")
        for row_index, flag in enumerate(missing_flags):
            if sequence_columns:
                mask_rows[row_index].append([int(flag)])
            else:
                mask_rows[row_index].append(int(flag))
    return pd.concat(
        [split_frame.reset_index(drop=True), pd.Series(mask_rows, name="tab_missing")],
        axis=1,
    )


def fit_tabular_normalisation_stats(
    train_frame: pd.DataFrame,
    tabular_columns: list[str],
    sequence_columns: list[str],
) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for column in tabular_columns:
        if column in sequence_columns:
            continue
        min_value = train_frame[column].min(skipna=True)
        max_value = train_frame[column].max(skipna=True)
        stats[column] = (float(min_value), float(max_value))
    return stats


def extract_sequence_min_max(
    sequences: list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    min_values = None
    max_values = None
    for sequence in sequences:
        sequence = coerce_sequence_array(sequence)
        if sequence is None:
            continue
        # Ignore the warning for a stay with no observed value for a feature
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feature_min = np.nanmin(sequence, axis=0)
            feature_max = np.nanmax(sequence, axis=0)
        if min_values is None:
            min_values = feature_min
            max_values = feature_max
            continue
        # fmin and fmax keep later observed values from being masked by NaN
        min_values = np.fmin(min_values, feature_min)
        max_values = np.fmax(max_values, feature_max)
    return min_values, max_values


def infer_ts_feature_width(sequences: list[np.ndarray]) -> int:
    """Return the time-series feature width, or zero when no rows are present."""
    for sequence in sequences:
        array = np.asarray(sequence)
        if array.ndim == 2:
            return int(array.shape[1])
    return 0


def fit_ts_imputation_stats(train_sequences: list[np.ndarray]) -> np.ndarray:
    """Return one training median per time-series feature."""
    feature_width = infer_ts_feature_width(train_sequences)
    if feature_width == 0:
        return np.zeros((0,), dtype=np.float32)
    stacked = [
        np.asarray(sequence, dtype=np.float32)
        for sequence in train_sequences
        if np.asarray(sequence).ndim == 2
        and np.asarray(sequence).shape[0] > 0
    ]
    if not stacked:
        return np.full((feature_width,), np.nan, dtype=np.float32)
    values = np.concatenate(stacked, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(values, axis=0)
    return np.asarray(medians, dtype=np.float32)


def coerce_sequence_array(sequence: np.ndarray | object) -> np.ndarray | None:
    if not isinstance(sequence, np.ndarray) or sequence.size == 0:
        return None
    if sequence.dtype == object:
        items = [
            np.asarray(item, dtype=np.float32)
            for item in sequence
            if item is not None
        ]
        if not items:
            return None
        return np.stack(items)
    if sequence.ndim == 1:
        return np.asarray(sequence, dtype=np.float32)[None, :]
    if sequence.ndim != 2:
        raise ValueError("Embedding sequences must be 2D arrays.")
    return np.asarray(sequence, dtype=np.float32)


def validate_alignment_lengths(
    key_table: pd.DataFrame,
    tabular: pd.DataFrame,
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    ts_sequences: np.ndarray | None = None,
) -> None:
    if len(tabular) != len(key_table):
        raise ValueError("Tabular CSV row count does not match the key CSV row count.")
    if len(ecg_embeddings) != len(key_table):
        raise ValueError("ECG embedding row count does not match the key CSV row count.")
    if len(cxr_embeddings) != len(key_table):
        raise ValueError("CXR embedding row count does not match the key CSV row count.")
    if ts_sequences is not None and len(ts_sequences) != len(key_table):
        raise ValueError("TS sequence row count does not match the key CSV row count.")


def build_modality_time_and_missing(
    split_frame: pd.DataFrame,
    modality_name: str,
) -> tuple[list[list[float]], list[int]]:
    if split_frame.empty:
        return [], []

    time_column = f"{modality_name}_time"
    feature_column = f"{modality_name}_feats"
    record_values = split_frame["record"].dropna().unique().tolist()
    if len(record_values) != 1:
        raise ValueError("Split output expects exactly one record mode per split frame.")
    record_mode = record_values[0]
    reference_times = split_frame["general_intime"].tolist()
    raw_times = split_frame[time_column].tolist()
    feature_values = split_frame[feature_column].tolist()
    modality_times: list[list[float]] = []
    modality_missing: list[int] = []

    for reference_time, raw_value, features in zip(
        reference_times,
        raw_times,
        feature_values,
        strict=True,
    ):
        sequence = coerce_sequence_array(features)
        feature_count = 0 if sequence is None else int(len(sequence))
        modality_missing.append(int(feature_count == 0))

        if feature_count == 0 or record_mode == "average":
            modality_times.append([])
            continue

        timestamps = parse_time_values(raw_value, time_column)
        if len(timestamps) != feature_count:
            raise ValueError(
                f"{time_column} count does not match {feature_column} length for "
                f"record mode '{record_mode}'."
            )
        offsets = (
            (pd.DatetimeIndex(timestamps) - pd.Timestamp(reference_time))
            / pd.Timedelta(minutes=1)
        )
        modality_times.append(np.asarray(offsets, dtype=np.float32).tolist())

    return modality_times, modality_missing


def build_ts_columns(
    split_frame: pd.DataFrame,
    ts_medians: np.ndarray,
    ts_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> tuple[list[list[list[float]]], list[list[list[int]]], list[list[float]]]:
    """Build time-series values, missing-value flags and time offsets per stay."""
    ts_time, _ = build_modality_time_and_missing(split_frame, "ts")

    medians = np.asarray(ts_medians, dtype=np.float32)
    min_values, max_values = ts_normalisation_stats
    if min_values is not None and max_values is not None:
        range_values = np.asarray(max_values, dtype=np.float32) - np.asarray(
            min_values, dtype=np.float32
        )
        zero_range = range_values == 0
        safe_range = np.where(zero_range, 1.0, range_values)
    else:
        zero_range = None
        safe_range = None

    ts_feats: list[list[list[float]]] = []
    ts_missing: list[list[list[int]]] = []
    for value in split_frame["ts_feats"].tolist():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            ts_feats.append([])
            ts_missing.append([])
            continue
        mask = np.isnan(array).astype(np.int8)
        filled = np.where(np.isnan(array), medians[None, :], array).astype(np.float32)
        if min_values is not None and safe_range is not None:
            filled = (filled - np.asarray(min_values, dtype=np.float32)) / safe_range
            if zero_range is not None and zero_range.any():
                filled[:, zero_range] = 0.0
        # Use zero when training has no value for a feature and keep its missing flag
        filled = np.nan_to_num(filled, nan=0.0)
        ts_feats.append(filled.tolist())
        ts_missing.append(mask.tolist())

    return ts_feats, ts_missing, ts_time


def build_split_frame(
    split_keys: pd.DataFrame,
    tabular_lookup: pd.DataFrame,
    outcome_columns: list[str],
    selected_tabular_columns: list[str],
) -> pd.DataFrame:
    split_frame = split_keys.merge(
        tabular_lookup,
        on="benchmark_row_id",
        how="left",
        validate="one_to_one",
    )
    # An empty split is valid and only non-empty splits need the alignment check
    if not split_frame.empty and split_frame[selected_tabular_columns].isna().all(axis=None):
        raise ValueError("Selected tabular columns are entirely missing after alignment.")

    split_frame = split_frame.reset_index(drop=True)
    ts_time_columns = ["ts_time"] if "ts_time" in split_frame.columns else []
    split_frame = split_frame[
        [
            "ed_hadm_id",
            "ed_stay_id",
            "general_intime",
            "record",
            "ecg_time",
            "cxr_time",
            *ts_time_columns,
            *outcome_columns,
            *selected_tabular_columns,
        ]
    ]
    for column in ["ed_hadm_id", "ed_stay_id"]:
        numeric_values = pd.to_numeric(split_frame[column], errors="coerce")
        finite_values = numeric_values.dropna()
        if not finite_values.mod(1).eq(0).all():
            raise ValueError(f"{column} contains non-integer identifier values.")
        split_frame[column] = numeric_values.astype("Int64")
    return split_frame


def attach_embedding_columns(
    split_frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    ts_sequences: np.ndarray | None = None,
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    row_indices = split_keys["_source_row"].to_numpy(dtype=int)
    split_frame["ecg_feats"] = list(ecg_embeddings[row_indices])
    split_frame["cxr_feats"] = list(cxr_embeddings[row_indices])
    if ts_sequences is not None:
        split_frame["ts_feats"] = list(ts_sequences[row_indices])
    return split_frame


def prepare_built_split_frame_for_export(
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    split_keys: pd.DataFrame,
    split_frame: pd.DataFrame,
    export_tabular_columns: list[str],
    sequence_columns: list[str],
    categorical_columns: list[str],
    ohe_columns: list[str],
    ts_sequences: np.ndarray | None = None,
) -> pd.DataFrame:
    split_frame = attach_embedding_columns(
        split_frame,
        split_keys,
        ecg_embeddings,
        cxr_embeddings,
        ts_sequences,
    )
    if ohe_columns:
        split_frame = apply_ohe_to_frame(split_frame, categorical_columns, ohe_columns)
    return coerce_tabular_numeric_frame(
        split_frame,
        export_tabular_columns,
        sequence_columns,
    )


def prepare_split_frame_for_export(
    split_keys: pd.DataFrame,
    tabular_lookup: pd.DataFrame,
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    outcome_columns: list[str],
    selected_tabular_columns: list[str],
    export_tabular_columns: list[str],
    sequence_columns: list[str],
    categorical_columns: list[str],
    ohe_columns: list[str],
    ts_sequences: np.ndarray | None = None,
) -> pd.DataFrame:
    split_frame = build_split_frame(
        split_keys=split_keys,
        tabular_lookup=tabular_lookup,
        outcome_columns=outcome_columns,
        selected_tabular_columns=selected_tabular_columns,
    )
    return prepare_built_split_frame_for_export(
        ecg_embeddings,
        cxr_embeddings,
        split_keys,
        split_frame,
        export_tabular_columns,
        sequence_columns,
        categorical_columns,
        ohe_columns,
        ts_sequences,
    )


def finalise_split_frame_for_export(
    split_frame: pd.DataFrame,
    export_tabular_columns: list[str],
    sequence_columns: list[str],
    scalar_medians: pd.Series,
    sequence_medians: dict[str, float],
    sequence_lengths: dict[str, int],
    normalise: bool,
    normalisation_stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    split_frame = apply_tabular_imputation(
        split_frame,
        export_tabular_columns,
        sequence_columns,
        scalar_medians,
        sequence_medians,
        sequence_lengths,
    )
    if normalise:
        split_frame = apply_tabular_normalisation(split_frame, normalisation_stats)
    return split_frame


def normalise_sequence_column(
    sequences: list[np.ndarray],
    min_values: np.ndarray | None = None,
    max_values: np.ndarray | None = None,
) -> list[list[list[float]]]:
    normalised_sequences: list[list[list[float]]] = []
    if min_values is not None and max_values is not None:
        range_values = max_values - min_values
        zero_range = range_values == 0
        safe_range = np.where(zero_range, 1.0, range_values)
    else:
        zero_range = None
        safe_range = None
    for sequence in sequences:
        sequence = coerce_sequence_array(sequence)
        if sequence is None:
            normalised_sequences.append([])
            continue
        float32_sequence = np.asarray(sequence, dtype=np.float32)
        if min_values is not None and safe_range is not None:
            float32_sequence = (float32_sequence - min_values) / safe_range
            if zero_range is not None and zero_range.any():
                float32_sequence[:, zero_range] = 0.0
        normalised_sequences.append(float32_sequence.tolist())
    return normalised_sequences


def build_export_frame(
    split_frame: pd.DataFrame,
    outcome_columns: list[str],
    tabular_columns: list[str],
    sequence_columns: list[str],
    ecg_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    cxr_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    ts_medians: np.ndarray | None = None,
    ts_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> pd.DataFrame:
    scalar_columns = [column for column in tabular_columns if column not in sequence_columns]
    ed_hadm_id = split_frame["ed_hadm_id"].astype("Int64")
    ed_stay_id = split_frame["ed_stay_id"].astype("Int64")
    outcome_series = {
        column: split_frame[column].astype("Int64")
        for column in outcome_columns
    }
    scalar_values = {
        column: pd.to_numeric(split_frame[column], errors="coerce").astype("float32")
        for column in scalar_columns
    }
    if sequence_columns:
        tabular_values: list[list[list[float]]] = []
        for _, row in split_frame[tabular_columns].iterrows():
            row_values: list[list[float]] = []
            for column in tabular_columns:
                if column in sequence_columns:
                    row_values.append(list(row[column]))
                else:
                    row_values.append([float(scalar_values[column].iloc[row.name])])
            tabular_values.append(row_values)
    else:
        tabular_values = np.column_stack(
            [scalar_values[column].to_numpy(dtype="float32") for column in tabular_columns]
        ).tolist()
    tabular_missing = split_frame["tab_missing"].tolist()
    ecg_times, ecg_missing = build_modality_time_and_missing(split_frame, "ecg")
    cxr_times, cxr_missing = build_modality_time_and_missing(split_frame, "cxr")
    ecg_values = normalise_sequence_column(
        split_frame["ecg_feats"].tolist(),
        *ecg_normalisation_stats,
    )
    cxr_values = normalise_sequence_column(
        split_frame["cxr_feats"].tolist(),
        *cxr_normalisation_stats,
    )

    output_columns: dict[str, object] = {
        "ed_hadm_id": ed_hadm_id,
        "ed_stay_id": ed_stay_id,
        **outcome_series,
        "tab_feats": tabular_values,
        "tab_missing": tabular_missing,
        "ecg_feats": ecg_values,
        "ecg_time": ecg_times,
        "ecg_missing": pd.Series(ecg_missing, dtype="int8"),
        "cxr_feats": cxr_values,
        "cxr_time": cxr_times,
        "cxr_missing": pd.Series(cxr_missing, dtype="int8"),
    }

    if "ts_feats" in split_frame.columns:
        medians = (
            np.zeros((0,), dtype=np.float32) if ts_medians is None else ts_medians
        )
        ts_values, ts_missing, ts_times = build_ts_columns(
            split_frame,
            medians,
            ts_normalisation_stats,
        )
        output_columns["ts_feats"] = ts_values
        output_columns["ts_missing"] = ts_missing
        output_columns["ts_time"] = ts_times

    return pd.DataFrame(output_columns)


def build_arrow_schema(
    outcome_columns: list[str],
    use_sequence_tabular: bool,
    include_ts: bool = False,
) -> object:
    schema_fields = [
        pa.field("ed_hadm_id", pa.int64()),
        pa.field("ed_stay_id", pa.int64()),
    ]
    for outcome in outcome_columns:
        schema_fields.append(pa.field(outcome, pa.int64()))
    if use_sequence_tabular:
        schema_fields.append(pa.field("tab_feats", pa.list_(pa.list_(pa.float32()))))
        schema_fields.append(pa.field("tab_missing", pa.list_(pa.list_(pa.int8()))))
    else:
        schema_fields.append(pa.field("tab_feats", pa.list_(pa.float32())))
        schema_fields.append(pa.field("tab_missing", pa.list_(pa.int8())))
    schema_fields.append(pa.field("ecg_feats", pa.list_(pa.list_(pa.float32()))))
    schema_fields.append(pa.field("ecg_time", pa.list_(pa.float32())))
    schema_fields.append(pa.field("ecg_missing", pa.int8()))
    schema_fields.append(pa.field("cxr_feats", pa.list_(pa.list_(pa.float32()))))
    schema_fields.append(pa.field("cxr_time", pa.list_(pa.float32())))
    schema_fields.append(pa.field("cxr_missing", pa.int8()))
    if include_ts:
        schema_fields.append(pa.field("ts_feats", pa.list_(pa.list_(pa.float32()))))
        schema_fields.append(pa.field("ts_missing", pa.list_(pa.list_(pa.int8()))))
        schema_fields.append(pa.field("ts_time", pa.list_(pa.float32())))
    return pa.schema(schema_fields)


def write_split_parquet(
    split_frame: pd.DataFrame,
    output_path: Path,
    outcome_columns: list[str],
    tabular_columns: list[str],
    sequence_columns: list[str],
    ecg_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    cxr_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    ts_medians: np.ndarray | None = None,
    ts_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
    chunk_size: int = 5_000,
) -> None:
    include_ts = "ts_feats" in split_frame.columns
    schema = build_arrow_schema(outcome_columns, bool(sequence_columns), include_ts)
    writer = None
    try:
        for start in range(0, len(split_frame), chunk_size):
            chunk = split_frame.iloc[start : start + chunk_size].reset_index(drop=True)
            export_frame = build_export_frame(
                chunk,
                outcome_columns,
                tabular_columns,
                sequence_columns,
                ecg_normalisation_stats,
                cxr_normalisation_stats,
                ts_medians,
                ts_normalisation_stats,
            )
            table = pa.Table.from_pandas(export_frame, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema,
                    compression="snappy",
                    use_dictionary=True,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def build_summary(
    keys_path: Path,
    tabular_path: Path,
    ecg_embeddings_path: Path,
    cxr_embeddings_path: Path,
    output_dir: Path,
    assigned: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
    include_all_val_test_stays: bool,
    tab_feats_columns: list[str],
    numeric_tabular_columns: list[str],
    categorical_tabular_columns: list[str],
    output_suffix: str,
    normalised: bool,
    ohe_enabled: bool,
    ts_embeddings_path: Path | None = None,
    ts_feature_width: int | None = None,
    ts_columns: list[str] | None = None,
    reference_keys_path: Path | None = None,
) -> dict[str, object]:
    output_stem = keys_path.stem.removesuffix("_keys")
    outcome_columns = [
        column for column in assigned.columns if column.startswith("deterioration_")
    ]
    original_counts = assigned.groupby("split").size().to_dict()
    train_subjects = set(split_tables["train"]["subject_id"])
    val_subjects = set(split_tables["val"]["subject_id"])
    test_subjects = set(split_tables["test"]["subject_id"])
    return {
        "input_keys_path": str(keys_path),
        "completeness_modalities": ["ecg", "cxr"],
        "reference_keys_path": str(reference_keys_path) if reference_keys_path else None,
        "input_tabular_path": str(tabular_path),
        "input_ecg_embeddings_path": str(ecg_embeddings_path),
        "input_cxr_embeddings_path": str(cxr_embeddings_path),
        "input_ts_path": None if ts_embeddings_path is None else str(ts_embeddings_path),
        "ts_feature_width": ts_feature_width,
        "ts_columns": ts_columns,
        "output_dir": str(output_dir),
        "train_folds": list(TRAIN_FOLDS),
        "validation_fold": VALIDATION_FOLD,
        "test_fold": TEST_FOLD,
        "include_all_val_test_stays": include_all_val_test_stays,
        "normalised": normalised,
        "ohe": ohe_enabled,
        "n_input_rows": int(len(assigned)),
        "n_selected_rows": int(sum(len(split_table) for split_table in split_tables.values())),
        "n_rows_by_split": {
            split_name: int(len(split_table))
            for split_name, split_table in split_tables.items()
        },
        "n_unique_patients_by_split": {
            split_name: int(split_table["subject_id"].nunique())
            for split_name, split_table in split_tables.items()
        },
        "n_positive_by_split": {
            split_name: {
                outcome: int(split_table[outcome].eq(1).sum())
                for outcome in outcome_columns
            }
            for split_name, split_table in split_tables.items()
        },
        "n_rows_removed_by_first_stay_filter": {
            "val": int(original_counts.get("val", 0) - len(split_tables["val"])),
            "test": int(original_counts.get("test", 0) - len(split_tables["test"])),
        },
        "n_patient_overlap": {
            "train_val": int(len(train_subjects.intersection(val_subjects))),
            "train_test": int(len(train_subjects.intersection(test_subjects))),
            "val_test": int(len(val_subjects.intersection(test_subjects))),
        },
        # Keep the existing summary key for compatibility with downstream readers
        "tab_feats_columns": tab_feats_columns,
        "continous_columns": numeric_tabular_columns,
        "categorical_columns": categorical_tabular_columns,
        "output_files": {
            split_name: str(
                output_dir / f"{split_name}_{output_stem}{output_suffix}.parquet"
            )
            for split_name in SPLIT_NAMES
        },
        "summary_path": str(
            output_dir / f"{output_stem}{output_suffix}_split_summary.json"
        ),
    }


def write_split_outputs_sequentially(
    split_tables: dict[str, pd.DataFrame],
    tabular_lookup: pd.DataFrame,
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    outcome_columns: list[str],
    selected_tabular_columns: list[str],
    sequence_columns: list[str],
    categorical_columns: list[str],
    ohe_columns: list[str],
    scalar_medians: pd.Series,
    sequence_medians: dict[str, float],
    sequence_lengths: dict[str, int],
    normalise: bool,
    normalisation_stats: dict[str, tuple[float, float]],
    ecg_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None],
    cxr_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None],
    output_dir: Path,
    output_stem: str,
    output_suffix: str,
    ts_sequences: np.ndarray | None = None,
    ts_medians: np.ndarray | None = None,
    ts_normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_tabular_columns = [
        column for column in selected_tabular_columns if column not in categorical_columns
    ] + ohe_columns
    for split_name in SPLIT_NAMES:
        split_frame = prepare_split_frame_for_export(
            split_keys=split_tables[split_name],
            tabular_lookup=tabular_lookup,
            ecg_embeddings=ecg_embeddings,
            cxr_embeddings=cxr_embeddings,
            outcome_columns=outcome_columns,
            selected_tabular_columns=selected_tabular_columns,
            export_tabular_columns=export_tabular_columns,
            sequence_columns=sequence_columns,
            categorical_columns=categorical_columns,
            ohe_columns=ohe_columns,
            ts_sequences=ts_sequences,
        )
        split_frame = finalise_split_frame_for_export(
            split_frame,
            export_tabular_columns,
            sequence_columns,
            scalar_medians,
            sequence_medians,
            sequence_lengths,
            normalise,
            normalisation_stats,
        )
        write_split_parquet(
            split_frame,
            output_dir / f"{split_name}_{output_stem}{output_suffix}.parquet",
            outcome_columns,
            export_tabular_columns,
            sequence_columns,
            ecg_normalisation_stats if normalise else (None, None),
            cxr_normalisation_stats if normalise else (None, None),
            ts_medians,
            ts_normalisation_stats if normalise else (None, None),
        )


def print_summary(summary: dict[str, object]) -> None:
    print(f"Saved split outputs to {summary['output_dir']}.")
    for split_name in SPLIT_NAMES:
        n_rows = summary["n_rows_by_split"][split_name]
        n_patients = summary["n_unique_patients_by_split"][split_name]
        print(f"{split_name}: {n_rows} rows across {n_patients} patients")
    removed = summary["n_rows_removed_by_first_stay_filter"]
    if not summary["include_all_val_test_stays"]:
        print(
            "Removed rows by first-stay filter: "
            f"val={removed['val']}, test={removed['test']}"
        )
    overlaps = summary["n_patient_overlap"]
    if any(overlaps.values()):
        print(
            "Warning: patient overlap across selected splits detected: "
            f"train/val={overlaps['train_val']}, "
            f"train/test={overlaps['train_test']}, "
            f"val/test={overlaps['val_test']}"
        )


def main() -> None:
    args = parse_args()
    bundle_paths = build_bundle_paths(resolve_bundle_dir(args.bundle, DETERIORATION_OUTPUT_DIR))
    bundle_dir = bundle_paths["bundle_dir"]
    keys_path, tabular_path = bundle_paths["keys"], bundle_paths["tabular"]
    ecg_embeddings_path = bundle_paths["ecg_embeddings"]
    cxr_embeddings_path = bundle_paths["cxr_embeddings"]
    ts_path = bundle_paths["ts"]
    output_dir = args.output_dir if args.output_dir is not None else bundle_dir
    output_stem = bundle_dir.name

    key_table = read_key_table(keys_path)
    reference_keys, reference_keys_path = read_reference_keys(bundle_dir)
    if reference_keys is None:
        assigned, split_tables = build_split_tables(key_table, args.include_all_val_test_stays)
    else:
        assigned, split_tables = build_complete_split_tables(
            key_table, reference_keys, args.include_all_val_test_stays
        )
    tabular = read_tabular_table(tabular_path)
    if args.ohe:
        # Expand acuity first so its columns use the same path as other indicators
        tabular = expand_vitals_acuity(tabular)
    ecg_embeddings = read_embedding_array(ecg_embeddings_path, "ECG")
    cxr_embeddings = read_embedding_array(cxr_embeddings_path, "CXR")
    # Time-series data is optional and is present only for --record all bundles
    ts_present = ts_path.exists()
    ts_sequences = read_embedding_array(ts_path, "TS") if ts_present else None
    validate_alignment_lengths(
        key_table, tabular, ecg_embeddings, cxr_embeddings, ts_sequences
    )
    outcome_columns = [
        column for column in key_table.columns if column.startswith("deterioration_")
    ]

    selected_tabular_columns = get_selected_tabular_columns(tabular)
    tabular_lookup = tabular[["benchmark_row_id", *selected_tabular_columns]].copy()
    numeric_columns = get_numeric_tabular_columns(tabular, selected_tabular_columns)
    sequence_columns = get_sequence_tabular_columns(tabular, selected_tabular_columns)
    passthrough_columns = get_passthrough_binary_columns(selected_tabular_columns)
    # Keep raw acuity as one numeric feature unless --ohe already expanded it
    if (
        not args.ohe
        and VITALS_ACUITY_COLUMN in selected_tabular_columns
        and VITALS_ACUITY_COLUMN not in numeric_columns
    ):
        numeric_columns = sorted([*numeric_columns, VITALS_ACUITY_COLUMN])
    categorical_columns = [
        column
        for column in selected_tabular_columns
        if column not in numeric_columns
        and column not in sequence_columns
        and column not in passthrough_columns
    ]

    train_frame = build_split_frame(
        split_keys=split_tables["train"],
        tabular_lookup=tabular_lookup,
        outcome_columns=outcome_columns,
        selected_tabular_columns=selected_tabular_columns,
    )
    if args.ohe:
        ohe_columns = fit_ohe_columns(train_frame, categorical_columns)
        train_frame = apply_ohe_to_frame(train_frame, categorical_columns, ohe_columns)
    else:
        ohe_columns = []
    export_tabular_columns = [
        column for column in selected_tabular_columns if column not in categorical_columns
    ] + ohe_columns
    # Report scalar and indicator columns in their export order
    numeric_column_set = set(numeric_columns)
    indicator_column_set = set(passthrough_columns) | set(ohe_columns)
    final_continuous_columns = [
        column for column in export_tabular_columns if column in numeric_column_set
    ]
    final_categorical_columns = [
        column for column in export_tabular_columns if column in indicator_column_set
    ]
    train_frame = prepare_built_split_frame_for_export(
        ecg_embeddings=ecg_embeddings,
        cxr_embeddings=cxr_embeddings,
        split_keys=split_tables["train"],
        split_frame=train_frame,
        export_tabular_columns=export_tabular_columns,
        sequence_columns=sequence_columns,
        categorical_columns=categorical_columns,
        ohe_columns=[],
    )
    scalar_medians, sequence_medians, sequence_lengths = fit_tabular_imputation_stats(
        train_frame,
        export_tabular_columns,
        sequence_columns,
    )
    train_frame = finalise_split_frame_for_export(
        train_frame,
        export_tabular_columns,
        sequence_columns,
        scalar_medians,
        sequence_medians,
        sequence_lengths,
        False,
        {},
    )
    if ts_present:
        train_ts_rows = list(
            ts_sequences[split_tables["train"]["_source_row"].to_numpy(dtype=int)]
        )
        ts_medians = fit_ts_imputation_stats(train_ts_rows)
        ts_feature_width = infer_ts_feature_width(list(ts_sequences))
        ts_columns = read_ts_columns(bundle_paths["summary"])
    else:
        ts_medians = None
        ts_feature_width = None
        ts_columns = None
    output_suffix = "_normalised" if args.normalise else ""
    if args.ohe:
        output_suffix = f"{output_suffix}_ohe"
    split_output_dir = build_split_output_dir(bundle_dir, output_dir, output_suffix)
    if args.normalise:
        normalisation_stats = fit_tabular_normalisation_stats(
            train_frame,
            numeric_columns,
            sequence_columns,
        )
        ecg_normalisation_stats = fit_embedding_normalisation_stats(
            split_tables["train"],
            ecg_embeddings,
            sequence_min_max=extract_sequence_min_max,
        )
        cxr_normalisation_stats = fit_embedding_normalisation_stats(
            split_tables["train"],
            cxr_embeddings,
            sequence_min_max=extract_sequence_min_max,
        )
        ts_normalisation_stats = (
            fit_embedding_normalisation_stats(
                split_tables["train"], ts_sequences, sequence_min_max=extract_sequence_min_max
            )
            if ts_present
            else (None, None)
        )
    else:
        normalisation_stats = {}
        ecg_normalisation_stats = (None, None)
        cxr_normalisation_stats = (None, None)
        ts_normalisation_stats = (None, None)
    summary = build_summary(
        keys_path=keys_path,
        tabular_path=tabular_path,
        ecg_embeddings_path=ecg_embeddings_path,
        cxr_embeddings_path=cxr_embeddings_path,
        output_dir=split_output_dir,
        assigned=assigned,
        split_tables=split_tables,
        include_all_val_test_stays=args.include_all_val_test_stays,
        tab_feats_columns=export_tabular_columns,
        numeric_tabular_columns=final_continuous_columns,
        categorical_tabular_columns=final_categorical_columns,
        output_suffix=output_suffix,
        normalised=args.normalise,
        ohe_enabled=args.ohe,
        ts_embeddings_path=ts_path if ts_present else None,
        ts_feature_width=ts_feature_width,
        ts_columns=ts_columns,
        reference_keys_path=reference_keys_path,
    )
    write_split_outputs_sequentially(
        split_tables,
        tabular_lookup,
        ecg_embeddings,
        cxr_embeddings,
        outcome_columns,
        selected_tabular_columns,
        sequence_columns,
        categorical_columns,
        ohe_columns,
        scalar_medians,
        sequence_medians,
        sequence_lengths,
        args.normalise,
        normalisation_stats,
        ecg_normalisation_stats,
        cxr_normalisation_stats,
        split_output_dir,
        output_stem,
        output_suffix,
        ts_sequences,
        ts_medians,
        ts_normalisation_stats,
    )
    summary_path = split_output_dir / f"{output_stem}{output_suffix}_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print_summary(summary)


if __name__ == "__main__":
    main()
