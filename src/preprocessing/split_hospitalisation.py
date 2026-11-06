"""Split a hospitalisation bundle into train, validation and test parquet files.

The splitter discovers the modalities from the producer summary, creates one shared stratified
80/10/10 partition, fits imputation, encoding and normalisation on training rows only, and writes
one parquet file per split with a summary JSON.

Time offsets use the previous-window or current-window reference stored in the key table. Time
series keep a missing-value mask before imputation, and each patient remains one output row.
Use ``--ohe`` to retain string categorical features as one-hot columns. Without it they are
dropped and recorded in the summary.

Usage::

    python -m src.preprocessing.split_hospitalisation \
        --bundle ../outputs/hospitalisation/3.1/hospitalisation_dualWindow_averageRecord_requiredHistory \
        --normalise --ohe
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
HOSPITALISATION_OUTPUT_DIR = OUTPUTS_DIR / "hospitalisation" / "3.1"

OUTCOME_PREFIX = "hospitalisation_"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_LABELS_FILENAME = "split_labels.csv"

SPLIT_STRATEGY = "shared_stratified_80_10_10"
DEFAULT_SEED = 42
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1
# Pool strata smaller than this before splitting
MIN_STRATUM_SIZE = 20

# Static feature families with timestamps and labels excluded
TABULAR_PREFIXES = (
    "demographics_", "biometrics_", "encounter_",
    "history_", "vitals_", "conditions_",
    "medications_", "careorders_", "labvalues_",
)
TABULAR_ID_COLUMNS = ("benchmark_row_id", "subject_id", "ed_stay_id", "ed_hadm_id")
PARQUET_ID_COLUMNS = ("ed_hadm_id", "ed_stay_id", "subject_id")

# Producer time-series statistics
TS_STAT_NAMES = (
    "mean", "median", "min", "max", "std",
    "first", "last", "rate_change", "coeff",
)
TABULAR_STAT_SUFFIXES = tuple(f"_{stat}" for stat in TS_STAT_NAMES)
TS_FAMILY_PREFIXES = ("vitals_", "labvalues_")

VITALS_ACUITY_COLUMN = "vitals_acuity"
VITALS_ACUITY_MISSING_COLUMN = "vitals_acuity_missing"
# Hospitalisation acuity levels are 0-3
VITALS_ACUITY_KEEP_LEVELS = (0, 1, 2, 3)

# Binary indicators that are already encoded and need no scaling
PASSTHROUGH_BINARY_NAMES = ("demographics_gender",)
PASSTHROUGH_BINARY_PREFIXES = ("vitals_acuity_",)

# Columns used to build the shared strata
GENDER_COLUMN = "demographics_gender"
RACE_COLUMN = "demographics_ethnicity"
MALE_CODE = 1

# Modality arrays use the pickle format written by the producer
EMBEDDING_FORMAT = "pickle"
EMBEDDING_EXTENSION = "pkl"

DATE_COLUMNS = (
    "general_intime", "prediction_time", "prev_start",
    "prev_end", "curr_start", "curr_end",
)
KEY_REQUIRED_COLUMNS = ("subject_id", "ed_stay_id", "ed_hadm_id", "general_intime", "record")

# Key columns used as time references for each window
WINDOW_TIME_REFERENCE = {"prev": "general_intime", "curr": "prediction_time"}


@dataclass(frozen=True)
class ModalitySpec:
    """One non-static modality in the bundle."""

    name: str
    kind: str  # "ts" | "ecg" | "cxr" | "notes"
    window: str  # "prev" | "curr"

    @property
    def per_step_mask(self) -> bool:
        """Time-series blocks carry a mask and embeddings carry a scalar flag."""
        return self.kind == "ts"

    @property
    def time_reference(self) -> str:
        return WINDOW_TIME_REFERENCE[self.window]

    @property
    def artefact_suffix(self) -> str:
        return f"_{self.name}" if self.kind == "ts" else f"_{self.name}_embeddings"


def parse_modality_specs(names: list[str]) -> tuple[ModalitySpec, ...]:
    """Turn the producer summary's modality names into specs."""
    specs: list[ModalitySpec] = []
    for name in names:
        if name == "static":
            continue
        kind, _, window = name.rpartition("_")
        if kind not in ("ts", "ecg", "cxr", "notes") or window not in ("prev", "curr"):
            raise ValueError(
                f"Unrecognised modality '{name}'. Expected <ts|ecg|cxr|notes>_<prev|curr>."
            )
        specs.append(ModalitySpec(name, kind, window))
    if not specs:
        raise ValueError("Bundle summary lists no non-static modalities.")
    return tuple(specs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a hospitalisation bundle into train/val/test parquet files, fitting every "
            "statistic on the training split only."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help=(
            "Bundle folder written by prepare_hospitalisation.py, for example "
            "hospitalisation_dualWindow_averageRecord_requiredHistory. A bare name is resolved "
            f"under {HOSPITALISATION_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Parent directory for the split outputs. Defaults to a bundle-named subfolder "
            "inside the bundle directory."
        ),
    )
    parser.add_argument(
        "--normalise",
        "-n",
        action="store_true",
        help=(
            "Min-max scale continuous tabular columns and every modality sequence to [0, 1] "
            "using the training split, and append _normalised to the output suffix."
        ),
    )
    parser.add_argument(
        "--ohe",
        action="store_true",
        help=(
            "One-hot encode the string categoricals using the training split's categories and "
            "append _ohe to the output suffix. Without it those columns are dropped from the "
            "tabular feature array and recorded in the summary."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the stratified split.",
    )
    parser.add_argument(
        "--split-ids-dir",
        type=Path,
        default=None,
        help=(
            "Reuse patient assignments from a previous split output directory. Use this to "
            "keep patients in the same train/validation/test split across prev, curr and dual "
            "observation-window bundles from the same history cohort. Reads a shared "
            f"{SPLIT_LABELS_FILENAME} (subject_id,split), including bundle-prefixed names. "
            "Otherwise read the per-outcome "
            "training_ids_{outcome}.csv / validation_ids_{outcome}.csv / "
            "testing_ids_{outcome}.csv trio, which needs --split-ids-outcome."
        ),
    )
    parser.add_argument(
        "--split-ids-outcome",
        type=str,
        default=None,
        help=(
            "Outcome whose per-outcome id files to read, for example in_hosp_death. "
            f"Not needed when the directory holds a shared {SPLIT_LABELS_FILENAME}."
        ),
    )
    return parser.parse_args()


def read_producer_summary(bundle_dir: Path) -> dict[str, object]:
    path = bundle_dir / f"{bundle_dir.name}_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundle summary not found at {path}. Run prepare_hospitalisation.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_bundle_paths(
    bundle_dir: Path,
    specs: tuple[ModalitySpec, ...],
    embedding_format: str = EMBEDDING_FORMAT,
) -> dict[str, Path]:
    """Resolve the input files for one bundle."""
    if embedding_format != EMBEDDING_FORMAT:
        raise ValueError(
            f"Bundle summary declares embedding_format {embedding_format!r}, but this stage reads "
            f"{EMBEDDING_FORMAT!r} (.{EMBEDDING_EXTENSION}) files only. Re-run "
            "prepare_hospitalisation.py to rebuild the bundle."
        )
    extension = EMBEDDING_EXTENSION
    stem = bundle_dir.name
    paths = {
        "bundle_dir": bundle_dir,
        "keys": bundle_dir / f"{stem}_keys.csv",
        "tabular": bundle_dir / f"{stem}_tabular.csv",
        "summary": bundle_dir / f"{stem}_summary.json",
    }
    for spec in specs:
        paths[spec.name] = bundle_dir / f"{stem}{spec.artefact_suffix}.{extension}"
    return paths


def read_key_table(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    parse_dates = [column for column in DATE_COLUMNS if column in header.columns]
    key_table = pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    missing = [
        column for column in KEY_REQUIRED_COLUMNS if column not in key_table.columns
    ]
    if missing:
        raise ValueError(f"Key table is missing required columns: {missing}.")
    validate_unique_keys(key_table, ["subject_id"], "Hospitalisation key table")
    validate_unique_keys(key_table, ["ed_stay_id"], "Hospitalisation key table")
    return key_table


def read_tabular_table(path: Path) -> pd.DataFrame:
    tabular = pd.read_csv(path, low_memory=False)
    if "subject_id" not in tabular.columns:
        raise ValueError("Tabular CSV must contain subject_id.")
    validate_unique_keys(tabular, ["subject_id"], "Hospitalisation tabular data")
    return tabular


def read_embedding_array(path: Path, modality: str) -> np.ndarray:
    """Read one modality array from pickle."""
    if path.suffix != f".{EMBEDDING_EXTENSION}":
        raise ValueError(
            f"{modality} embeddings must be a .{EMBEDDING_EXTENSION} file, got {path.suffix}."
        )
    embeddings = pd.read_pickle(path)
    validate_embedding_array(modality, embeddings)
    return embeddings


def validate_alignment_lengths(
    key_table: pd.DataFrame,
    tabular: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> None:
    if len(tabular) != len(key_table):
        raise ValueError("Tabular CSV row count does not match the key CSV row count.")
    for name, array in embeddings.items():
        if len(array) != len(key_table):
            raise ValueError(
                f"{name} row count ({len(array)}) does not match the key CSV row count "
                f"({len(key_table)})."
            )


def validate_presence_flags(
    key_table: pd.DataFrame, modality: str, embeddings: np.ndarray
) -> None:
    """Check that modality flags match empty and non-empty sequences."""
    column = f"has_{modality}"
    if column not in key_table.columns:
        return
    flags = key_table[column].to_numpy(dtype=bool)
    non_empty = np.array([sequence.shape[0] > 0 for sequence in embeddings])
    if not np.array_equal(flags, non_empty):
        mismatched = int((flags != non_empty).sum())
        raise ValueError(
            f"{column} disagrees with {modality} sequence emptiness for {mismatched} rows."
        )


def infer_outcome_columns(frame: pd.DataFrame) -> list[str]:
    outcomes = sorted(
        column for column in frame.columns if column.startswith(OUTCOME_PREFIX)
    )
    if not outcomes:
        raise ValueError(f"No columns carrying the '{OUTCOME_PREFIX}' prefix were found.")
    return outcomes


def expand_vitals_acuity(tabular: pd.DataFrame) -> pd.DataFrame:
    """Replace acuity with fixed level indicators and a missing flag."""
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


def get_selected_tabular_columns(tabular: pd.DataFrame) -> list[str]:
    selected = [
        column
        for column in tabular.columns
        if column.startswith(TABULAR_PREFIXES) and column not in TABULAR_ID_COLUMNS
    ]
    if not selected:
        raise ValueError(
            f"Tabular CSV contains no columns starting with {TABULAR_PREFIXES}."
        )
    return sorted(selected)


def classify_tabular_columns(
    tabular: pd.DataFrame, selected_columns: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Classify selected columns as continuous, binary or categorical."""
    continuous: list[str] = []
    passthrough: list[str] = []
    categorical: list[str] = []
    for column in selected_columns:
        if column in PASSTHROUGH_BINARY_NAMES or column.startswith(
            PASSTHROUGH_BINARY_PREFIXES
        ):
            passthrough.append(column)
            continue
        if column.startswith(TS_FAMILY_PREFIXES) and column.endswith(
            TABULAR_STAT_SUFFIXES
        ):
            continuous.append(column)
            continue
        values = pd.to_numeric(tabular[column], errors="coerce")
        if values.isna().all() and not tabular[column].isna().all():
            categorical.append(column)
            continue
        unique_values = set(values.dropna().unique().tolist())
        if not unique_values:
            continue
        if unique_values.issubset({0, 1}):
            passthrough.append(column)
            continue
        if len(unique_values) < 10:
            categorical.append(column)
            continue
        continuous.append(column)
    return sorted(continuous), sorted(passthrough), sorted(categorical)


def build_stratification_frame(
    key_table: pd.DataFrame, tabular: pd.DataFrame
) -> pd.DataFrame:
    """Build the gender, race and outcome data used for stratification."""
    lookup = tabular.set_index("subject_id")
    subjects = key_table["subject_id"]
    frame = pd.DataFrame(index=range(len(key_table)))
    if GENDER_COLUMN in lookup.columns:
        gender = pd.to_numeric(
            lookup[GENDER_COLUMN].reindex(subjects).to_numpy(), errors="coerce"
        )
        frame["gender"] = np.where(gender == MALE_CODE, "M", "F")
    else:
        frame["gender"] = "U"
    if RACE_COLUMN in lookup.columns:
        frame["race_group"] = (
            pd.Series(lookup[RACE_COLUMN].reindex(subjects).to_numpy())
            .fillna("UNKNOWN")
            .astype(str)
            .to_numpy()
        )
    else:
        frame["race_group"] = "UNKNOWN"
    for outcome in infer_outcome_columns(key_table):
        frame[outcome] = key_table[outcome].to_numpy()
    frame["subject_id"] = subjects.to_numpy()
    return frame


def build_strata(frame: pd.DataFrame, min_size: int = MIN_STRATUM_SIZE) -> pd.Series:
    """Build stratification groups and pool groups that are too small."""
    outcomes = sorted(
        column for column in frame.columns if column.startswith(OUTCOME_PREFIX)
    )
    label_pattern = frame[outcomes].astype("Int64").astype(str).agg("".join, axis=1)
    strata = label_pattern + "|" + frame["gender"].astype(str) + "|" + frame[
        "race_group"
    ].astype(str)

    counts = strata.value_counts()
    undersized = set(counts[counts < min_size].index)
    if undersized:
        strata = strata.where(~strata.isin(undersized), label_pattern + "|POOLED")
    counts = strata.value_counts()
    undersized = set(counts[counts < min_size].index)
    if undersized:
        strata = strata.where(~strata.isin(undersized), "POOLED")
    counts = strata.value_counts()
    undersized = set(counts[counts < min_size].index)
    if undersized and len(counts) > len(undersized):
        largest = counts.index[0]
        strata = strata.where(~strata.isin(undersized), largest)
    return strata.rename("stratum")


def assign_shared_split(
    frame: pd.DataFrame, strata: pd.Series, seed: int = DEFAULT_SEED
) -> pd.Series:
    """Assign one shared, reproducible train/val/test label per row."""
    from sklearn.model_selection import train_test_split

    n_rows = len(frame)
    labels = pd.Series(["train"] * n_rows, index=range(n_rows), dtype="object")
    if n_rows == 0:
        return labels
    indices = np.arange(n_rows)
    strata_values = strata.to_numpy()
    train_index, holdout_index = train_test_split(
        indices,
        test_size=1 - TRAIN_FRACTION,
        random_state=seed,
        stratify=strata_values,
    )
    val_index, test_index = train_test_split(
        holdout_index,
        test_size=0.5,
        random_state=seed,
        stratify=strata_values[holdout_index],
    )
    labels.iloc[train_index] = "train"
    labels.iloc[val_index] = "val"
    labels.iloc[test_index] = "test"
    return labels


def write_split_labels(
    assigned: pd.DataFrame, output_path: Path
) -> None:
    """Save the patient-to-split assignment for reuse across bundles."""
    labels = assigned[["subject_id", "split"]].sort_values("subject_id", kind="stable")
    labels.to_csv(output_path, index=False)


def read_external_split_labels(
    frame: pd.DataFrame, ids_dir: Path, outcome: str | None = None
) -> pd.Series:
    """Read a shared split file or the older per-outcome split files."""
    # Accept the bare name or the bundle-prefixed name
    shared_path = ids_dir / SPLIT_LABELS_FILENAME
    if not shared_path.exists():
        candidates = sorted(ids_dir.glob(f"*_{SPLIT_LABELS_FILENAME}"))
        if len(candidates) > 1:
            raise ValueError(
                f"{ids_dir} holds {len(candidates)} *_{SPLIT_LABELS_FILENAME} files. Keep one "
                f"or rename the intended partition to {SPLIT_LABELS_FILENAME}."
            )
        if candidates:
            shared_path = candidates[0]
    if shared_path.exists():
        table = pd.read_csv(shared_path)
        missing = {"subject_id", "split"} - set(table.columns)
        if missing:
            raise ValueError(f"{shared_path} is missing columns: {sorted(missing)}.")
        unknown = set(table["split"]) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"{shared_path} carries unknown split labels: {sorted(unknown)}.")
        assignment = dict(
            zip(table["subject_id"].astype("int64"), table["split"], strict=True)
        )
        source = shared_path
    else:
        if not outcome:
            raise ValueError(
                f"No {SPLIT_LABELS_FILENAME} in {ids_dir}, so --split-ids-outcome is required "
                "to select which per-outcome id files to read."
            )
        filenames = {
            "train": f"training_ids_{outcome}.csv",
            "val": f"validation_ids_{outcome}.csv",
            "test": f"testing_ids_{outcome}.csv",
        }
        assignment = {}
        for split_name, filename in filenames.items():
            path = ids_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"External split ids not found at {path}.")
            for subject_id in pd.read_csv(path)["subject_id"].astype("int64").tolist():
                assignment[subject_id] = split_name
        source = ids_dir

    labels = frame["subject_id"].astype("int64").map(assignment)
    unassigned = int(labels.isna().sum())
    if unassigned:
        raise ValueError(
            f"{unassigned} cohort patients appear in none of the external split ids at "
            f"{source}."
        )
    return labels.reset_index(drop=True)


def build_split_tables(
    key_table: pd.DataFrame, split_labels: pd.Series
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    assigned = key_table.copy().reset_index(drop=True)
    assigned["_source_row"] = assigned.index
    assigned["split"] = split_labels.to_numpy()
    split_tables = {
        split_name: assigned.loc[assigned["split"].eq(split_name)]
        .sort_values(["subject_id"], kind="stable")
        .reset_index(drop=True)
        for split_name in SPLIT_NAMES
    }
    return assigned, split_tables


def validate_split_invariants(
    assigned: pd.DataFrame, split_tables: dict[str, pd.DataFrame]
) -> None:
    """Every row must carry exactly one split label, and the tables must cover every row."""
    counts = assigned["split"].value_counts(dropna=False)
    unknown = set(counts.index) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(
            f"Rows must carry exactly one label from {SPLIT_NAMES}. Found {sorted(unknown)}."
        )
    # Check overlap first so the error identifies the duplicated patients
    seen: set[int] = set()
    for name, table in split_tables.items():
        subjects = set(table["subject_id"].tolist())
        overlap = seen & subjects
        if overlap:
            raise ValueError(
                f"Split '{name}' shares {len(overlap)} patients with an earlier split."
            )
        seen |= subjects
    total = sum(len(table) for table in split_tables.values())
    if total != len(assigned):
        raise ValueError(
            f"Split tables cover {total} rows but the key table has {len(assigned)}."
        )


def coerce_tabular_numeric_frame(
    split_frame: pd.DataFrame, tabular_columns: list[str]
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    for column in tabular_columns:
        split_frame[column] = pd.to_numeric(
            split_frame[column], errors="coerce"
        ).astype("float32")
    return split_frame


def fit_tabular_imputation_stats(
    train_frame: pd.DataFrame, tabular_columns: list[str]
) -> pd.Series:
    return train_frame[tabular_columns].median(skipna=True)


def apply_tabular_imputation(
    split_frame: pd.DataFrame, tabular_columns: list[str], medians: pd.Series
) -> pd.DataFrame:
    """Fill missing tabular values with training medians and record their mask."""
    split_frame = split_frame.copy()
    mask_rows: list[list[int]] = [[] for _ in range(len(split_frame))]
    for column in tabular_columns:
        missing_flags = split_frame[column].isna().to_numpy(dtype="int8").tolist()
        split_frame[column] = (
            split_frame[column].fillna(medians[column]).astype("float32")
        )
        for row_index, flag in enumerate(missing_flags):
            mask_rows[row_index].append(int(flag))
    return pd.concat(
        [split_frame.reset_index(drop=True), pd.Series(mask_rows, name="tab_missing")],
        axis=1,
    )


def fit_tabular_normalisation_stats(
    train_frame: pd.DataFrame, numeric_columns: list[str]
) -> dict[str, tuple[float, float]]:
    return {
        column: (
            float(train_frame[column].min(skipna=True)),
            float(train_frame[column].max(skipna=True)),
        )
        for column in numeric_columns
    }


def coerce_sequence_array(sequence: object) -> np.ndarray | None:
    if not isinstance(sequence, np.ndarray) or sequence.size == 0:
        return None
    if sequence.ndim == 1:
        return np.asarray(sequence, dtype=np.float32)[None, :]
    if sequence.ndim != 2:
        raise ValueError("Embedding sequences must be 2D arrays.")
    return np.asarray(sequence, dtype=np.float32)


def extract_sequence_min_max(
    sequences: list[np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    min_values = None
    max_values = None
    for sequence in sequences:
        array = coerce_sequence_array(sequence)
        if array is None:
            continue
        with warnings.catch_warnings():
            # Sparse time-series rows may contain an all-NaN feature
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feature_min = np.nanmin(array, axis=0)
            feature_max = np.nanmax(array, axis=0)
        if min_values is None:
            min_values, max_values = feature_min, feature_max
            continue
        # Ignore all-NaN rows when another row contains the feature
        min_values = np.fmin(min_values, feature_min)
        max_values = np.fmax(max_values, feature_max)
    return min_values, max_values


def fit_sequence_imputation_stats(train_sequences: list[np.ndarray]) -> np.ndarray:
    """Per-feature medians over the training rows (NaN for features never observed)."""
    feature_width = 0
    for sequence in train_sequences:
        array = np.asarray(sequence)
        if array.ndim == 2:
            feature_width = int(array.shape[1])
            break
    if feature_width == 0:
        return np.zeros((0,), dtype=np.float32)
    stacked = [
        np.asarray(sequence, dtype=np.float32)
        for sequence in train_sequences
        if np.asarray(sequence).ndim == 2 and np.asarray(sequence).shape[0] > 0
    ]
    if not stacked:
        return np.full((feature_width,), np.nan, dtype=np.float32)
    values = np.concatenate(stacked, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(values, axis=0)
    return np.asarray(medians, dtype=np.float32)


def build_modality_time_and_missing(
    split_frame: pd.DataFrame, spec: ModalitySpec
) -> tuple[list[list[float]], list[int]]:
    """Return minute offsets and missing flags for one modality."""
    if split_frame.empty:
        return [], []
    time_column = f"{spec.name}_time"
    feature_column = f"{spec.name}_feats"
    record_values = split_frame["record"].dropna().unique().tolist()
    if len(record_values) != 1:
        raise ValueError("Split output expects exactly one record mode per split frame.")
    record_mode = record_values[0]

    reference_times = split_frame[spec.time_reference].tolist()
    raw_times = (
        split_frame[time_column].tolist()
        if time_column in split_frame.columns
        else [None] * len(split_frame)
    )
    feature_values = split_frame[feature_column].tolist()

    modality_times: list[list[float]] = []
    modality_missing: list[int] = []
    for reference_time, raw_value, features in zip(
        reference_times, raw_times, feature_values, strict=True
    ):
        sequence = coerce_sequence_array(features)
        feature_count = 0 if sequence is None else int(len(sequence))
        modality_missing.append(int(feature_count == 0))
        if feature_count == 0 or record_mode == "average":
            modality_times.append([])
            continue
        timestamps = parse_time_values(raw_value, time_column)
        if not timestamps:
            modality_times.append([])
            continue
        if len(timestamps) != feature_count:
            raise ValueError(
                f"{time_column} count ({len(timestamps)}) does not match {feature_column} "
                f"length ({feature_count}) for record mode '{record_mode}'."
            )
        offsets = (
            pd.DatetimeIndex(timestamps) - pd.Timestamp(reference_time)
        ) / pd.Timedelta(minutes=1)
        modality_times.append(np.asarray(offsets, dtype=np.float32).tolist())
    return modality_times, modality_missing


def build_masked_sequence_columns(
    split_frame: pd.DataFrame,
    spec: ModalitySpec,
    medians: np.ndarray,
    normalisation_stats: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> tuple[list[list[list[float]]], list[list[list[int]]]]:
    """Impute a time-series block and return its values and missing mask."""
    medians = np.asarray(medians, dtype=np.float32)
    min_values, max_values = normalisation_stats
    if min_values is not None and max_values is not None:
        range_values = np.asarray(max_values, dtype=np.float32) - np.asarray(
            min_values, dtype=np.float32
        )
        zero_range = range_values == 0
        safe_range = np.where(zero_range, 1.0, range_values)
    else:
        zero_range = None
        safe_range = None

    feats: list[list[list[float]]] = []
    missing: list[list[list[int]]] = []
    for value in split_frame[f"{spec.name}_feats"].tolist():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.size == 0:
            feats.append([])
            missing.append([])
            continue
        mask = np.isnan(array).astype(np.int8)
        filled = np.where(np.isnan(array), medians[None, :], array).astype(np.float32)
        if min_values is not None and safe_range is not None:
            filled = (filled - np.asarray(min_values, dtype=np.float32)) / safe_range
            if zero_range is not None and zero_range.any():
                filled[:, zero_range] = 0.0
        # Use zero for features never observed in training
        filled = np.nan_to_num(filled, nan=0.0)
        feats.append(filled.tolist())
        missing.append(mask.tolist())
    return feats, missing


def normalise_sequence_column(
    sequences: list[np.ndarray],
    min_values: np.ndarray | None = None,
    max_values: np.ndarray | None = None,
) -> list[list[list[float]]]:
    normalised: list[list[list[float]]] = []
    if min_values is not None and max_values is not None:
        range_values = max_values - min_values
        zero_range = range_values == 0
        safe_range = np.where(zero_range, 1.0, range_values)
    else:
        zero_range = None
        safe_range = None
    for sequence in sequences:
        array = coerce_sequence_array(sequence)
        if array is None:
            normalised.append([])
            continue
        values = np.asarray(array, dtype=np.float32)
        if min_values is not None and safe_range is not None:
            values = (values - min_values) / safe_range
            if zero_range is not None and zero_range.any():
                values[:, zero_range] = 0.0
        normalised.append(values.tolist())
    return normalised


def build_split_frame(
    split_keys: pd.DataFrame,
    tabular_lookup: pd.DataFrame,
    outcome_columns: list[str],
    selected_tabular_columns: list[str],
    specs: tuple[ModalitySpec, ...],
) -> pd.DataFrame:
    split_frame = split_keys.merge(
        tabular_lookup, on="subject_id", how="left", validate="one_to_one"
    )
    if not split_frame.empty and split_frame[selected_tabular_columns].isna().all(
        axis=None
    ):
        raise ValueError("Selected tabular columns are entirely missing after alignment.")
    split_frame = split_frame.reset_index(drop=True)
    time_columns = [
        f"{spec.name}_time"
        for spec in specs
        if f"{spec.name}_time" in split_frame.columns
    ]
    reference_columns = sorted({spec.time_reference for spec in specs})
    keep = [
        *PARQUET_ID_COLUMNS,
        *reference_columns,
        "record",
        *time_columns,
        *outcome_columns,
        *selected_tabular_columns,
    ]
    split_frame = split_frame[keep]
    for column in PARQUET_ID_COLUMNS:
        numeric_values = pd.to_numeric(split_frame[column], errors="coerce")
        finite_values = numeric_values.dropna()
        if not finite_values.mod(1).eq(0).all():
            raise ValueError(f"{column} contains non-integer identifier values.")
        split_frame[column] = numeric_values.astype("Int64")
    return split_frame


def attach_embedding_columns(
    split_frame: pd.DataFrame,
    split_keys: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    row_indices = split_keys["_source_row"].to_numpy(dtype=int)
    for name, array in embeddings.items():
        split_frame[f"{name}_feats"] = list(array[row_indices])
    return split_frame


def build_export_frame(
    split_frame: pd.DataFrame,
    outcome_columns: list[str],
    tabular_columns: list[str],
    specs: tuple[ModalitySpec, ...],
    normalisation_stats: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
    medians: dict[str, np.ndarray],
) -> pd.DataFrame:
    output: dict[str, object] = {
        column: split_frame[column].astype("Int64") for column in PARQUET_ID_COLUMNS
    }
    for column in outcome_columns:
        output[column] = split_frame[column].astype("Int64")
    output["tab_feats"] = np.column_stack(
        [
            pd.to_numeric(split_frame[column], errors="coerce")
            .astype("float32")
            .to_numpy()
            for column in tabular_columns
        ]
    ).tolist()
    output["tab_missing"] = split_frame["tab_missing"].tolist()

    for spec in specs:
        times, missing = build_modality_time_and_missing(split_frame, spec)
        stats = normalisation_stats.get(spec.name, (None, None))
        if spec.per_step_mask:
            feats, step_missing = build_masked_sequence_columns(
                split_frame, spec, medians.get(spec.name, np.zeros((0,), np.float32)), stats
            )
            output[f"{spec.name}_feats"] = feats
            output[f"{spec.name}_missing"] = step_missing
        else:
            output[f"{spec.name}_feats"] = normalise_sequence_column(
                split_frame[f"{spec.name}_feats"].tolist(), *stats
            )
            output[f"{spec.name}_missing"] = pd.Series(missing, dtype="int8")
        output[f"{spec.name}_time"] = times
    return pd.DataFrame(output)


def build_arrow_schema(
    outcome_columns: list[str], specs: tuple[ModalitySpec, ...]
) -> object:
    import pyarrow as pa

    fields = [pa.field(column, pa.int64()) for column in PARQUET_ID_COLUMNS]
    fields += [pa.field(outcome, pa.int64()) for outcome in outcome_columns]
    fields.append(pa.field("tab_feats", pa.list_(pa.float32())))
    fields.append(pa.field("tab_missing", pa.list_(pa.int8())))
    for spec in specs:
        fields.append(pa.field(f"{spec.name}_feats", pa.list_(pa.list_(pa.float32()))))
        if spec.per_step_mask:
            fields.append(
                pa.field(f"{spec.name}_missing", pa.list_(pa.list_(pa.int8())))
            )
        else:
            fields.append(pa.field(f"{spec.name}_missing", pa.int8()))
        fields.append(pa.field(f"{spec.name}_time", pa.list_(pa.float32())))
    return pa.schema(fields)


def write_split_parquet(
    export_frame: pd.DataFrame,
    output_path: Path,
    outcome_columns: list[str],
    specs: tuple[ModalitySpec, ...],
    chunk_size: int = 5_000,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = build_arrow_schema(outcome_columns, specs)
    writer = pq.ParquetWriter(
        output_path, schema, compression="snappy", use_dictionary=True
    )
    try:
        if export_frame.empty:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
            return
        for start in range(0, len(export_frame), chunk_size):
            chunk = export_frame.iloc[start : start + chunk_size]
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
    finally:
        writer.close()


def build_summary(
    bundle_paths: dict[str, Path],
    output_dir: Path,
    specs: tuple[ModalitySpec, ...],
    args: argparse.Namespace,
    producer_summary: dict[str, object],
    assigned: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
    outcome_columns: list[str],
    tab_feats_columns: list[str],
    continuous_columns: list[str],
    categorical_columns: list[str],
    dropped_categorical_columns: list[str],
    n_strata: int | None,
    split_strategy: str,
    output_files: dict[str, Path],
    summary_path: Path,
) -> dict[str, object]:
    return {
        "input_keys_path": str(bundle_paths["keys"]),
        "input_tabular_path": str(bundle_paths["tabular"]),
        **{
            f"input_{spec.name}_path": str(bundle_paths[spec.name]) for spec in specs
        },
        "modalities": ["static", *[spec.name for spec in specs]],
        "window": producer_summary.get("window"),
        "record": producer_summary.get("record"),
        "ts_layout": producer_summary.get("ts_layout"),
        "ts_feature_width": producer_summary.get("ts_feature_width"),
        "ts_columns": producer_summary.get("ts_columns"),
        # The producer measures only the embedding and ts modalities, so its feature_dims covers
        # every entry of ``modalities`` except the ``static`` one it names first. The static width
        # is not decided until this stage one-hot expands the categoricals (or, without --ohe,
        # drops them), so it is added here rather than copied. Keyed ``static`` to match the
        # modalities list above -- train.py reads the same block off the batch as ``tabular``
        "feature_dims": {
            "static": len(tab_feats_columns),
            **(producer_summary.get("feature_dims") or {}),
        },
        "time_reference_by_modality": {
            spec.name: spec.time_reference for spec in specs
        },
        "output_dir": str(output_dir),
        "split_strategy": split_strategy,
        "strata_label_order": outcome_columns,
        "n_strata": n_strata,
        "seed": args.seed,
        "gender_encoding": "1=M",
        "normalised": args.normalise,
        "ohe": args.ohe,
        "n_input_rows": int(len(assigned)),
        "n_selected_rows": int(sum(len(table) for table in split_tables.values())),
        "n_rows_by_split": {
            name: int(len(table)) for name, table in split_tables.items()
        },
        "n_unique_patients_by_split": {
            name: int(table["subject_id"].nunique())
            for name, table in split_tables.items()
        },
        "n_positive_by_split": {
            name: {
                outcome: int(table[outcome].eq(1).sum()) for outcome in outcome_columns
            }
            for name, table in split_tables.items()
        },
        "prevalence_by_split": {
            name: {
                outcome: (
                    round(float(table[outcome].mean()), 6) if len(table) else None
                )
                for outcome in outcome_columns
            }
            for name, table in split_tables.items()
        },
        # A shared partition can still skew coverage of a 27%-available modality. Record it so
        # a per-split modality imbalance is visible rather than inferred
        "modality_availability_by_split": {
            name: {
                spec.name: (
                    round(float(table[f"has_{spec.name}"].mean()), 6)
                    if len(table) and f"has_{spec.name}" in table.columns
                    else None
                )
                for spec in specs
            }
            for name, table in split_tables.items()
        },
        "n_patient_overlap": {
            f"{a}_{b}": int(
                len(
                    set(split_tables[a]["subject_id"]) & set(split_tables[b]["subject_id"])
                )
            )
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
        },
        "tab_feats_columns": tab_feats_columns,
        "continous_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "dropped_categorical_columns": dropped_categorical_columns,
        "output_files": {name: str(path) for name, path in output_files.items()},
        "summary_path": str(summary_path),
    }


def print_summary(summary: dict[str, object]) -> None:
    print(f"Split strategy: {summary['split_strategy']} (seed {summary['seed']})")
    print(f"Rows by split: {summary['n_rows_by_split']}")
    print(f"tab_feats width: {len(summary['tab_feats_columns'])}")
    if summary["dropped_categorical_columns"]:
        print(
            "WARNING: dropping "
            f"{len(summary['dropped_categorical_columns'])} categorical columns from "
            f"tab_feats: {summary['dropped_categorical_columns']}"
        )
    for name, availability in summary["modality_availability_by_split"].items():
        formatted = ", ".join(
            f"{modality} {value:.1%}" if value is not None else f"{modality} n/a"
            for modality, value in availability.items()
        )
        print(f"  {name:<5} {formatted}")
    if any(summary["n_patient_overlap"].values()):
        raise ValueError(f"Patient overlap across splits: {summary['n_patient_overlap']}")


def main() -> None:
    args = parse_args()
    bundle_dir = resolve_bundle_dir(args.bundle, HOSPITALISATION_OUTPUT_DIR)
    producer_summary = read_producer_summary(bundle_dir)
    specs = parse_modality_specs(list(producer_summary.get("modalities", [])))
    bundle_paths = build_bundle_paths(
        bundle_dir,
        specs,
        str(producer_summary.get("embedding_format", EMBEDDING_FORMAT)),
    )

    print(f"[1/6] Reading bundle {bundle_dir.name}.")
    key_table = read_key_table(bundle_paths["keys"])
    tabular = read_tabular_table(bundle_paths["tabular"])
    embeddings = {
        spec.name: read_embedding_array(bundle_paths[spec.name], spec.name)
        for spec in specs
    }
    validate_alignment_lengths(key_table, tabular, embeddings)
    for spec in specs:
        validate_presence_flags(key_table, spec.name, embeddings[spec.name])
    outcome_columns = infer_outcome_columns(key_table)
    print(
        f"    {len(key_table):,} rows, modalities: "
        f"{', '.join(spec.name for spec in specs)}."
    )

    print("[2/6] Assigning the shared split.")
    strat_frame = build_stratification_frame(key_table, tabular)
    if args.split_ids_dir is not None:
        split_labels = read_external_split_labels(
            strat_frame, args.split_ids_dir, args.split_ids_outcome
        )
        split_strategy = (
            f"external_ids:{args.split_ids_outcome}"
            if args.split_ids_outcome
            else "external_ids:shared"
        )
        n_strata = None
    else:
        strata = build_strata(strat_frame)
        split_labels = assign_shared_split(strat_frame, strata, seed=args.seed)
        split_strategy = SPLIT_STRATEGY
        n_strata = int(strata.nunique())
    assigned, split_tables = build_split_tables(key_table, split_labels)
    validate_split_invariants(assigned, split_tables)
    print(
        f"    {split_strategy}: "
        + ", ".join(f"{name} {len(table):,}" for name, table in split_tables.items())
        + (f" over {n_strata} strata." if n_strata is not None else ".")
    )

    print("[3/6] Selecting and classifying tabular columns.")
    tabular = expand_vitals_acuity(tabular)
    selected_columns = get_selected_tabular_columns(tabular)
    continuous, passthrough, categorical = classify_tabular_columns(
        tabular, selected_columns
    )
    tabular_lookup = tabular[["subject_id", *selected_columns]]
    if args.ohe:
        dropped_categorical: list[str] = []
    else:
        dropped_categorical = list(categorical)
    print(
        f"    {len(continuous)} continuous, {len(passthrough)} passthrough binary, "
        f"{len(categorical)} categorical."
    )

    print("[4/6] Fitting statistics on the training split only.")
    train_keys = split_tables["train"]
    train_frame = build_split_frame(
        train_keys, tabular_lookup, outcome_columns, selected_columns, specs
    )
    ohe_columns = (
        fit_ohe_columns(train_frame, categorical) if args.ohe and categorical else []
    )
    export_tabular_columns = [
        column for column in selected_columns if column not in categorical
    ] + ohe_columns
    # Report the indicator columns actually written, including pass-through binaries and dummies
    indicator_column_set = set(passthrough) | set(ohe_columns)
    final_categorical_columns = [
        column for column in export_tabular_columns if column in indicator_column_set
    ]
    train_frame = attach_embedding_columns(train_frame, train_keys, embeddings)
    if ohe_columns:
        train_frame = apply_ohe_to_frame(train_frame, categorical, ohe_columns)
    train_frame = coerce_tabular_numeric_frame(train_frame, export_tabular_columns)
    medians_tabular = fit_tabular_imputation_stats(train_frame, export_tabular_columns)
    train_imputed = apply_tabular_imputation(
        train_frame, export_tabular_columns, medians_tabular
    )
    normalisation_columns = [
        column for column in export_tabular_columns if column in set(continuous)
    ]
    tabular_norm_stats = (
        fit_tabular_normalisation_stats(train_imputed, normalisation_columns)
        if args.normalise
        else {}
    )
    sequence_medians = {
        spec.name: fit_sequence_imputation_stats(
            list(embeddings[spec.name][train_keys["_source_row"].to_numpy(dtype=int)])
        )
        for spec in specs
        if spec.per_step_mask
    }
    embedding_norm_stats = (
        {
            spec.name: fit_embedding_normalisation_stats(
                train_keys, embeddings[spec.name], sequence_min_max=extract_sequence_min_max
            )
            for spec in specs
        }
        if args.normalise
        else {}
    )

    suffix = ""
    if args.normalise:
        suffix += "_normalised"
    if args.ohe:
        suffix += "_ohe"
    output_parent = args.output_dir if args.output_dir is not None else bundle_dir
    output_dir = output_parent / f"{bundle_dir.name}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[5/6] Writing splits to {output_dir}.")
    split_labels_path = output_dir / f"{bundle_dir.name}{suffix}_{SPLIT_LABELS_FILENAME}"
    write_split_labels(assigned, split_labels_path)
    output_files: dict[str, Path] = {"split_labels": split_labels_path}
    for split_name in SPLIT_NAMES:
        split_keys = split_tables[split_name]
        split_frame = build_split_frame(
            split_keys, tabular_lookup, outcome_columns, selected_columns, specs
        )
        split_frame = attach_embedding_columns(split_frame, split_keys, embeddings)
        if ohe_columns:
            split_frame = apply_ohe_to_frame(split_frame, categorical, ohe_columns)
        split_frame = coerce_tabular_numeric_frame(split_frame, export_tabular_columns)
        split_frame = apply_tabular_imputation(
            split_frame, export_tabular_columns, medians_tabular
        )
        if args.normalise:
            split_frame = apply_tabular_normalisation(split_frame, tabular_norm_stats)
        export_frame = build_export_frame(
            split_frame,
            outcome_columns,
            export_tabular_columns,
            specs,
            embedding_norm_stats,
            sequence_medians,
        )
        output_path = output_dir / f"{split_name}_{bundle_dir.name}{suffix}.parquet"
        write_split_parquet(export_frame, output_path, outcome_columns, specs)
        output_files[split_name] = output_path
        print(f"    {split_name}: {len(export_frame):,} rows -> {output_path.name}")

    print("[6/6] Writing the split summary.")
    summary_path = output_dir / f"{bundle_dir.name}{suffix}_split_summary.json"
    summary = build_summary(
        bundle_paths=bundle_paths,
        output_dir=output_dir,
        specs=specs,
        args=args,
        producer_summary=producer_summary,
        assigned=assigned,
        split_tables=split_tables,
        outcome_columns=outcome_columns,
        tab_feats_columns=export_tabular_columns,
        continuous_columns=normalisation_columns,
        categorical_columns=final_categorical_columns,
        dropped_categorical_columns=dropped_categorical,
        n_strata=n_strata,
        split_strategy=split_strategy,
        output_files=output_files,
        summary_path=summary_path,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print_summary(summary)
    print(f"Saved split summary to {summary_path}.")


if __name__ == "__main__":
    main()
