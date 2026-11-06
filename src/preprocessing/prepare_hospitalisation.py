"""Build the hospitalisation bundle from the extracted cohort data.

The output has one row per patient, keyed by the index admission. It contains static features and
up to six non-static modalities::

    static      static.csv                         patient-level
    ts_prev     events_prev.csv                    previous admission  [prev_start, prev_end]
    ts_curr     events_curr.csv                    index ED phase      [curr_start, curr_end)
    notes_prev  notes_bhc                          previous admission
    ecg_curr    ecg_ecgfm                          index ED phase
    cxr_prev    cxr                                previous admission
    cxr_curr    cxr                                index ED phase

The current window is the ED phase of the last admission. The previous window is the second-to-last
ED-linked admission. Both windows are read from ``cohort_keys.csv``.

Usage::

    INPUT=../outputs/hospitalisation/3.1/extract/optionalHistory

    python -m src.preprocessing.prepare_hospitalisation --input-dir $INPUT

``--input-dir`` points to the ``requiredHistory`` or ``optionalHistory`` extract directory, not its
parent. The history setting is read from ``extract_summary.json`` and included in the bundle name.
``--output-dir`` defaults to the release directory beside ``extract/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.prep_utils import (  # noqa: E402
    TS_STAT_NAMES,
    EmbeddingFormat,
    aggregate_cxr_studies,
    get_ecg_feature_columns,
    get_feature_columns,
    identifier_matches,
    join_selected_values,
    normalise_nullable_integer_column,
    parse_row_ids,
    to_sequence_array,
    validate_embedding_array,
    validate_row_ids,
    write_embedding_file,
)

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

HOSPITALISATION_DIR = OUTPUTS_DIR / "hospitalisation" / "3.1"

DEFAULT_INPUT_DIR = HOSPITALISATION_DIR / "extract" / "optionalHistory"
DEFAULT_OUTPUT_DIR = HOSPITALISATION_DIR
DEFAULT_NOTES_DIR = OUTPUTS_DIR / "notes_bhc"
DEFAULT_ECG_KEYS_PATH = OUTPUTS_DIR / "ecg_ecgfm" / "ecgfm_embedding_keys.csv"
DEFAULT_ECG_EMBEDDINGS_PATH = OUTPUTS_DIR / "ecg_ecgfm" / "ecgfm_embeddings.pkl"
DEFAULT_CXR_KEYS_PATH = OUTPUTS_DIR / "cxr" / "cxr_embedding_keys.csv"
DEFAULT_CXR_EMBEDDINGS_PATH = OUTPUTS_DIR / "cxr" / "cxr_embeddings.pkl"

NOTES_KEYS_FILENAME = "notes_embedding_keys.csv"
NOTES_EMBEDDINGS_FILENAME = "notes_embeddings.pkl"

RecordMode: TypeAlias = Literal["first", "last", "average", "all"]
RECORD_MODES = get_args(RecordMode)
CohortSelection: TypeAlias = Literal["all", "complete"]
COHORT_SELECTIONS = get_args(CohortSelection)
TsLayout: TypeAlias = Literal["modality", "tabular"]
TS_LAYOUTS = get_args(TsLayout)
# Modality arrays use the pickle format expected by the split stage
EMBEDDING_FORMAT: EmbeddingFormat = "pickle"
EMBEDDING_EXTENSION = "pkl"

TEST_ROW_LIMIT = 2000

OUTCOME_PREFIX = "hospitalisation_"
HOSPITALISATION_LABELS = ["ext_stay_7", "in_hosp_death", "non_home_discharge"]
OUTCOME_COLUMNS = [f"{OUTCOME_PREFIX}{label}" for label in HOSPITALISATION_LABELS]

# Static feature families with timestamps and labels excluded
STATIC_FEATURE_PREFIXES = (
    "demographics_",
    "biometrics_",
    "encounter_",
    "history_",
    "vitals_",
    "conditions_",
    "medications_",
    "careorders_",
)
STATIC_KEY_COLUMNS = ["subject_id", "last_hadm_id", "ed_stay_id"]

# Outcome columns and source fields that must not enter the bundle
COHORT_KEY_LEAK_COLUMNS = (
    "discharge_location",
    "los_days",
    "dod",
    "deathtime",
    "admission_location",
    "ext_stay_7",
    "in_hosp_death",
    "non_home_discharge",
    "prev_non_home_discharge",
)

TS_WINDOWS = ("prev", "curr")
# Per-window bounds and source columns from cohort_keys.csv
WINDOW_SPECS: dict[str, dict[str, str]] = {
    "prev": {
        "start": "prev_start",
        "end": "prev_end",
        "closed": "both",
        "events_filename": "events_prev.csv",
        "elapsed": "minutes_since_prev_start",
        "hadm": "prev_hadm_id",
    },
    "curr": {
        "start": "curr_start",
        "end": "curr_end",
        "closed": "left",
        "events_filename": "events_curr.csv",
        "elapsed": "minutes_since_curr_start",
        "hadm": "last_hadm_id",
    },
}

COHORT_KEYS_FILENAME = "cohort_keys.csv"
STATIC_FILENAME = "static.csv"
EXTRACT_SUMMARY_FILENAME = "extract_summary.json"

COHORT_KEYS_PARSE_DATES = [
    "edregtime", "edouttime", "admittime", "prev_edregtime",
    "prev_admittime", "prev_dischtime", "curr_start", "curr_end",
    "prev_start", "prev_end",
]


@dataclass(frozen=True)
class ModalitySpec:
    """Describe a non-static modality and its observation window."""

    name: str
    kind: str  # "ts" | "ecg" | "cxr" | "notes"
    window: str  # "prev" | "curr"


MODALITY_SPECS: tuple[ModalitySpec, ...] = (
    ModalitySpec("ts_prev", "ts", "prev"),
    ModalitySpec("ts_curr", "ts", "curr"),
    ModalitySpec("ecg_curr", "ecg", "curr"),
    ModalitySpec("cxr_prev", "cxr", "prev"),
    ModalitySpec("cxr_curr", "cxr", "curr"),
    ModalitySpec("notes_prev", "notes", "prev"),
)
EMBEDDING_FEATURE_PREFIXES = {"ecg": "ecg_ft", "cxr": "cxr_ft", "notes": "notes_ft"}

WindowSelection: TypeAlias = Literal["dual", "prev", "curr"]
WINDOW_SELECTIONS = get_args(WindowSelection)
WINDOW_TOKENS = {"dual": "dualWindow", "prev": "prevWindow", "curr": "currWindow"}


def active_windows(window: WindowSelection) -> tuple[str, ...]:
    """Return the observation windows one bundle covers."""
    if window not in WINDOW_SELECTIONS:
        raise ValueError(f"window must be one of {WINDOW_SELECTIONS}.")
    return TS_WINDOWS if window == "dual" else (window,)


def active_modality_specs(window: WindowSelection) -> tuple[ModalitySpec, ...]:
    """Return the non-static modalities included for a window selection."""
    windows = active_windows(window)
    return tuple(spec for spec in MODALITY_SPECS if spec.window in windows)

# Identity columns for each bundle row
ID_COLUMNS = [
    "benchmark_row_id", "subject_id", "ed_stay_id",
    "ed_hadm_id", "prev_hadm_id", "prev_ed_stay_id",
]
# Window reference times keep offsets near zero
WINDOW_COLUMNS = [
    "general_intime", "prediction_time", "prev_start", "prev_end",
    "curr_start", "curr_end", "days_since_last_discharge", "has_prev_band",
]


def modality_key_columns(spec: ModalitySpec) -> list[str]:
    """Return the key-table columns for one modality."""
    name = spec.name
    if spec.kind == "ts":
        return [f"{name}_time", f"has_{name}", f"n_{name}_records"]
    columns = [
        f"{name}_key_row_ids",
        f"{name}_ids",
        f"{name}_time",
        f"n_{name}_records",
        f"n_{name}_in_window",
        f"has_{name}",
    ]
    if spec.kind == "cxr":
        columns[3:3] = [f"{name}_dicom_ids", f"{name}_paths", f"{name}_view_positions"]
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the seven-modality hospitalisation bundle from the "
            "build_hospitalisation.py outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent directory for the bundle folder (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory holding cohort_keys.csv, static.csv, events_prev.csv, events_curr.csv "
            "and extract_summary.json. One of the extract's requiredHistory/ or optionalHistory/ "
            f"subdirectories, never their parent (default: {DEFAULT_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=DEFAULT_NOTES_DIR,
        help="Directory holding notes_embedding_keys.csv and notes_embeddings.pkl.",
    )
    parser.add_argument(
        "--ecg-keys-path", type=Path, default=DEFAULT_ECG_KEYS_PATH,
        help="ECG-FM embedding key table.",
    )
    parser.add_argument(
        "--ecg-embeddings-path", type=Path, default=DEFAULT_ECG_EMBEDDINGS_PATH,
        help="ECG-FM embedding table.",
    )
    parser.add_argument(
        "--cxr-keys-path", type=Path, default=DEFAULT_CXR_KEYS_PATH,
        help="CXR embedding key table (one row per DICOM).",
    )
    parser.add_argument(
        "--cxr-embeddings-path", type=Path, default=DEFAULT_CXR_EMBEDDINGS_PATH,
        help="CXR embedding table.",
    )
    parser.add_argument(
        "--window",
        choices=WINDOW_SELECTIONS,
        default="dual",
        help=(
            "Observation windows: 'dual' (default), 'prev' or 'curr'. ECG is recorded in the "
            "current window and notes in the previous one."
        ),
    )
    parser.add_argument(
        "--record",
        choices=RECORD_MODES,
        default="average",
        help=(
            "How to represent multiple in-window items. 'all' keeps every time point as a raw "
            "sequence. 'average' reduces the window to nine summary statistics. 'first'/'last' "
            "keep the earliest "
            "or latest single timepoint. For ecg_curr/cxr_prev/cxr_curr: 'all' stacks every "
            "in-window vector, 'average' means them into one, 'first'/'last' select one. Notes "
            "are one per admission, so notes_prev is unaffected."
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=COHORT_SELECTIONS,
        default="all",
        help=(
            "Rows to export: 'all' keeps every extracted patient with modality presence flags. "
            "'complete' keeps only patients with all non-static modalities."
        ),
    )
    parser.add_argument(
        "--ts-layout",
        choices=TS_LAYOUTS,
        default="modality",
        help=(
            "Where the vitals/labs statistics go: 'modality' writes ts_prev/ts_curr pickles, "
            "'tabular' folds them into _tabular.csv instead. 'tabular' is incompatible with "
            "--record all, whose variable-length sequences cannot fold into a CSV column."
        ),
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        action="store_true",
        help=f"Process only the first {TEST_ROW_LIMIT} rows of cohort_keys.csv.",
    )
    args = parser.parse_args()
    if args.record == "all" and args.ts_layout == "tabular":
        parser.error(
            "--record all cannot be combined with --ts-layout tabular: a variable-length "
            "per-timepoint sequence has no fixed-width tabular representation."
        )
    return args


def build_output_paths(
    output_dir: Path,
    record: RecordMode,
    cohort: CohortSelection,
    ts_layout: TsLayout,
    window: WindowSelection = "dual",
    history: str = "required",
    test_mode: bool = False,
) -> dict[str, Path]:
    """Build the bundle directory and every artefact path for one run."""
    if record not in RECORD_MODES:
        raise ValueError(f"record must be one of {RECORD_MODES}.")
    if cohort not in COHORT_SELECTIONS:
        raise ValueError(f"cohort must be one of {COHORT_SELECTIONS}.")
    if ts_layout not in TS_LAYOUTS:
        raise ValueError(f"ts_layout must be one of {TS_LAYOUTS}.")
    specs = active_modality_specs(window)

    extension = EMBEDDING_EXTENSION
    # The summary records cohort selection, so both cohort modes share one bundle directory
    # The history token always identifies the source extract directory
    prefix = (
        f"hospitalisation_{WINDOW_TOKENS[window]}_{record}Record_{history}History"
    )
    if ts_layout == "tabular":
        prefix = f"{prefix}_tabularTs"
    if test_mode:
        prefix = f"{prefix}_test{TEST_ROW_LIMIT}"

    bundle_dir = output_dir / prefix
    paths = {
        "bundle_dir": bundle_dir,
        "keys": bundle_dir / f"{prefix}_keys.csv",
        "tabular": bundle_dir / f"{prefix}_tabular.csv",
        "summary": bundle_dir / f"{prefix}_summary.json",
    }
    for spec in specs:
        if spec.kind == "ts":
            if ts_layout == "modality":
                paths[spec.name] = bundle_dir / f"{prefix}_{spec.name}.{extension}"
        else:
            paths[spec.name] = bundle_dir / f"{prefix}_{spec.name}_embeddings.{extension}"
    return paths


def validate_unique_keys(
    data: pd.DataFrame, key_columns: list[str], table_name: str
) -> None:
    """Raise when a table carries duplicate keys."""
    duplicated = data.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        examples = data.loc[duplicated, key_columns].head(5).to_dict("records")
        raise ValueError(
            f"{table_name} has duplicate keys for {key_columns}. "
            f"Example duplicate keys: {examples}"
        )


def read_extract_summary(input_dir: Path) -> dict[str, object]:
    """Read extract_summary.json, tolerating its absence."""
    path = input_dir / EXTRACT_SUMMARY_FILENAME
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def detect_history_requirement(
    extract_summary: dict[str, object], cohort_keys: pd.DataFrame
) -> str:
    """Return the extract's ``--history`` setting, falling back to the cohort composition.

    ``required`` is the ``has_prev_band`` subset of ``optional``, so a cohort where every row is
    in-band was extracted with ``--history required`` even when the summary is missing or stale.
    The fallback also covers extracts written before the field was named ``history``.
    """
    requirement = extract_summary.get("history")
    if isinstance(requirement, str) and requirement:
        return requirement
    if "has_prev_band" in cohort_keys.columns:
        return "required" if bool(cohort_keys["has_prev_band"].all()) else "optional"
    return "required"


def read_cohort_keys(input_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read cohort_keys.csv, one row per patient, with window bounds parsed."""
    path = input_dir / COHORT_KEYS_FILENAME
    frame = pd.read_csv(path, nrows=nrows, low_memory=False)
    for column in COHORT_KEYS_PARSE_DATES:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    missing = [
        column
        for column in ("subject_id", "last_hadm_id", "ed_stay_id", "curr_start", "curr_end")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}.")
    if frame[["curr_start", "curr_end"]].isna().any().any():
        raise ValueError(f"{path} contains rows with an unparseable current-stay window.")
    # Convert nullable IDs before joining them to integer admission IDs
    for column in ("prev_hadm_id", "prev_ed_stay_id"):
        if column in frame.columns:
            normalise_nullable_integer_column(frame, column)
    validate_unique_keys(frame, ["subject_id"], "Hospitalisation cohort keys")
    return frame


def read_static(input_dir: Path) -> pd.DataFrame:
    """Read static.csv, one row per patient, already family-prefixed by the extractor."""
    path = input_dir / STATIC_FILENAME
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in STATIC_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required key columns: {missing}.")
    missing_outcomes = [column for column in OUTCOME_COLUMNS if column not in frame.columns]
    if missing_outcomes:
        raise ValueError(f"{path} is missing outcome columns: {missing_outcomes}.")
    validate_unique_keys(frame, ["subject_id"], "Hospitalisation static block")
    return frame


def select_static_features(static: pd.DataFrame) -> list[str]:
    """Return the ordered static feature columns, excluding keys, timestamps and labels."""
    features = [
        column
        for column in static.columns
        if column.startswith(STATIC_FEATURE_PREFIXES)
        and column not in STATIC_KEY_COLUMNS
        and column not in OUTCOME_COLUMNS
    ]
    if not features:
        raise ValueError(
            "static.csv contains no feature columns with a recognised family prefix "
            f"({STATIC_FEATURE_PREFIXES})."
        )
    return features


def read_events(input_dir: Path, window: str) -> tuple[pd.DataFrame, list[str]]:
    """Read one window's wide event table and its ordered feature columns.

    The feature columns are read from the header rather than hard-coded, which preserves the
    hyphen in ``labvalues_rdw-sd`` and the parentheses in
    ``labvalues_alanine_aminotransferase_(alt)`` byte-for-byte.
    """
    spec = WINDOW_SPECS[window]
    path = input_dir / spec["events_filename"]
    frame = pd.read_csv(path, low_memory=False)
    elapsed = spec["elapsed"]
    if elapsed not in frame.columns:
        raise ValueError(f"{path} is missing the '{elapsed}' column.")
    columns = frame.columns.tolist()
    features = columns[columns.index(elapsed) + 1 :]
    if not features:
        raise ValueError(f"{path} contains no vitals_/labvalues_ feature columns.")
    if "charttime" in frame.columns:
        frame["charttime"] = pd.to_datetime(frame["charttime"], errors="coerce")
    return frame, features


def assert_feature_names_unambiguous(feature_columns: list[str]) -> None:
    """Reject feature names that would make ``<feature>_<window>_<stat>`` ambiguous.

    The split stage's column classifier parses the trailing statistic token, so a marker whose
    own name ended in one (a hypothetical ``labvalues_..._max``) would be misread.
    """
    ambiguous = [
        feature
        for feature in feature_columns
        if feature.endswith(tuple(f"_{stat}" for stat in TS_STAT_NAMES))
    ]
    if ambiguous:
        raise ValueError(
            "Event feature names must not end in a statistic suffix, which would make the "
            f"'<feature>_<window>_<stat>' column names ambiguous: {ambiguous}."
        )


def assert_windows_agree(prev_features: list[str], curr_features: list[str]) -> None:
    """Both windows must carry identical, identically ordered feature columns."""
    if prev_features != curr_features:
        only_prev = [column for column in prev_features if column not in curr_features]
        only_curr = [column for column in curr_features if column not in prev_features]
        raise ValueError(
            "events_prev.csv and events_curr.csv feature columns differ, so the two windows "
            f"cannot be concatenated. Only in prev: {only_prev[:5]}. "
            f"Only in curr: {only_curr[:5]}. Same set but different order: "
            f"{sorted(prev_features) == sorted(curr_features)}."
        )
    assert_feature_names_unambiguous(prev_features)


def build_ts_columns(
    feature_columns: list[str], windows: tuple[str, ...] = TS_WINDOWS
) -> list[str]:
    """Return the ordered statistic column names for the active windows.

    Window-outer, then feature (events-header order), then statistic, so each window occupies a
    contiguous block and a window ablation is a slice rather than a gather.
    """
    return [
        f"{feature}_{window}_{stat}"
        for window in windows
        for feature in feature_columns
        for stat in TS_STAT_NAMES
    ]


def aggregate_window_stats(
    events: pd.DataFrame,
    feature_columns: list[str],
    subject_ids: pd.Index | np.ndarray | list[int],
    window: str,
) -> pd.DataFrame:
    """Compute nine summary statistics for each patient and feature in one window."""
    elapsed = WINDOW_SPECS[window]["elapsed"]
    ordered_columns = [
        f"{feature}_{window}_{stat}"
        for feature in feature_columns
        for stat in TS_STAT_NAMES
    ]
    subject_index = pd.Index(np.asarray(subject_ids, dtype="int64"), name="subject_id")
    long = events.melt(
        id_vars=["subject_id", elapsed],
        value_vars=feature_columns,
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(index=subject_index, columns=ordered_columns, dtype="float64")

    long = long.sort_values(["feature", "subject_id", elapsed], kind="stable")
    long["tv"] = long[elapsed] * long["value"]
    long["tt"] = long[elapsed] ** 2
    grouped = long.groupby(["subject_id", "feature"], sort=False)
    agg = grouped.agg(
        mean=("value", "mean"),
        median=("value", "median"),
        min=("value", "min"),
        max=("value", "max"),
        first=("value", "first"),
        last=("value", "last"),
        n=("value", "size"),
        sum_t=(elapsed, "sum"),
        sum_v=("value", "sum"),
        sum_tv=("tv", "sum"),
        sum_tt=("tt", "sum"),
    )
    agg["std"] = grouped["value"].std(ddof=0)
    n = agg["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = agg["sum_tv"] / n - (agg["sum_t"] / n) * (agg["sum_v"] / n)
        vart = agg["sum_tt"] / n - (agg["sum_t"] / n) ** 2
        agg["coeff"] = np.where((n >= 2) & (vart > 0), cov / vart, np.nan)
        agg["rate_change"] = np.where(
            (n >= 2) & (agg["first"] != 0),
            (agg["last"] - agg["first"]) / agg["first"],
            np.nan,
        )
    wide = agg[list(TS_STAT_NAMES)].unstack("feature")
    wide.columns = [f"{feature}_{window}_{stat}" for stat, feature in wide.columns]
    return wide.reindex(index=subject_index, columns=ordered_columns)


def build_ts_statistic_vectors(
    stats_by_window: dict[str, pd.DataFrame],
    ts_columns: list[str],
) -> dict[str, tuple[np.ndarray, list[bool], list[int]]]:
    """Convert window statistics into one sequence per patient."""
    result: dict[str, tuple[np.ndarray, list[bool], list[int]]] = {}
    for window, stats in stats_by_window.items():
        window_columns = [column for column in ts_columns if column in stats.columns]
        values = stats[window_columns].to_numpy(dtype=np.float32)
        width = values.shape[1]
        sequences: list[np.ndarray] = []
        has_values: list[bool] = []
        n_records: list[int] = []
        for row in values:
            if np.isnan(row).all():
                sequences.append(np.empty((0, width), dtype=np.float32))
                has_values.append(False)
                n_records.append(0)
                continue
            sequences.append(row.reshape(1, width))
            has_values.append(True)
            n_records.append(1)
        result[f"ts_{window}"] = (to_sequence_array(sequences), has_values, n_records)
    return result


def extract_ts_sequences(
    events: pd.DataFrame,
    feature_columns: list[str],
    subject_ids: list[int],
    window: str,
    record: RecordMode,
) -> tuple[np.ndarray, list[str], list[bool], list[int]]:
    """Build one raw vitals and labs sequence per patient for one window."""
    elapsed = WINDOW_SPECS[window]["elapsed"]
    n_features = len(feature_columns)
    sequences = [
        np.empty((0, n_features), dtype=np.float32) for _ in range(len(subject_ids))
    ]
    ts_time = ["" for _ in range(len(subject_ids))]
    has_ts = [False for _ in range(len(subject_ids))]
    n_ts_records = [0 for _ in range(len(subject_ids))]

    groups = dict(
        list(
            events.sort_values(["subject_id", elapsed], kind="stable").groupby(
                "subject_id", sort=False
            )
        )
    )
    with warnings.catch_warnings():
        # np.nanmean warns for features with no values in this window
        warnings.simplefilter("ignore", RuntimeWarning)
        for output_row, subject_id in enumerate(subject_ids):
            group = groups.get(subject_id)
            if group is None or group.empty:
                continue
            values = group[feature_columns].to_numpy(dtype=np.float32)
            times = group["charttime"].astype(str).tolist()
            if record == "all":
                sequence, row_time = values, "|".join(times)
            elif record == "average":
                sequence = np.nanmean(values, axis=0, keepdims=True).astype(np.float32)
                row_time = ""
            elif record == "first":
                sequence, row_time = values[:1], times[0]
            else:  # "last"
                sequence, row_time = values[-1:], times[-1]
            sequences[output_row] = sequence
            has_ts[output_row] = True
            n_ts_records[output_row] = int(sequence.shape[0])
            ts_time[output_row] = row_time
    return to_sequence_array(sequences), ts_time, has_ts, n_ts_records


def select_window_candidates(
    cohort: pd.DataFrame,
    records: pd.DataFrame,
    record: RecordMode,
    time_column: str,
    start_column: str,
    end_column: str,
    closed: str,
    item_column: str,
    sort_column: str,
    rank_column: str,
    count_column: str,
) -> pd.DataFrame:
    """Rank a modality's records inside each patient's window and keep the selected rows.

    Unlike the deterioration equivalent this takes the window bounds as columns, because the two
    hospitalisation windows have different bounds and different closure.
    """
    candidates = cohort[
        ["benchmark_row_id", "subject_id", start_column, end_column]
    ].merge(records, on="subject_id", how="inner")
    if candidates.empty:
        candidates[rank_column] = pd.Series(dtype="Int64")
        candidates[count_column] = pd.Series(dtype="Int64")
        return candidates
    in_window = candidates[time_column].between(
        candidates[start_column], candidates[end_column], inclusive=closed
    )
    candidates = candidates.loc[in_window].copy()
    if candidates.empty:
        candidates[rank_column] = pd.Series(dtype="Int64")
        candidates[count_column] = pd.Series(dtype="Int64")
        return candidates

    candidates[count_column] = candidates.groupby("benchmark_row_id")[
        item_column
    ].transform("size")
    candidates = candidates.sort_values(
        ["benchmark_row_id", time_column, sort_column],
        ascending=[True, record != "last", record != "last"],
    )
    candidates[rank_column] = candidates.groupby("benchmark_row_id").cumcount() + 1
    if record in ("average", "all"):
        return candidates
    return candidates.loc[candidates[rank_column].eq(1)].copy()


def select_ecg_records(
    cohort: pd.DataFrame, ecg_keys: pd.DataFrame, spec: ModalitySpec, record: RecordMode
) -> pd.DataFrame:
    """Select the ECG records inside one window, collapsed per record mode."""
    name = spec.name
    window = WINDOW_SPECS[spec.window]
    ecg = ecg_keys.loc[
        ecg_keys["subject_id"].isin(set(cohort["subject_id"])),
        ["subject_id", "study_id", "ecg_time"],
    ].copy()
    ecg["ecg_key_row_id"] = ecg.index
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    if ecg["ecg_time"].isna().any():
        raise ValueError("ECG keys contain rows with unparseable ecg_time values.")
    validate_unique_keys(ecg, ["subject_id", "study_id"], "ECG keys")

    candidates = select_window_candidates(
        cohort,
        ecg,
        record,
        time_column="ecg_time",
        start_column=window["start"],
        end_column=window["end"],
        closed=window["closed"],
        item_column="study_id",
        sort_column="ecg_key_row_id",
        rank_column=f"{name}_match_rank",
        count_column=f"n_{name}_in_window",
    )
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                f"{name}_key_row_ids",
                f"{name}_ids",
                f"{name}_time",
                f"n_{name}_records",
                f"n_{name}_in_window",
            ]
        )
    if record in ("average", "all"):
        selected = candidates.groupby("subject_id", as_index=False).agg(
            **{
                f"{name}_key_row_ids": ("ecg_key_row_id", join_selected_values),
                f"{name}_ids": ("study_id", join_selected_values),
                f"{name}_time": ("ecg_time", join_selected_values),
                f"n_{name}_records": ("study_id", "size"),
                f"n_{name}_in_window": (f"n_{name}_in_window", "first"),
            }
        )
    else:
        selected = candidates.rename(
            columns={
                "ecg_key_row_id": f"{name}_key_row_ids",
                "study_id": f"{name}_ids",
                "ecg_time": f"{name}_time",
            }
        )
        selected[f"n_{name}_records"] = 1
    # Average mode stores one vector even when several records were selected
    if record == "average":
        selected[f"n_{name}_records"] = 1
    return selected[
        [
            "subject_id",
            f"{name}_key_row_ids",
            f"{name}_ids",
            f"{name}_time",
            f"n_{name}_records",
            f"n_{name}_in_window",
        ]
    ]


def select_cxr_studies(
    cohort: pd.DataFrame, cxr_keys: pd.DataFrame, spec: ModalitySpec, record: RecordMode
) -> pd.DataFrame:
    """Select the CXR studies inside one window, collapsed per record mode."""
    name = spec.name
    window = WINDOW_SPECS[spec.window]
    studies = aggregate_cxr_studies(
        cxr_keys.loc[cxr_keys["subject_id"].isin(set(cohort["subject_id"]))]
    )
    candidates = select_window_candidates(
        cohort,
        studies,
        record,
        time_column="cxr_time",
        start_column=window["start"],
        end_column=window["end"],
        closed=window["closed"],
        item_column="cxr_study_id",
        sort_column="cxr_first_key_row_id",
        rank_column=f"{name}_match_rank",
        count_column=f"n_{name}_in_window",
    )
    output_columns = [
        "subject_id",
        f"{name}_key_row_ids",
        f"{name}_ids",
        f"{name}_time",
        f"{name}_dicom_ids",
        f"{name}_paths",
        f"{name}_view_positions",
        f"n_{name}_records",
        f"n_{name}_in_window",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)
    if record in ("average", "all"):
        selected = candidates.groupby("subject_id", as_index=False).agg(
            **{
                f"{name}_key_row_ids": ("cxr_key_row_ids", join_selected_values),
                f"{name}_ids": ("cxr_study_id", join_selected_values),
                f"{name}_time": ("cxr_image_times", join_selected_values),
                f"{name}_dicom_ids": ("cxr_dicom_ids", join_selected_values),
                f"{name}_paths": ("cxr_paths", join_selected_values),
                f"{name}_view_positions": ("cxr_view_positions", join_selected_values),
                f"n_{name}_records": ("n_cxr_images", "sum"),
                f"n_{name}_in_window": (f"n_{name}_in_window", "first"),
            }
        )
    else:
        selected = candidates.rename(
            columns={
                "cxr_key_row_ids": f"{name}_key_row_ids",
                "cxr_study_id": f"{name}_ids",
                "cxr_image_times": f"{name}_time",
                "cxr_dicom_ids": f"{name}_dicom_ids",
                "cxr_paths": f"{name}_paths",
                "cxr_view_positions": f"{name}_view_positions",
                "n_cxr_images": f"n_{name}_records",
            }
        )
    if record == "average":
        # Average mode stores one vector
        selected[f"n_{name}_records"] = 1
    return selected[output_columns]


def load_notes_index(notes_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Read the note-grained embedding key table and locate its embedding table."""
    keys_path = notes_dir / NOTES_KEYS_FILENAME
    embeddings_path = notes_dir / NOTES_EMBEDDINGS_FILENAME
    if not keys_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Notes embeddings not found in {notes_dir}. Expected {NOTES_KEYS_FILENAME} and "
            f"{NOTES_EMBEDDINGS_FILENAME}. Run `python -m src.preprocessing.extract_notes` first."
        )
    keys = pd.read_csv(keys_path, low_memory=False)
    for column in ("note_id", "subject_id", "hadm_id"):
        if column not in keys.columns:
            raise ValueError(f"{keys_path} is missing the '{column}' column.")
    keys["notes_key_row_id"] = keys.index
    if "note_time" in keys.columns:
        keys["note_time"] = pd.to_datetime(keys["note_time"], errors="coerce")
    # One discharge note per admission keeps the previous-admission join unambiguous
    validate_unique_keys(keys, ["hadm_id"], "Notes embedding keys")
    return keys, embeddings_path


def select_previous_admission_notes(
    cohort: pd.DataFrame, notes_keys: pd.DataFrame, spec: ModalitySpec
) -> pd.DataFrame:
    """Join notes to the PREVIOUS admission by hadm_id.

    Deliberately keyed on ``prev_hadm_id`` and never ``ed_hadm_id``: the index admission's
    discharge summary states discharge location and death, so it would leak all three labels.
    """
    name = spec.name
    output_columns = [
        "subject_id",
        f"{name}_key_row_ids",
        f"{name}_ids",
        f"{name}_time",
        f"n_{name}_records",
        f"n_{name}_in_window",
    ]
    linkable = cohort.loc[cohort["prev_hadm_id"].notna(), ["subject_id", "prev_hadm_id"]]
    if linkable.empty:
        return pd.DataFrame(columns=output_columns)
    matched = linkable.merge(
        notes_keys[["notes_key_row_id", "note_id", "hadm_id", "note_time"]],
        left_on="prev_hadm_id",
        right_on="hadm_id",
        how="inner",
    )
    if matched.empty:
        return pd.DataFrame(columns=output_columns)
    matched = matched.sort_values(["subject_id", "notes_key_row_id"], kind="stable")
    selected = pd.DataFrame(
        {
            "subject_id": matched["subject_id"].to_numpy(),
            f"{name}_key_row_ids": matched["notes_key_row_id"].astype(str).to_numpy(),
            f"{name}_ids": matched["note_id"].astype(str).to_numpy(),
            f"{name}_time": matched["note_time"].astype(str).to_numpy(),
            f"n_{name}_records": 1,
            f"n_{name}_in_window": 1,
        }
    )
    return selected[output_columns]


def build_key_table(
    cohort: pd.DataFrame,
    static: pd.DataFrame,
    selections: dict[str, pd.DataFrame],
    record: RecordMode,
    specs: tuple[ModalitySpec, ...] = MODALITY_SPECS,
) -> pd.DataFrame:
    """Assemble the bundle key table: identity, windows, per-modality selection and labels."""
    keys = pd.DataFrame(
        {
            "benchmark_row_id": cohort["ed_stay_id"].to_numpy(),
            "subject_id": cohort["subject_id"].to_numpy(),
            "ed_stay_id": cohort["ed_stay_id"].to_numpy(),
            "ed_hadm_id": cohort["last_hadm_id"].to_numpy(),
            "prev_hadm_id": cohort["prev_hadm_id"].to_numpy(),
            "prev_ed_stay_id": cohort["prev_ed_stay_id"].to_numpy(),
        }
    )
    # split_hospitalisation.py creates the stratified folds
    keys["general_intime"] = cohort["prev_start"].to_numpy()
    keys["prediction_time"] = cohort["curr_start"].to_numpy()
    for column in ("prev_start", "prev_end", "curr_start", "curr_end"):
        keys[column] = cohort[column].to_numpy()
    keys["days_since_last_discharge"] = cohort.get(
        "days_since_last_discharge", pd.Series(np.nan, index=cohort.index)
    ).to_numpy()
    keys["has_prev_band"] = cohort.get(
        "has_prev_band", pd.Series(True, index=cohort.index)
    ).to_numpy()
    keys["record"] = record

    for spec in specs:
        if spec.kind == "ts":
            continue
        name = spec.name
        selected = selections.get(name)
        columns = modality_key_columns(spec)
        value_columns = [column for column in columns if column != f"has_{name}"]
        if selected is None or selected.empty:
            for column in value_columns:
                keys[column] = pd.NA
        else:
            keys = keys.merge(selected, on="subject_id", how="left", validate="one_to_one")
        keys[f"n_{name}_records"] = (
            pd.to_numeric(keys[f"n_{name}_records"], errors="coerce").fillna(0).astype("int64")
        )
        keys[f"n_{name}_in_window"] = (
            pd.to_numeric(keys[f"n_{name}_in_window"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        keys[f"has_{name}"] = keys[f"{name}_key_row_ids"].notna()

    outcomes = static.set_index("subject_id")[OUTCOME_COLUMNS]
    missing = keys.loc[~keys["subject_id"].isin(outcomes.index), "subject_id"].tolist()
    if missing:
        raise ValueError(
            "Cohort references patients missing from static.csv. "
            f"Example subject_id values: {missing[:5]}."
        )
    keys = keys.join(outcomes, on="subject_id")
    if keys[OUTCOME_COLUMNS].isna().any().any():
        raise ValueError("static.csv carries null outcome labels for the exported cohort.")
    for column in OUTCOME_COLUMNS:
        keys[column] = keys[column].astype("int64")

    leaked = [column for column in COHORT_KEY_LEAK_COLUMNS if column in keys.columns]
    if leaked:
        raise ValueError(f"Key table carries outcome-equivalent columns: {leaked}.")
    validate_unique_keys(keys, ["subject_id"], "Hospitalisation key table")
    validate_unique_keys(keys, ["ed_stay_id"], "Hospitalisation key table")
    return keys


def select_cohort_subjects(
    keys: pd.DataFrame,
    cohort: CohortSelection,
    specs: tuple[ModalitySpec, ...] = MODALITY_SPECS,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the exported row mask and the per-modality reconciliation counts."""
    if cohort not in COHORT_SELECTIONS:
        raise ValueError(f"cohort must be one of {COHORT_SELECTIONS}.")
    present = {
        spec.name: keys[f"has_{spec.name}"].to_numpy(dtype=bool) for spec in specs
    }
    reconciliation = {
        "n_extracted_rows": int(len(keys)),
        **{f"n_with_{name}": int(flags.sum()) for name, flags in present.items()},
    }
    complete = np.ones(len(keys), dtype=bool)
    for flags in present.values():
        complete &= flags
    reconciliation["n_complete_case_rows"] = int(complete.sum())
    keep = complete if cohort == "complete" else np.ones(len(keys), dtype=bool)
    reconciliation["n_exported_rows"] = int(keep.sum())
    return keep, reconciliation


def load_embedding_subset(
    embeddings_path: Path, row_ids: list[int], feature_prefix: str, modality: str
) -> tuple[pd.DataFrame, dict[int, int], list[str]]:
    """Load only the rows an embedding modality needs, then release the source table.

    The CXR table is 3.2 GB and the ECG/notes tables ~850 MB each. Subsetting to the selected rows
    and dropping the source before the next modality loads keeps peak memory bounded and turns the
    per-patient gather into an index into a small frame.
    """
    source = pd.read_pickle(embeddings_path)
    if not isinstance(source, pd.DataFrame):
        raise ValueError(f"{embeddings_path} is not a DataFrame of embeddings.")
    if modality == "ecg":
        feature_columns = get_ecg_feature_columns(source)
    else:
        feature_columns = get_feature_columns(source, feature_prefix)
    validate_row_ids(row_ids, len(source), f"{modality} key row ids")
    identity_columns = [
        column for column in ("subject_id", "study_id", "hadm_id", "note_id")
        if column in source.columns
    ]
    unique_ids = sorted(set(row_ids))
    subset = source.iloc[unique_ids][identity_columns + feature_columns].copy()
    del source
    remap = {row_id: position for position, row_id in enumerate(unique_ids)}
    return subset, remap, feature_columns


def extract_embedding_modality(
    keys: pd.DataFrame,
    spec: ModalitySpec,
    embeddings_path: Path,
    record: RecordMode,
) -> tuple[np.ndarray, int]:
    """Build one row-aligned embedding sequence array for a modality."""
    name = spec.name
    row_id_column = f"{name}_key_row_ids"
    all_row_ids: list[int] = []
    per_row_ids: list[list[int]] = []
    for output_row, value in enumerate(keys[row_id_column].tolist()):
        if pd.isna(value):
            per_row_ids.append([])
            continue
        row_ids = parse_row_ids(value, row_id_column, output_row)
        per_row_ids.append(row_ids)
        all_row_ids.extend(row_ids)

    subset, remap, feature_columns = load_embedding_subset(
        embeddings_path, all_row_ids, EMBEDDING_FEATURE_PREFIXES[spec.kind], spec.kind
    )
    feature_dim = len(feature_columns)
    features = subset[feature_columns].to_numpy(dtype=np.float32)
    subject_ids = (
        subset["subject_id"].to_numpy() if "subject_id" in subset.columns else None
    )

    sequences: list[np.ndarray] = []
    for output_row, row_ids in enumerate(per_row_ids):
        if not row_ids:
            sequences.append(np.empty((0, feature_dim), dtype=np.float32))
            continue
        positions = [remap[row_id] for row_id in row_ids]
        if subject_ids is not None:
            expected = keys.iloc[output_row]["subject_id"]
            if not all(
                identifier_matches(subject_ids[position], expected)
                for position in positions
            ):
                raise ValueError(
                    f"{name} subject_id does not match the key table at output row "
                    f"{output_row}."
                )
        selected = features[positions]
        if record == "average":
            selected = selected.mean(axis=0, keepdims=True)
        sequences.append(selected)
    return to_sequence_array(sequences), feature_dim


def build_tabular_modality(
    keys: pd.DataFrame,
    static: pd.DataFrame,
    static_columns: list[str],
    ts_stats: pd.DataFrame | None,
    output_path: Path,
) -> list[str]:
    """Write ``_tabular.csv``: one row per key row, in keys.csv order."""
    id_columns = ["benchmark_row_id", "subject_id", "ed_stay_id", "ed_hadm_id"]
    static_selected = static.set_index("subject_id")[[*OUTCOME_COLUMNS, *static_columns]]
    order = keys[id_columns].reset_index(drop=True)
    tabular = order.join(static_selected, on="subject_id")
    if ts_stats is not None:
        tabular = tabular.join(ts_stats, on="subject_id")
    if len(tabular) != len(keys) or not tabular["subject_id"].equals(order["subject_id"]):
        raise ValueError("Tabular data are not ordered with the key table.")

    normalise_nullable_integer_column(tabular, "ed_stay_id")
    normalise_nullable_integer_column(tabular, "ed_hadm_id")
    bool_columns = tabular.select_dtypes(include="bool").columns
    if len(bool_columns):
        tabular[bool_columns] = tabular[bool_columns].astype("int8")
    tabular.to_csv(output_path, index=False)
    return [
        column for column in tabular.columns if column not in id_columns + OUTCOME_COLUMNS
    ]


def validate_bundle_alignment(
    keys: pd.DataFrame, modalities: dict[str, np.ndarray]
) -> None:
    """Check every artefact stays row-aligned with keys.csv and agrees with its presence flag."""
    validate_unique_keys(keys, ["subject_id"], "Exported key table")
    validate_unique_keys(keys, ["ed_stay_id"], "Exported key table")
    for name, embeddings in modalities.items():
        if len(embeddings) != len(keys):
            raise ValueError(f"{name} does not have one row per exported key row.")
        validate_embedding_array(name, embeddings)
        flags = keys[f"has_{name}"].to_numpy(dtype=bool)
        non_empty = np.array([sequence.shape[0] > 0 for sequence in embeddings])
        if not np.array_equal(flags, non_empty):
            mismatched = int((flags != non_empty).sum())
            raise ValueError(
                f"has_{name} disagrees with {name} sequence emptiness for {mismatched} rows. "
                "The loader derives availability from sequence length, so the two must agree."
            )
        widths = {sequence.shape[1] for sequence in embeddings}
        if len(widths) != 1:
            raise ValueError(f"{name} sequences have inconsistent feature widths: {widths}.")


def build_summary(
    args: argparse.Namespace,
    history: str,
    extract_summary: dict[str, object],
    keys: pd.DataFrame,
    output_paths: dict[str, Path],
    reconciliation: dict[str, int],
    feature_columns: list[str],
    ts_columns: list[str],
    static_columns: list[str],
    tabular_columns: list[str],
    feature_dims: dict[str, int],
    source_row_limit: int | None,
    specs: tuple[ModalitySpec, ...] = MODALITY_SPECS,
) -> dict[str, object]:
    """Build the bundle summary consumed by split_hospitalisation.py and the docs."""
    n_rows = len(keys)
    windows = active_windows(args.window)
    return {
        "outcome_prefix": OUTCOME_PREFIX,
        "history": history,
        "window": args.window,
        "windows": list(windows),
        "modalities": ["static", *[spec.name for spec in specs]],
        "window_definition": {
            window: [WINDOW_SPECS[window]["start"], WINDOW_SPECS[window]["end"]]
            for window in windows
        },
        "time_reference_by_modality": {
            spec.name: (
                "general_intime" if spec.window == "prev" else "prediction_time"
            )
            for spec in specs
        },
        "record": args.record,
        "cohort": args.cohort,
        "ts_layout": args.ts_layout,
        "embedding_format": EMBEDDING_FORMAT,
        "embedding_representation": "sequence",
        "test_mode": args.test_mode,
        "test_row_limit": source_row_limit,
        # coeff and rate_change are NaN for single-reading (patient, feature) cells rather than
        # the other bundle writes. See the module docstring
        "n1_change_statistics": "nan",
        "gender_encoding": "1=M",
        "cohort_reconciliation": reconciliation,
        "n_exported_rows": int(n_rows),
        "n_unique_patients": int(keys["subject_id"].nunique()),
        **{
            f"n_with_{spec.name}": int(keys[f"has_{spec.name}"].sum())
            for spec in specs
        },
        "n_positive_by_outcome": {
            outcome: int(keys[outcome].eq(1).sum()) for outcome in OUTCOME_COLUMNS
        },
        "prevalence_by_outcome": {
            outcome: (round(float(keys[outcome].mean()), 6) if n_rows else None)
            for outcome in OUTCOME_COLUMNS
        },
        "ts_windows": list(TS_WINDOWS),
        "ts_n_features": len(feature_columns),
        "ts_feature_columns": list(feature_columns),
        "ts_statistics": list(TS_STAT_NAMES),
        "ts_columns": list(ts_columns),
        "ts_feature_width": len(ts_columns),
        "feature_dims": feature_dims,
        "n_static_columns": len(static_columns),
        "static_columns": list(static_columns),
        "n_tabular_columns": len(tabular_columns),
        "input_extract_dir": str(args.input_dir),
        "input_notes_dir": str(args.notes_dir),
        "input_ecg_keys_path": str(args.ecg_keys_path),
        "input_ecg_embeddings_path": str(args.ecg_embeddings_path),
        "input_cxr_keys_path": str(args.cxr_keys_path),
        "input_cxr_embeddings_path": str(args.cxr_embeddings_path),
        "input_extract_summary": {
            key: extract_summary.get(key)
            for key in ("history", "stage_counts", "label_columns", "lookback_days",
                        "current_window")
        },
        "bundle_dir": str(output_paths["bundle_dir"]),
        "key_path": str(output_paths["keys"]),
        "tabular_path": str(output_paths["tabular"]),
        **{
            f"{spec.name}_path": (
                str(output_paths[spec.name]) if spec.name in output_paths else None
            )
            for spec in specs
        },
        "summary_path": str(output_paths["summary"]),
    }


def write_summary_and_report(
    summary: dict[str, object],
    output_paths: dict[str, Path],
    keys: pd.DataFrame,
    specs: tuple[ModalitySpec, ...] = MODALITY_SPECS,
) -> None:
    """Write the summary JSON and print the artefact and cohort counts."""
    with output_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved key table to {output_paths['keys']}.")
    print(f"Saved tabular data to {output_paths['tabular']}.")
    for spec in specs:
        if spec.name in output_paths:
            print(f"Saved {spec.name} to {output_paths[spec.name]}.")
    print(f"Saved summary counts to {output_paths['summary']}.")
    print(f"Exported rows: {summary['n_exported_rows']:,}")
    for spec in specs:
        count = summary[f"n_with_{spec.name}"]
        share = count / len(keys) if len(keys) else 0.0
        print(f"  {spec.name:<11} {count:>7,}  ({share:.1%})")
    for outcome in OUTCOME_COLUMNS:
        n_positive = summary["n_positive_by_outcome"][outcome]
        rate = summary["prevalence_by_outcome"][outcome]
        print(f"Outcome {outcome}: {n_positive:,} positive ({rate*100:.2f}%)")


def main() -> None:
    args = parse_args()
    source_row_limit = TEST_ROW_LIMIT if args.test_mode else None
    if args.test_mode:
        print(f"Test mode: using the first {TEST_ROW_LIMIT} rows of cohort_keys.csv.")

    specs = active_modality_specs(args.window)
    windows = active_windows(args.window)
    ts_specs = tuple(spec for spec in specs if spec.kind == "ts")
    embedding_specs = tuple(spec for spec in specs if spec.kind != "ts")

    print("[1/9] Reading the extracted cohort and static block.")
    extract_summary = read_extract_summary(args.input_dir)
    cohort = read_cohort_keys(args.input_dir, nrows=source_row_limit)
    static = read_static(args.input_dir)
    static = static.loc[static["subject_id"].isin(set(cohort["subject_id"]))]
    static_columns = select_static_features(static)
    history = detect_history_requirement(extract_summary, cohort)
    cohort = cohort.sort_values("subject_id", kind="stable").reset_index(drop=True)
    cohort["benchmark_row_id"] = cohort["ed_stay_id"]

    output_paths = build_output_paths(
        args.output_dir,
        args.record,
        args.cohort,
        args.ts_layout,
        window=args.window,
        history=history,
        test_mode=args.test_mode,
    )
    print(f"    History {history}, {len(cohort):,} extracted patients.")
    print(
        f"    Window '{args.window}' -> modalities: static, "
        f"{', '.join(spec.name for spec in specs)}."
    )

    print(f"[2/9] Reading the {'/'.join(windows)} event table(s).")
    events: dict[str, pd.DataFrame] = {}
    features_by_window: dict[str, list[str]] = {}
    for window in windows:
        frame, features = read_events(args.input_dir, window)
        frame = frame.loc[frame["subject_id"].isin(set(cohort["subject_id"]))].copy()
        events[window] = frame
        features_by_window[window] = features
    if len(windows) == 2:
        assert_windows_agree(features_by_window["prev"], features_by_window["curr"])
    else:
        assert_feature_names_unambiguous(features_by_window[windows[0]])
    feature_columns = features_by_window[windows[0]]
    ts_columns = build_ts_columns(feature_columns, windows)
    print(
        f"    {len(feature_columns)} markers x {len(TS_STAT_NAMES)} statistics x "
        f"{len(windows)} window(s) = {len(ts_columns)} statistic columns."
    )

    print("[3/9] Linking the embedding modalities.")
    selections: dict[str, pd.DataFrame] = {}
    notes_embeddings_path: Path | None = None
    needs_ecg = any(spec.kind == "ecg" for spec in embedding_specs)
    needs_cxr = any(spec.kind == "cxr" for spec in embedding_specs)
    needs_notes = any(spec.kind == "notes" for spec in embedding_specs)
    ecg_keys = pd.read_csv(args.ecg_keys_path, low_memory=False) if needs_ecg else None
    cxr_keys = pd.read_csv(args.cxr_keys_path, low_memory=False) if needs_cxr else None
    notes_keys = None
    if needs_notes:
        notes_keys, notes_embeddings_path = load_notes_index(args.notes_dir)
    for spec in embedding_specs:
        if spec.kind == "ecg":
            selections[spec.name] = select_ecg_records(cohort, ecg_keys, spec, args.record)
        elif spec.kind == "cxr":
            selections[spec.name] = select_cxr_studies(cohort, cxr_keys, spec, args.record)
        else:
            selections[spec.name] = select_previous_admission_notes(
                cohort, notes_keys, spec
            )
    del notes_keys, ecg_keys, cxr_keys

    print("[4/9] Building the key table.")
    keys = build_key_table(cohort, static, selections, args.record, specs=specs)

    print("[5/9] Aggregating the windowed time series.")
    subject_ids = keys["subject_id"].astype("int64").to_numpy()
    stats_by_window = {
        window: aggregate_window_stats(
            events[window], feature_columns, subject_ids, window
        )
        for window in windows
    }
    ts_arrays: dict[str, np.ndarray] = {}
    if args.ts_layout == "tabular":
        # Store statistics in _tabular.csv and keep presence flags in the key table
        vectors = build_ts_statistic_vectors(stats_by_window, ts_columns)
        for spec in ts_specs:
            _, has_ts, n_records = vectors[spec.name]
            keys[f"{spec.name}_time"] = ""
            keys[f"has_{spec.name}"] = has_ts
            keys[f"n_{spec.name}_records"] = n_records
    elif args.record == "average":
        # Store the nine window statistics as one fixed-width vector
        vectors = build_ts_statistic_vectors(stats_by_window, ts_columns)
        for spec in ts_specs:
            sequences, has_ts, n_records = vectors[spec.name]
            ts_arrays[spec.name] = sequences
            keys[f"{spec.name}_time"] = ""
            keys[f"has_{spec.name}"] = has_ts
            keys[f"n_{spec.name}_records"] = n_records
    else:
        # all keeps every time point. first and last keep one
        for spec in ts_specs:
            sequences, times, has_ts, n_records = extract_ts_sequences(
                events[spec.window],
                feature_columns,
                subject_ids.tolist(),
                spec.window,
                args.record,
            )
            ts_arrays[spec.name] = sequences
            keys[f"{spec.name}_time"] = times
            keys[f"has_{spec.name}"] = has_ts
            keys[f"n_{spec.name}_records"] = n_records
    del events

    keep, reconciliation = select_cohort_subjects(keys, args.cohort, specs=specs)
    if not keep.all():
        print(f"    Cohort '{args.cohort}' keeps {int(keep.sum()):,} of {len(keys):,} rows.")
        keys = keys.loc[keep].reset_index(drop=True)
        ts_arrays = {name: array[keep] for name, array in ts_arrays.items()}
        subject_ids = keys["subject_id"].astype("int64").to_numpy()
        stats_by_window = {
            window: stats.reindex(pd.Index(subject_ids, name="subject_id"))
            for window, stats in stats_by_window.items()
        }

    key_columns = [
        *ID_COLUMNS,
        *WINDOW_COLUMNS,
        "record",
        *[column for spec in specs for column in modality_key_columns(spec)],
        *OUTCOME_COLUMNS,
    ]
    keys = keys[key_columns]

    print(f"[6/9] Writing keys ({len(keys):,} rows).")
    output_paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    keys.to_csv(output_paths["keys"], index=False)

    print("[7/9] Building the tabular modality.")
    ts_stats = None
    if args.ts_layout == "tabular":
        ts_stats = pd.concat(
            [stats_by_window[window] for window in windows], axis=1
        )[ts_columns]
    tabular_columns = build_tabular_modality(
        keys, static, static_columns, ts_stats, output_paths["tabular"]
    )
    del stats_by_window, ts_stats

    print("[8/9] Extracting modality embeddings.")
    feature_dims: dict[str, int] = {}
    embedding_arrays: dict[str, np.ndarray] = {}
    for spec in embedding_specs:
        source_path = {
            "ecg": args.ecg_embeddings_path,
            "cxr": args.cxr_embeddings_path,
            "notes": notes_embeddings_path,
        }[spec.kind]
        sequences, feature_dim = extract_embedding_modality(
            keys, spec, source_path, args.record
        )
        embedding_arrays[spec.name] = sequences
        feature_dims[spec.name] = feature_dim
        print(f"    {spec.name}: {feature_dim} features, {int(keys[f'has_{spec.name}'].sum()):,} rows")
    for name, array in ts_arrays.items():
        feature_dims[name] = int(array[0].shape[1]) if len(array) else 0

    print("[9/9] Validating alignment and writing artefacts.")
    validate_bundle_alignment(keys, {**embedding_arrays, **ts_arrays})
    for name, array in {**embedding_arrays, **ts_arrays}.items():
        write_embedding_file(
            array, output_paths[name], f"{name}_sequence", EMBEDDING_FORMAT
        )

    summary = build_summary(
        args=args,
        history=history,
        extract_summary=extract_summary,
        keys=keys,
        output_paths=output_paths,
        reconciliation=reconciliation,
        feature_columns=feature_columns,
        ts_columns=ts_columns,
        static_columns=static_columns,
        tabular_columns=tabular_columns,
        feature_dims=feature_dims,
        source_row_limit=source_row_limit,
        specs=specs,
    )
    write_summary_and_report(summary, output_paths, keys, specs=specs)


if __name__ == "__main__":
    main()
