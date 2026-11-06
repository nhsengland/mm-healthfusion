"""Build the deterioration data bundle.

Example::

    python -m src.preprocessing.prepare_deterioration --test-mode
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.prep_utils import (
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
    validate_unique_keys,
    write_embedding_file,
)

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
# Input produced by ``build_deterioration.py --mode ts``. The base file has one row per
# ED stay. The time-series file has one row per recorded time point
DEFAULT_BASE_PATH = OUTPUTS_DIR / "mds_ed" / "mds_ed_base.csv"
DEFAULT_TS_PATH = OUTPUTS_DIR / "mds_ed" / "mds_ed_ts.csv"
DEFAULT_ECG_KEYS_PATH = OUTPUTS_DIR / "ecg_ecgfm" / "ecgfm_embedding_keys.csv"
DEFAULT_CXR_KEYS_PATH = OUTPUTS_DIR / "cxr" / "cxr_embedding_keys.csv"
DEFAULT_ECG_EMBEDDINGS_PATH = OUTPUTS_DIR / "ecg_ecgfm" / "ecgfm_embeddings.pkl"
DEFAULT_CXR_EMBEDDINGS_PATH = OUTPUTS_DIR / "cxr" / "cxr_embeddings.pkl"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "deterioration" / "3.1_ohe"
RecordMode: TypeAlias = Literal["first", "last", "average", "all"]
RECORD_MODES = get_args(RecordMode)
PatientStaySelection: TypeAlias = Literal["first", "last", "random", "all"]
PATIENT_STAY_SELECTIONS = get_args(PatientStaySelection)
CohortSelection: TypeAlias = Literal["complete", "all"]
COHORT_SELECTIONS = get_args(CohortSelection)
EMBEDDING_FORMATS = get_args(EmbeddingFormat)
RANDOM_SEED = 42
TEST_ROW_LIMIT = 1050

OUTCOME_COLUMNS = ["deterioration_severe_hypoxemia", "deterioration_ecmo", "deterioration_vasopressors", "deterioration_inotropes", "deterioration_mechanical_ventilation", "deterioration_cardiac_arrest", "deterioration_icu_24h", "deterioration_icu_stay", "deterioration_mortality_1d"]
# Identifier and metadata columns retained from the base file
BASE_RENAMES = {
    "general_subject_id": "subject_id",
    "general_ed_stay_id": "ed_stay_id",
    "general_ed_hadm_id": "ed_hadm_id",
    "general_strat_fold": "strat_fold",
}
BASE_PARSE_DATES = ["general_intime", "general_90min"]
# Static feature groups copied to ``_tabular.csv`` from the base file. Time-series summary
# columns are added separately by ``aggregate_ts_stats``
BASE_TABULAR_PREFIXES = ("demographics_", "biometrics_", "vitals_", "diagnoses_")
# ``labvalues_base_excess`` uses -413 for missing values in the source file
TS_BASE_EXCESS_PREFIX = "labvalues_base_excess"
TS_BASE_EXCESS_SENTINEL = -413
OUTPUT_COLUMNS = [
    # Keep this alias for the downstream split step
    "benchmark_row_id", "subject_id", "ed_stay_id", "ed_hadm_id", "strat_fold",
    "general_intime", "general_90min", "ecgfm_key_row_id", "ecg_id", "ecg_time",
    "ecg_path", "has_ecg_embedding", "ecg_match_rank", "n_ecg_records_in_90min",
    "cxr_key_row_ids", "cxr_study_id", "cxr_time", "cxr_dicom_ids", "cxr_paths",
    "cxr_view_positions", "n_cxr_images", "has_cxr_embedding", "ts_time", "has_ts",
    "n_ts_records", "record", "cxr_match_rank", "n_cxr_studies_in_90min",
    "deterioration_severe_hypoxemia", "deterioration_ecmo", "deterioration_vasopressors",
    "deterioration_inotropes", "deterioration_mechanical_ventilation",
    "deterioration_cardiac_arrest", "deterioration_icu_24h", "deterioration_icu_stay",
    "deterioration_mortality_1d",
]
# Add these time-series columns in ``main`` after processing stays in cohort order
KEY_TABLE_TS_COLUMNS = ["ts_time", "has_ts", "n_ts_records"]
KEY_TABLE_COLUMNS = [column for column in OUTPUT_COLUMNS if column not in KEY_TABLE_TS_COLUMNS]
# ``--record all`` stores the raw sequence and its time and count columns. Other modes store
# time-series summaries in ``_tabular.csv`` and only keep ``has_ts`` in the key table
SUMMARY_KEY_COLUMNS = [column for column in OUTPUT_COLUMNS if column not in ("ts_time", "n_ts_records")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a deterioration key file linking outcome labels to ECG-FM "
            "and CXR embedding keys."
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for all deterioration key, embedding and summary outputs.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Stay-level static source (mds_ed_base.csv), one row per ED stay.",
    )
    parser.add_argument(
        "--ts-path",
        type=Path,
        default=DEFAULT_TS_PATH,
        help="Long vitals/labs time-series source (mds_ed_ts.csv), many rows per stay.",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        action="store_true",
        default=False,
        help=f"Process only the first {TEST_ROW_LIMIT} rows from mds_ed_base.csv.",
    )
    parser.add_argument(
        "--record",
        choices=RECORD_MODES,
        default="average",
        help=(
            "How to represent ECG records and CXR studies within the first 90 minutes: "
            "first (earliest), last (latest), average (all averaged into one "
            "vector array, default), or all (all vectors stacked in one "
            "variable-length array per stay, for either cohort mode)."
        ),
    )
    parser.add_argument(
        "--patient-stay",
        choices=PATIENT_STAY_SELECTIONS,
        default="all",
        help=(
            "ED stays to export per patient: first/last/random keeps one stay, "
            f"all retains all stays. Random uses seed {RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--cohort",
        choices=COHORT_SELECTIONS,
        default="complete",
        help=(
            "Rows to export: complete cases with ECG+CXR embeddings, or all linked "
            "rows with missing CXR/TS data retained."
        ),
    )
    parser.add_argument(
        "--embedding-format",
        choices=EMBEDDING_FORMATS,
        default="pickle",
        help="Format for ECG, CXR and time-series outputs: pickle or parquet.",
    )
    return parser.parse_args()


def build_output_paths(
    output_dir: Path,
    patient_stay: PatientStaySelection,
    record: RecordMode,
    cohort: CohortSelection,
    embedding_format: EmbeddingFormat = "pickle",
    test_mode: bool = False,
) -> dict[str, Path]:
    """Build the output bundle directory and artefact paths for one run."""
    embedding_extension = "parquet" if embedding_format == "parquet" else "pkl"
    prefix = f"deterioration_{patient_stay}Stay_{record}Record_{cohort}Cohort"
    if test_mode:
        prefix = f"{prefix}_test{TEST_ROW_LIMIT}"
    bundle_dir = output_dir / prefix
    return {
        "bundle_dir": bundle_dir,
        "keys": bundle_dir / f"{prefix}_keys.csv",
        "tabular": bundle_dir / f"{prefix}_tabular.csv",
        "ecg_embeddings": bundle_dir / f"{prefix}_ecg_embeddings.{embedding_extension}",
        "cxr_embeddings": bundle_dir / f"{prefix}_cxr_embeddings.{embedding_extension}",
        "ts_embeddings": bundle_dir / f"{prefix}_ts.{embedding_extension}",
        "summary": bundle_dir / f"{prefix}_summary.json",
    }


def read_base(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the base file, rename identifiers and clear missing outcome markers."""
    frame = pd.read_csv(path, parse_dates=BASE_PARSE_DATES, nrows=nrows, low_memory=False)
    frame = frame.rename(columns=BASE_RENAMES)
    replace_missing_outcomes(frame)
    normalise_nullable_integer_column(frame, "ed_stay_id")
    normalise_nullable_integer_column(frame, "ed_hadm_id")
    validate_unique_keys(frame, ["ed_stay_id"], "Deterioration base data")
    return frame


def get_ts_feature_columns(ts_columns: list[str]) -> list[str]:
    """Return time-series feature columns in the order used by the source file."""
    columns = list(ts_columns)
    if "minutes_since_intime" not in columns:
        raise ValueError("mds_ed_ts.csv is missing the 'minutes_since_intime' column.")
    features = columns[columns.index("minutes_since_intime") + 1 :]
    if not features:
        raise ValueError("mds_ed_ts.csv contains no vitals_/labvalues_ feature columns.")
    return features


def replace_missing_outcomes(data: pd.DataFrame) -> None:
    """Replace missing outcome markers with nulls."""
    outcome_columns = [column for column in OUTCOME_COLUMNS if column in data.columns]
    for column in outcome_columns:
        outcome = data[column].mask(data[column].isin((-999, "-999")), np.nan)
        data[column] = outcome.astype("float64")


def select_patient_stays(deterioration: pd.DataFrame, selection: PatientStaySelection | None) -> pd.DataFrame:
    """Select the requested number of stays per patient."""
    if selection is None:
        return deterioration
    if selection not in PATIENT_STAY_SELECTIONS:
        raise ValueError(f"selection must be one of {PATIENT_STAY_SELECTIONS}.")

    deterioration = deterioration.sort_values(
        ["subject_id", "general_intime", "ed_stay_id"],
        kind="stable",
    )
    if selection == "all":
        return deterioration
    if selection == "random":
        return deterioration.groupby("subject_id", group_keys=False).sample(
            n=1, random_state=RANDOM_SEED
        )
    return deterioration.drop_duplicates("subject_id", keep=selection)


def select_window_candidates(
    deterioration: pd.DataFrame,
    records: pd.DataFrame,
    record: RecordMode,
    time_column: str,
    study_column: str,
    sort_column: str,
    rank_column: str,
    count_column: str,
) -> pd.DataFrame:
    """Rank records in each stay's first 90 minutes and select the requested rows."""
    candidates = deterioration[
        ["benchmark_row_id", "subject_id", "general_intime", "general_90min"]
    ].merge(records, on="subject_id", how="inner")
    candidates = candidates.loc[
        candidates[time_column].between(
            candidates["general_intime"],
            candidates["general_90min"],
            inclusive="both",
        )
    ].copy()
    if candidates.empty:
        candidates[rank_column] = pd.Series(dtype="Int64")
        candidates[count_column] = pd.Series(dtype="Int64")
        return candidates

    candidates[count_column] = candidates.groupby("benchmark_row_id")[
        study_column
    ].transform("size")
    candidates = candidates.sort_values(
        ["benchmark_row_id", time_column, sort_column],
        ascending=[True, record != "last", record != "last"],
    )
    candidates[rank_column] = candidates.groupby("benchmark_row_id").cumcount() + 1
    if record in ("average", "all"):
        return candidates
    return candidates.loc[candidates[rank_column].eq(1)].copy()


def validate_outcome_alignment(key_table: pd.DataFrame, deterioration: pd.DataFrame) -> None:
    """Check that output rows retain their source outcome labels."""
    source_outcomes = deterioration.set_index("benchmark_row_id")[OUTCOME_COLUMNS]
    expected_outcomes = source_outcomes.loc[key_table["benchmark_row_id"]].reset_index(
        drop=True
    )
    exported_outcomes = key_table[OUTCOME_COLUMNS].reset_index(drop=True)
    if not exported_outcomes.equals(expected_outcomes):
        raise ValueError(
            "Output key rows do not retain their source deterioration outcome labels."
        )


def build_key_table(
    base: pd.DataFrame,
    ecg_keys: pd.DataFrame,
    cxr_keys: pd.DataFrame,
    record: RecordMode,
    patient_stay_selection: PatientStaySelection | None = None,
) -> pd.DataFrame:
    if record not in RECORD_MODES:
        raise ValueError(f"record must be one of {RECORD_MODES}.")

    # The base frame has one row per stay and already has renamed identifiers and parsed dates
    # Keep the benchmark alias for the downstream split step
    deterioration = base[
        [
            "subject_id",
            "ed_stay_id",
            "ed_hadm_id",
            "strat_fold",
            "general_intime",
            "general_90min",
            *OUTCOME_COLUMNS,
        ]
    ].copy()
    deterioration["benchmark_row_id"] = deterioration["ed_stay_id"]
    if deterioration[["general_intime", "general_90min"]].isna().any().any():
        raise ValueError("Deterioration data contain unparseable required datetime values.")
    deterioration = select_patient_stays(deterioration, patient_stay_selection)

    ecg = ecg_keys.loc[
        ecg_keys["subject_id"].isin(set(deterioration["subject_id"])),
        ["subject_id", "study_id", "ecg_time", "path"],
    ].copy()
    ecg["ecgfm_key_row_id"] = ecg.index
    ecg = ecg.rename(columns={"study_id": "ecg_id", "path": "ecg_path"})
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    if ecg["ecg_time"].isna().any():
        raise ValueError("ECG keys contain rows with unparseable ecg_time values.")
    validate_unique_keys(ecg, ["subject_id", "ecg_id"], "ECG keys")

    ecg_candidates = select_window_candidates(
        deterioration,
        ecg,
        record,
        time_column="ecg_time",
        study_column="ecg_id",
        sort_column="ecgfm_key_row_id",
        rank_column="ecg_match_rank",
        count_column="n_ecg_records_in_90min",
    )
    n_matched_benchmark_rows = ecg_candidates["benchmark_row_id"].nunique()
    if n_matched_benchmark_rows != len(deterioration):
        raise ValueError(
            "Not all deterioration rows have an ECG recording within the first 90 minutes."
        )
    if record in ("average", "all"):
        selected_ecg = (
            ecg_candidates.groupby(["benchmark_row_id", "subject_id"], as_index=False)
            .agg(
                ecgfm_key_row_id=("ecgfm_key_row_id", join_selected_values),
                ecg_id=("ecg_id", join_selected_values),
                ecg_time=("ecg_time", join_selected_values),
                ecg_path=("ecg_path", join_selected_values),
                n_ecg_records_in_90min=("n_ecg_records_in_90min", "first"),
            )
        )
        selected_ecg["ecg_match_rank"] = pd.NA
    else:
        selected_ecg = ecg_candidates
    selected_ecg = selected_ecg[
        [
            "benchmark_row_id",
            "subject_id",
            "ecgfm_key_row_id",
            "ecg_id",
            "ecg_time",
            "ecg_path",
            "ecg_match_rank",
            "n_ecg_records_in_90min",
        ]
    ].copy()
    linked = deterioration.merge(
        selected_ecg,
        on=["benchmark_row_id", "subject_id"],
        how="inner",
        validate="one_to_one",
    )
    linked["has_ecg_embedding"] = True

    cxr_studies = aggregate_cxr_studies(
        cxr_keys.loc[cxr_keys["subject_id"].isin(set(deterioration["subject_id"]))]
    )
    cxr_candidates = select_window_candidates(
        deterioration,
        cxr_studies,
        record,
        time_column="cxr_time",
        study_column="cxr_study_id",
        sort_column="cxr_first_key_row_id",
        rank_column="cxr_match_rank",
        count_column="n_cxr_studies_in_90min",
    )
    if record in ("average", "all") and not cxr_candidates.empty:
        selected_cxr = (
            cxr_candidates.groupby(["benchmark_row_id", "subject_id"], as_index=False)
            .agg(
                cxr_key_row_ids=("cxr_key_row_ids", join_selected_values),
                cxr_study_id=("cxr_study_id", join_selected_values),
                cxr_time=("cxr_image_times", join_selected_values),
                cxr_dicom_ids=("cxr_dicom_ids", join_selected_values),
                cxr_paths=("cxr_paths", join_selected_values),
                cxr_view_positions=("cxr_view_positions", join_selected_values),
                n_cxr_images=("n_cxr_images", "sum"),
                n_cxr_studies_in_90min=("n_cxr_studies_in_90min", "first"),
            )
        )
        selected_cxr["cxr_match_rank"] = pd.NA
    else:
        selected_cxr = cxr_candidates
    selected_cxr = selected_cxr[
        [
            "benchmark_row_id",
            "subject_id",
            "cxr_key_row_ids",
            "cxr_study_id",
            "cxr_time",
            "cxr_dicom_ids",
            "cxr_paths",
            "cxr_view_positions",
            "n_cxr_images",
            "cxr_match_rank",
            "n_cxr_studies_in_90min",
        ]
    ].copy()

    output = linked.merge(
        selected_cxr,
        on=["benchmark_row_id", "subject_id"],
        how="left",
        validate="one_to_one",
    )
    output["record"] = record
    output["has_cxr_embedding"] = output["cxr_study_id"].notna()
    output["n_cxr_images"] = output["n_cxr_images"].fillna(0).astype("int64")
    output["n_cxr_studies_in_90min"] = (
        output["n_cxr_studies_in_90min"].fillna(0).astype("int64")
    )
    normalise_nullable_integer_column(output, "ed_stay_id")
    normalise_nullable_integer_column(output, "ed_hadm_id")
    output["ecg_match_rank"] = output["ecg_match_rank"].astype("Int64")
    output["cxr_match_rank"] = output["cxr_match_rank"].astype("Int64")
    output = output[KEY_TABLE_COLUMNS].sort_values(
        ["benchmark_row_id", "ecg_match_rank", "cxr_match_rank"], na_position="last"
    )
    validate_unique_keys(output, ["ed_stay_id"], "Output key table")
    validate_outcome_alignment(output, deterioration)
    return output


def select_cohort_keys(key_table: pd.DataFrame, cohort: CohortSelection) -> pd.DataFrame:
    """Return all linked rows or only rows with both embeddings."""
    required_columns = {"has_ecg_embedding", "has_cxr_embedding"}
    missing_columns = required_columns.difference(key_table.columns)
    if missing_columns:
        raise ValueError(
            f"Key table is missing cohort selection columns: {sorted(missing_columns)}."
        )
    if cohort == "all":
        return key_table.reset_index(drop=True)
    return key_table.loc[
        key_table["has_ecg_embedding"] & key_table["has_cxr_embedding"]
    ].reset_index(drop=True)


def load_ts_long(
    path: Path,
    feature_columns: list[str] | None = None,
    stay_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read the time-series file and replace its base-excess missing marker."""
    ts_long = pd.read_csv(path, low_memory=False)
    if feature_columns is None:
        feature_columns = get_ts_feature_columns(ts_long.columns.tolist())
    if stay_ids is not None:
        ts_long = ts_long.loc[ts_long["ed_stay_id"].isin(stay_ids)]
    ts_long = ts_long.copy()
    base_excess_columns = [column for column in feature_columns if column.startswith(TS_BASE_EXCESS_PREFIX)]
    if base_excess_columns:
        ts_long[base_excess_columns] = ts_long[base_excess_columns].replace(
            TS_BASE_EXCESS_SENTINEL, np.nan
        )
    return ts_long, feature_columns


def aggregate_ts_stats(
    ts_long: pd.DataFrame,
    feature_columns: list[str],
    stay_ids: pd.Index | np.ndarray | list[int],
) -> pd.DataFrame:
    """Calculate per-stay summary statistics for each time-series feature."""
    ordered_columns = [f"{feature}_{stat}" for feature in feature_columns for stat in TS_STAT_NAMES]
    stay_index = pd.Index(np.asarray(stay_ids, dtype="int64"), name="ed_stay_id")
    long = ts_long.melt(
        id_vars=["ed_stay_id", "minutes_since_intime"],
        value_vars=feature_columns,
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])
    if long.empty:
        return pd.DataFrame(
            index=stay_index, columns=ordered_columns, dtype="float64"
        )
    long = long.sort_values(
        ["feature", "ed_stay_id", "minutes_since_intime"], kind="stable"
    )
    long["tv"] = long["minutes_since_intime"] * long["value"]
    long["tt"] = long["minutes_since_intime"] ** 2
    grouped = long.groupby(["ed_stay_id", "feature"], sort=False)
    agg = grouped.agg(
        mean=("value", "mean"),
        median=("value", "median"),
        min=("value", "min"),
        max=("value", "max"),
        first=("value", "first"),
        last=("value", "last"),
        n=("value", "size"),
        sum_t=("minutes_since_intime", "sum"),
        sum_v=("value", "sum"),
        sum_tv=("tv", "sum"),
        sum_tt=("tt", "sum"),
    )
    agg["std"] = grouped["value"].std(ddof=0)
    n = agg["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = agg["sum_tv"] / n - (agg["sum_t"] / n) * (agg["sum_v"] / n)
        vart = agg["sum_tt"] / n - (agg["sum_t"] / n) ** 2
        agg["coeff"] = np.where((n >= 2) & (vart > 0), cov / vart, 0.0)
        agg["rate_change"] = np.where(
            agg["first"] != 0, (agg["last"] - agg["first"]) / agg["first"], np.nan
        )
    wide = agg[TS_STAT_NAMES].unstack("feature")
    wide.columns = [f"{feature}_{stat}" for stat, feature in wide.columns]
    return wide.reindex(index=stay_index, columns=ordered_columns)


def build_tabular_modality(
    cohort_keys: pd.DataFrame,
    base: pd.DataFrame,
    ts_stats: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Write one tabular row per key-table row."""
    validate_unique_keys(cohort_keys, ["ed_stay_id"], "Tabular output key table")
    id_columns = ["benchmark_row_id", "subject_id", "ed_stay_id", "ed_hadm_id", "strat_fold"]
    feature_columns = [column for column in base.columns if column.startswith(BASE_TABULAR_PREFIXES)]
    base_selected = base.set_index("ed_stay_id")[[*OUTCOME_COLUMNS, *feature_columns]]
    order = cohort_keys[id_columns].reset_index(drop=True)
    missing_ids = order.loc[
        ~order["ed_stay_id"].isin(base_selected.index), "ed_stay_id"
    ].tolist()
    if missing_ids:
        raise ValueError(
            "Key table references stays missing from mds_ed_base.csv. "
            f"Example ed_stay_id values: {missing_ids[:5]}."
        )
    tabular = order.join(base_selected, on="ed_stay_id")
    if ts_stats is not None:
        tabular = tabular.join(ts_stats, on="ed_stay_id")
    if len(tabular) != len(cohort_keys) or not tabular["ed_stay_id"].equals(
        order["ed_stay_id"]
    ):
        raise ValueError("Tabular data are not ordered with the key table.")

    if "vitals_acuity" in tabular.columns:
        values = pd.to_numeric(tabular["vitals_acuity"], errors="coerce")
        tabular["vitals_acuity"] = values - 1
    normalise_nullable_integer_column(tabular, "ed_stay_id")
    normalise_nullable_integer_column(tabular, "ed_hadm_id")
    # Write boolean demographic flags as 0/1, like the other indicators
    bool_columns = tabular.select_dtypes(include="bool").columns
    if len(bool_columns):
        tabular[bool_columns] = tabular[bool_columns].astype("int8")
    tabular.to_csv(output_path, index=False)


def identifier_is_selected(actual: object, selected_values: object) -> bool:
    """Check an identifier against pipe-separated values."""
    return any(
        identifier_matches(actual, expected)
        for expected in str(selected_values).split("|")
    )


def extract_ecg_embeddings(
    cohort_keys: pd.DataFrame,
    source_embeddings: pd.DataFrame,
    record: RecordMode,
) -> np.ndarray:
    """Extract one ECG embedding sequence per stay."""
    feature_columns = get_ecg_feature_columns(source_embeddings)
    sequence_features: list[np.ndarray] = []
    for output_row, (_, key_row) in enumerate(cohort_keys.iterrows()):
        row_ids = parse_row_ids(
            key_row["ecgfm_key_row_id"], "ecgfm_key_row_id", output_row
        )
        validate_row_ids(row_ids, len(source_embeddings), "ecgfm_key_row_id")
        selected = source_embeddings.iloc[row_ids]
        if "subject_id" in selected and not all(
            identifier_matches(value, key_row["subject_id"])
            for value in selected["subject_id"]
        ):
            raise ValueError(
                f"ECG subject_id does not match the key table at output row {output_row}."
            )
        selected_features = selected.loc[:, feature_columns].to_numpy()
        if record == "average":
            selected_features = selected_features.mean(axis=0, keepdims=True)
        sequence_features.append(selected_features)
    return to_sequence_array(sequence_features)


def extract_cxr_embeddings(
    cohort_keys: pd.DataFrame,
    source_embeddings: pd.DataFrame,
    record: RecordMode,
) -> np.ndarray:
    """Extract one CXR embedding sequence per stay."""
    feature_columns = get_feature_columns(source_embeddings, "cxr_ft")
    feature_dtype = source_embeddings[feature_columns[0]].dtype
    sequence_features = [
        np.empty((0, len(feature_columns)), dtype=feature_dtype)
        for _ in range(len(cohort_keys))
    ]
    for output_row, (_, key_row) in enumerate(cohort_keys.iterrows()):
        if not key_row["has_cxr_embedding"]:
            continue
        row_ids = parse_row_ids(
            key_row["cxr_key_row_ids"], "cxr_key_row_ids", output_row
        )
        validate_row_ids(row_ids, len(source_embeddings), "cxr_key_row_ids")
        if len(row_ids) != key_row["n_cxr_images"]:
            raise ValueError(
                f"CXR row identifier count does not match n_cxr_images at output row "
                f"{output_row}."
            )
        selected = source_embeddings.iloc[row_ids]
        if "subject_id" in selected and not all(
            identifier_matches(value, key_row["subject_id"])
            for value in selected["subject_id"]
        ):
            raise ValueError(
                f"CXR subject_id does not match the key table at output row {output_row}."
            )
        if "study_id" in selected and not all(
            identifier_is_selected(value, key_row["cxr_study_id"])
            for value in selected["study_id"]
        ):
            raise ValueError(
                f"CXR study_id does not match the key table at output row {output_row}."
            )
        selected_features = selected.loc[:, feature_columns].to_numpy()
        if record == "average":
            selected_features = selected_features.mean(axis=0, keepdims=True)
        sequence_features[output_row] = selected_features
    return to_sequence_array(sequence_features)


def extract_ts_sequences(
    cohort_keys: pd.DataFrame,
    ts_long: pd.DataFrame,
    feature_columns: list[str],
    record: RecordMode,
) -> tuple[np.ndarray, list[str], list[bool], list[int]]:
    """Build one time-series sequence and its key-table metadata per stay."""
    n_features = len(feature_columns)
    sequences = [np.empty((0, n_features), dtype=np.float32) for _ in range(len(cohort_keys))]
    ts_time = ["" for _ in range(len(cohort_keys))]
    has_ts = [False for _ in range(len(cohort_keys))]
    n_ts_records = [0 for _ in range(len(cohort_keys))]

    stay_groups = dict(
        list(
            ts_long.sort_values(
                ["ed_stay_id", "minutes_since_intime"], kind="stable"
            ).groupby("ed_stay_id", sort=False)
        )
    )
    with warnings.catch_warnings():
        # nanmean warns when a feature is missing at every time point. NaN is the intended result
        warnings.simplefilter("ignore", RuntimeWarning)
        for output_row, stay_id in enumerate(cohort_keys["ed_stay_id"].tolist()):
            group = stay_groups.get(stay_id)
            if group is None or group.empty:
                continue
            values = group[feature_columns].to_numpy(dtype=np.float32)
            times = group["charttime"].astype(str).tolist()
            if record == "all":
                sequence, stay_time = values, "|".join(times)
            elif record == "average":
                sequence = np.nanmean(values, axis=0, keepdims=True).astype(np.float32)
                stay_time = ""
            elif record == "first":
                sequence, stay_time = values[:1], times[0]
            else:  # "last"
                sequence, stay_time = values[-1:], times[-1]
            sequences[output_row] = sequence
            has_ts[output_row] = True
            n_ts_records[output_row] = int(sequence.shape[0])
            ts_time[output_row] = stay_time
    return to_sequence_array(sequences), ts_time, has_ts, n_ts_records


def validate_embedding_alignment(
    cohort_keys: pd.DataFrame,
    ecg_embeddings: np.ndarray,
    cxr_embeddings: np.ndarray,
    ts_sequences: np.ndarray | None = None,
) -> None:
    """Check that each output modality has one row per selected stay."""
    validate_unique_keys(cohort_keys, ["ed_stay_id"], "Exported key table")
    modalities = [("ECG", ecg_embeddings), ("CXR", cxr_embeddings)]
    if ts_sequences is not None:
        modalities.append(("TS", ts_sequences))
    for name, embeddings in modalities:
        if len(embeddings) != len(cohort_keys):
            raise ValueError(
                f"{name} embeddings do not have one row per exported key row."
            )
        validate_embedding_array(name, embeddings)


def write_summary_and_report(
    summary: dict[str, object],
    output_paths: dict[str, Path],
    cohort_keys: pd.DataFrame,
    cohort: CohortSelection,
) -> None:
    """Write summary metadata and print the generated files and counts."""
    print("[9/10] Writing summary counts.")
    with output_paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[10/10] Processing finished.")
    artefacts = [
        ("key table", "keys"),
        ("tabular data", "tabular"),
        ("ECG embeddings", "ecg_embeddings"),
        ("CXR embeddings", "cxr_embeddings"),
    ]
    if summary.get("ts_embeddings_path") is not None:
        artefacts.append(("time-series", "ts_embeddings"))
    for description, key in artefacts:
        print(f"Saved {cohort} cohort {description} to {output_paths[key]}.")
    print(f"Saved summary counts to {output_paths['summary']}.")
    for label, key in [
        ("Deterioration base stays", "n_deterioration_benchmark_rows"),
        ("ECG-FM key rows", "n_ecgfm_key_rows"),
        ("CXR key rows", "n_cxr_key_rows"),
        ("Linked key rows", "n_linked_key_rows"),
        ("Unique ED stays", "n_unique_ed_stays"),
        ("Unique patients", "n_unique_patients"),
        ("Rows with ECG embedding", "n_with_ecg_embedding"),
        ("Rows with CXR embedding", "n_with_cxr_embedding"),
        ("Rows with time-series", "n_with_ts"),
        (f"Exported {cohort} cohort rows", "n_exported_rows"),
    ]:
        print(f"{label}: {summary[key]}")
    for outcome in OUTCOME_COLUMNS:
        n_positive = cohort_keys[outcome].eq(1).sum()
        print(f"Outcome {outcome}: {n_positive:,} positive cases")


def build_summary(
    base: pd.DataFrame,
    ecg_keys: pd.DataFrame,
    cxr_keys: pd.DataFrame,
    key_table: pd.DataFrame,
    cohort_keys: pd.DataFrame,
    output_paths: dict[str, Path],
    args: argparse.Namespace,
    source_row_limit: int | None,
    n_ts_features: int,
    ts_feature_columns: list[str],
) -> dict[str, object]:
    """Build summary metadata from exported counts and linkage details."""
    return {
        "n_deterioration_benchmark_rows": int(len(base)),
        "n_ecgfm_key_rows": int(len(ecg_keys)),
        "n_cxr_key_rows": int(len(cxr_keys)),
        "n_linked_key_rows": int(len(key_table)),
        "n_unique_ed_stays": int(cohort_keys["ed_stay_id"].nunique()),
        "n_unique_patients": int(cohort_keys["subject_id"].nunique()),
        "n_with_ecg_embedding": int(cohort_keys["has_ecg_embedding"].sum()),
        "n_with_cxr_embedding": int(cohort_keys["has_cxr_embedding"].sum()),
        "n_with_ts": int(cohort_keys["has_ts"].sum()),
        "n_complete_case_rows": int(
            (
                cohort_keys["has_ecg_embedding"]
                & cohort_keys["has_cxr_embedding"]
            ).sum()
        ),
        "patient_stay": args.patient_stay,
        "record": args.record,
        "embedding_format": args.embedding_format,
        "embedding_representation": "sequence",
        "ts_feature_dim": int(n_ts_features),
        # Keep the feature names for ``--record all`` so downstream splits can document them
        "ts_columns": (
            list(ts_feature_columns) if args.record == "all" else None
        ),
        "cohort": args.cohort,
        "completeness_modalities": ["ecg", "cxr"],
        "test_mode": args.test_mode,
        "test_row_limit": source_row_limit,
        "n_exported_rows": int(len(cohort_keys)),
        "n_positive_by_outcome": {
            outcome: int(cohort_keys[outcome].eq(1).sum()) for outcome in OUTCOME_COLUMNS
        },
        "bundle_dir": str(output_paths["bundle_dir"]),
        "key_path": str(output_paths["keys"]),
        "tabular_path": str(output_paths["tabular"]),
        "ecg_embeddings_path": str(output_paths["ecg_embeddings"]),
        "cxr_embeddings_path": str(output_paths["cxr_embeddings"]),
        "ts_embeddings_path": (
            str(output_paths["ts_embeddings"]) if args.record == "all" else None
        ),
    }


def main() -> None:
    args = parse_args()
    source_row_limit = TEST_ROW_LIMIT if args.test_mode else None
    output_paths = build_output_paths(
        args.output_dir,
        args.patient_stay,
        args.record,
        args.cohort,
        embedding_format=args.embedding_format,
        test_mode=args.test_mode,
    )
    if args.test_mode:
        print(f"Test mode: using the first {TEST_ROW_LIMIT} rows of mds_ed_base.csv.")
    print("[1/10] Loading stay-level base cohort and modality key tables.")
    base = read_base(args.base_path, nrows=source_row_limit)
    ecg_keys = pd.read_csv(DEFAULT_ECG_KEYS_PATH)
    cxr_keys = pd.read_csv(DEFAULT_CXR_KEYS_PATH)
    print("[2/10] Linking stays to ECG and selected CXR keys.")
    key_table = build_key_table(
        base=base,
        ecg_keys=ecg_keys,
        cxr_keys=cxr_keys,
        record=args.record,
        patient_stay_selection=args.patient_stay,
    )
    cohort_keys = select_cohort_keys(key_table, args.cohort)
    validate_unique_keys(cohort_keys, ["ed_stay_id"], "Exported key table")

    print("[3/10] Loading vitals/labs time-series and extracting per-stay sequences.")
    ts_long, ts_feature_columns = load_ts_long(
        args.ts_path, stay_ids=set(cohort_keys["ed_stay_id"].tolist())
    )
    ts_sequences, ts_time, has_ts, n_ts_records = extract_ts_sequences(
        cohort_keys, ts_long, ts_feature_columns, args.record
    )
    cohort_keys["has_ts"] = has_ts
    if args.record == "all":
        # ``all`` stores the raw sequence and the time and count columns needed downstream
        cohort_keys["ts_time"] = ts_time
        cohort_keys["n_ts_records"] = n_ts_records
        cohort_keys = cohort_keys[OUTPUT_COLUMNS]
    else:
        # Other modes store time-series summaries in the tabular file
        ts_sequences = None
        cohort_keys = cohort_keys[SUMMARY_KEY_COLUMNS]

    print(f"[4/10] Writing {args.cohort} cohort keys ({len(cohort_keys):,} rows).")
    output_paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    cohort_keys.to_csv(output_paths["keys"], index=False)

    print("[5/10] Building tabular modality (base features + time-series aggregates).")
    # ``all`` stores the raw sequence, while other modes append summary columns
    ts_stats = (
        None
        if args.record == "all"
        else aggregate_ts_stats(
            ts_long, ts_feature_columns, cohort_keys["ed_stay_id"].astype("int64").to_numpy()
        )
    )
    build_tabular_modality(cohort_keys, base, ts_stats, output_paths["tabular"])

    print("[6/10] Loading and extracting ECG embeddings.")
    ecg_source_embeddings = pd.read_pickle(DEFAULT_ECG_EMBEDDINGS_PATH)
    ecg_embeddings = extract_ecg_embeddings(
        cohort_keys, ecg_source_embeddings, args.record
    )
    del ecg_source_embeddings

    print("[7/10] Loading and extracting CXR embeddings.")
    cxr_source_embeddings = pd.read_pickle(DEFAULT_CXR_EMBEDDINGS_PATH)
    cxr_embeddings = extract_cxr_embeddings(
        cohort_keys, cxr_source_embeddings, args.record
    )
    del cxr_source_embeddings

    print("[8/10] Validating aligned modality rows and writing embeddings.")
    validate_embedding_alignment(cohort_keys, ecg_embeddings, cxr_embeddings, ts_sequences)
    write_embedding_file(
        ecg_embeddings,
        output_paths["ecg_embeddings"],
        "ecg_embedding_sequence",
        args.embedding_format,
    )
    write_embedding_file(
        cxr_embeddings,
        output_paths["cxr_embeddings"],
        "cxr_embedding_sequence",
        args.embedding_format,
    )
    if ts_sequences is not None:
        write_embedding_file(
            ts_sequences,
            output_paths["ts_embeddings"],
            "ts_sequence",
            args.embedding_format,
        )

    summary = build_summary(
        base=base,
        ecg_keys=ecg_keys,
        cxr_keys=cxr_keys,
        key_table=key_table,
        cohort_keys=cohort_keys,
        output_paths=output_paths,
        args=args,
        source_row_limit=source_row_limit,
        n_ts_features=len(ts_feature_columns),
        ts_feature_columns=ts_feature_columns,
    )
    write_summary_and_report(summary, output_paths, cohort_keys, args.cohort)

if __name__ == "__main__":
    main()
