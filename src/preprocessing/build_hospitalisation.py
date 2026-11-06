"""Build the dual-window hospitalisation cohort and event tables.

The builder writes one row per patient, organised by their last hospital admission, with static
features and event tables for the last and previous ED-linked admissions.

Usage::

    # full optionalHistory run (default)
    python -m src.preprocessing.build_hospitalisation

    # cohort and static tables only
    python -m src.preprocessing.build_hospitalisation --skip-events

    # require an in-band previous admission
    python -m src.preprocessing.build_hospitalisation --history required
"""

from __future__ import annotations

from collections.abc import Iterable

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np
import pandas as pd

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
MIMIC_DIR = WORKSPACE_ROOT / "data" / "MIMIC-IV"

MIMIC_HOSP_RELEASE = "3.1"
HOSP_DIR = MIMIC_DIR / "mimiciv" / MIMIC_HOSP_RELEASE / "hosp"
ED_DIR = MIMIC_DIR / "mimic-iv-ed" / "2.2" / "ed"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "hospitalisation" / MIMIC_HOSP_RELEASE / "extract"

LABEVENTS_PATH = HOSP_DIR / "labevents.csv"
LOOKBACK_DAYS = 90
MIN_AGE = 18
TEST_ROW_LIMIT = 50_000

HistoryRequirement: TypeAlias = Literal["required", "optional"]
HISTORY_REQUIREMENTS = get_args(HistoryRequirement)
HISTORY_SUBDIRS: dict[HistoryRequirement, str] = {
    "required": "requiredHistory",
    "optional": "optionalHistory",
}
# Names used for cohort counts in the run summary
COHORT_SIZE_STAGES: dict[HistoryRequirement, str] = {
    requirement: f"cohort_{requirement}_history" for requirement in HISTORY_REQUIREMENTS
}

REQUIRED_DEMOGRAPHICS = ("anchor_age", "gender", "marital_status", "insurance")
# Gender uses 1 for male and 0 for female
MALE_GENDER_CODE = "M"
# Text values treated as missing
PLACEHOLDER_VALUES = ("", "NULL", "NAN", "NONE")
# Unknown values remain valid categories
CATEGORY_LIKE_PLACEHOLDERS = ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER")
ARRIVAL_TRANSPORT_UNKNOWN_LEVEL = "UNKNOWN"
ADMISSION_LOCATION_NA_LEVEL = "INFORMATION NOT AVAILABLE"
RACE_POOLED_INTO_OTHER = ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER")

EVENT_COLUMNS = ("subject_id", "charttime", "itemid", "label", "fluid", "value", "valueuom", "linksto")

# Units used in the ED table
ED_VITALS_UNITS = {"temperature": "degF", "heartrate": "bpm", "resprate": "insp/min",
                   "o2sat": "%", "sbp": "mm Hg", "dbp": "mm Hg"}
# Temperature is converted to Celsius to match MDS-ED
ED_VITALS_TEMPERATURE_UNIT = "degC"
VITALS_SOURCE = "vitals_measurements"
LABS_SOURCE = "labevents"
ED_VITALS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")

# Prefixes used for vital and laboratory features
TS_SOURCE_PREFIXES = {VITALS_SOURCE: "vitals_", LABS_SOURCE: "labvalues_"}
# Metadata columns are written in this order
EVENT_META_COLUMNS = ("subject_id", "hadm_id", "window_start", "window_end", "charttime", "elapsed")
WINDOW_SPECS = {
    "prev": {"hadm_id": "prev_hadm_id", "window_start": "prev_start", "window_end": "prev_end",
             "elapsed": "minutes_since_prev_start", "filename": "events_prev.csv"},
    "curr": {"hadm_id": "last_hadm_id", "window_start": "curr_start", "window_end": "curr_end",
             "elapsed": "minutes_since_curr_start", "filename": "events_curr.csv"},
}

CONFIG_DIR = REPO_ROOT / "config"
LTC_MAPPING_PATH = CONFIG_DIR / "ltc_mapping.json"
# Electronic medication records
EMAR_PATH = HOSP_DIR / "emar.csv.gz"
# The medication table includes holds, stops and errors
EMAR_ADMINISTERED_EVENTS = ("Administered", "Confirmed", "Started")
ICD9TO10_PATH = CONFIG_DIR / "icd9to10.txt"
ONCOLOGY_PATH = CONFIG_DIR / "icd10oncology.txt"
# Laboratory tests and ED vital signs to extract
EVENT_ITEMS_PATH = CONFIG_DIR / "event_items.csv"
# Unit conversions for laboratory results
LAB_UNIT_HARMONISATION = {51249: {"%": ("g/dL", 1.0)}}
# Weight, height and BMI from outpatient records
OMR_PATH = HOSP_DIR / "omr.csv"
OMR_MEASURES = {"BMI (kg/m2)": "bmi", "Weight (Lbs)": "weight", "Height (Inches)": "height"}
# Convert measurements to metric units
OMR_UNIT_CONVERSIONS = {"Height (Inches)": 2.54, "Weight (Lbs)": 0.453592}
# Plausibility limits applied after unit conversion
OMR_BOUNDS = {"bmi": (None, 100.0), "weight": (20.0, 400.0), "height": (60.0, 400.0)}
OMR_LOOKBACK_DAYS = 90
OMR_INCLUDE_ANCHOR_DAY = True
EXPECTED_LAB_PANEL_SIZE = 50

# Thresholds for multimorbidity features
MM_CUTOFF = 1
CMM_CUTOFF = 3

# Most frequently administered drugs in this cohort
TOP_MEDICATIONS = 50
# Default values for medication timing features
LEGACY_DAYS_SINCE_FILL = 0
# Medication names combined into common groups
MEDICATION_COLLAPSES = {"vancomycin": "vancomycin", "acetaminophen": "acetaminophen",
                        "albuterol_0.083%_neb_soln": "albuterol_neb_soln",
                        "oxycodone_(immediate_release)": "oxycodone"}

# Order types counted as care contacts
POE_ORDER_TYPES = {"Cardiology": "pon_cardiology", "Respiratory": "pon_respiratory",
                   "Radiology": "pon_radiology", "TPN": "pon_tpn", "Nutrition": "pon_nutrition",
                   "Neurology": "pon_neurology", "Hemodialysis": "pon_hemodialysis"}


UNRECORDED_DISCHARGE_LEVEL = "UNKNOWN"
HOME_LIKE_DISCHARGE = ("HOME", "HOME HEALTH CARE", "DIED", "AGAINST ADVICE", "ASSISTED LIVING",
                       "UNKNOWN")
# Threshold for an extended stay
EXT_STAY_THRESHOLD_DAYS = 7
# Outcomes for the current admission
INDEX_OUTCOME_LABELS = ("ext_stay_7", "in_hosp_death", "non_home_discharge")
OUTCOME_PREFIX = "hospitalisation_"
STATIC_OUTCOME_LABELS = tuple(f"{OUTCOME_PREFIX}{label}" for label in INDEX_OUTCOME_LABELS)

# Stored in the static table for later window calculations
STATIC_TIMESTAMPS = ("edregtime", "edouttime", "admittime", "dischtime", "prev_edregtime",
                     "prev_admittime", "prev_dischtime")

# Fields from the previous admission, notes use the previous admission ID
PREVIOUS_ADMISSION_COLUMNS = (
    "prev_hadm_id", "prev_ed_stay_id", "prev_edregtime", "prev_admittime", "prev_dischtime",
    "prev_non_home_discharge", "prev_discharge_location", "prev_start", "prev_end",
    "days_since_last_discharge",
)

# Each static column belongs to one feature family
STATIC_KEY_COLUMNS = ("subject_id", "last_hadm_id", "ed_stay_id")
STATIC_PREFIXES = ("general_", "demographics_", "biometrics_", "encounter_", "vitals_", "history_",
                   "conditions_", "medications_", "careorders_", OUTCOME_PREFIX)
# Direct column renames
STATIC_RENAMES = {
    **{column: f"general_{column}" for column in STATIC_TIMESTAMPS},
    "anchor_age": "demographics_age",
    "gender": "demographics_gender",
    "race_group": "demographics_ethnicity",
    "marital_status": "demographics_marital_status",
    "insurance": "demographics_insurance",
    # Use the standard names for these measures
    "omr_weight_kg": "biometrics_weight",
    "omr_height_cm": "biometrics_height",
    "omr_bmi": "biometrics_bmi",
    "admission_location": "encounter_admission_location",
    "arrival_transport": "encounter_arrival_transport",
    "ed_phase_hours": "encounter_ed_phase_hours",
    # Keep the acuity name unchanged
    "prev_los_days": "history_los_days",
    "prev_discharge_location": "history_discharge_location",
    "prev_non_home_discharge": "history_non_home_discharge",
    "days_since_last_discharge": "history_days_since_discharge",
    "n_prior_admissions": "history_n_prior_admissions",
    "phys_men_multimorbidity": "conditions_phys_men_multimorbidity",
    "n_unique_conditions": "conditions_n_unique",
    "is_multimorbid": "conditions_is_multimorbid",
    "is_complex_multimorbid": "conditions_is_complex_multimorbid",
    "total_n_presc": "medications_total_n",
    # Counts distinct specialties, capped at 7
    "total_proc_count": "careorders_n_specialties",
    **{name: f"careorders_{name[len('pon_'):]}" for name in POE_ORDER_TYPES.values()},
    **{label: f"{OUTCOME_PREFIX}{label}" for label in INDEX_OUTCOME_LABELS},
}
# Rename groups of related columns consistently
CONDITION_RENAME_PREFIXES = (("physltc_", "conditions_phys_"), ("menltc_", "conditions_men_"))
MEDICATION_RENAME_SUFFIXES = (("n_presc_", "n"), ("dsf_", "dsf"), ("dsl_", "dsl"))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the dual-window hospitalisation cohort: cohort keys, static block, the two "
            "wide event tables and their schema."
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "PARENT directory for the output tables. Every table is written to the "
            "requiredHistory/ or optionalHistory/ subdirectory of it, named for --history, so "
            "the two extracts never share a directory and neither is the unlabelled default "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=LOOKBACK_DAYS,
        help=(
            "Upper age of the previous discharge, in days. Sets has_prev_band without filtering "
            f"(default: {LOOKBACK_DAYS})."
        ),
    )
    parser.add_argument(
        "--history",
        choices=HISTORY_REQUIREMENTS,
        default="optional",
        help=(
            "Whether an in-band previous admission is required for cohort membership. "
            "'required' narrows to the has_prev_band subset BEFORE the labevents pass, making "
            "that scan ~7x cheaper but unable to yield the superset; 'optional' keeps every "
            "patient, with the previous window left empty for those without one "
            "(default: optional)."
        ),
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Exit after the cohort and static block, skipping the labevents pass.",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        action="store_true",
        help=f"Restrict to the first {TEST_ROW_LIMIT} admissions.",
    )
    args = parser.parse_args()

    if args.lookback_days <= 0:
        parser.error(
            f"--lookback-days must be positive; got {args.lookback_days}. A non-positive band "
            "admits no previous stay at all."
        )
    return args


def is_valid_categorical(values: pd.Series) -> pd.Series:
    """Check whether category values are present and usable."""
    text = values.astype("string").str.strip()
    return text.notna() & ~text.str.upper().isin(PLACEHOLDER_VALUES)


def read_admissions(nrows: int | None = None) -> pd.DataFrame:
    """Read adult hospital stays and check their dates."""
    admissions = pd.read_csv(
        HOSP_DIR / "admissions.csv",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "edregtime",
                 "edouttime", "marital_status", "insurance", "race", "admission_location",
                 "discharge_location"],
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
        nrows=nrows,
    )
    patients = pd.read_csv(
        HOSP_DIR / "patients.csv",
        usecols=["subject_id", "anchor_age", "gender", "dod"],
        parse_dates=["dod"],
    )

    admissions = admissions.dropna(subset=["edregtime", "edouttime"])
    admissions = admissions[
        (admissions["edregtime"] < admissions["admittime"])
        & (admissions["edregtime"] < admissions["dischtime"])
        & (admissions["edouttime"] < admissions["dischtime"])
        & (admissions["edouttime"] > admissions["edregtime"])
        & (admissions["admittime"] < admissions["dischtime"])
    ]
    # Ignore death times that precede the admission
    admissions = admissions[
        admissions["deathtime"].isna() | admissions["deathtime"].ge(admissions["admittime"])
    ]
    admissions = admissions.merge(patients, on="subject_id", how="left")
    admissions = admissions[admissions["anchor_age"] >= MIN_AGE].copy()

    return add_index_outcomes(admissions)


def flag_in_hospital_death(admissions: pd.DataFrame) -> pd.Series:
    """Mark stays where death was recorded during the admission."""
    flagged = admissions["deathtime"].notna()
    legacy = admissions["dod"].le(admissions["dischtime"]) & admissions["dod"].gt(
        admissions["admittime"]
    )
    outside = flagged & ~admissions["deathtime"].between(
        admissions["admittime"], admissions["dischtime"]
    )
    print(
        f"        in-hospital death: {int(flagged.sum()):,} admissions flagged from deathtime "
        f"({int((flagged & ~legacy).sum()):,} missed by the legacy dod rule, "
        f"{int((~flagged & legacy).sum()):,} flagged by it that deathtime does not)"
    )
    if int(outside.sum()):
        print(
            f"        note: {int(outside.sum()):,} deathtime values fall outside their own "
            "[admittime, dischtime] interval and are flagged anyway"
        )
    return flagged.astype("int8")


def add_index_outcomes(admissions: pd.DataFrame) -> pd.DataFrame:
    """Add length of stay and outcome fields to each admission."""
    admissions = admissions.copy()
    admissions["discharge_location"] = admissions["discharge_location"].fillna(
        UNRECORDED_DISCHARGE_LEVEL
    )
    admissions["los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86_400
    admissions["ext_stay_7"] = (
        admissions["los_days"] > EXT_STAY_THRESHOLD_DAYS
    ).astype("int8")
    in_hospital_death = flag_in_hospital_death(admissions)
    admissions["in_hosp_death"] = in_hospital_death
    admissions["non_home_discharge"] = (
        ~admissions["discharge_location"].isin(HOME_LIKE_DISCHARGE) & in_hospital_death.eq(0)
    ).astype("int8")
    return admissions


def require_ed_linkage(admissions: pd.DataFrame) -> pd.DataFrame:
    """Keep admissions linked to emergency department records."""
    edstays = pd.read_csv(
        ED_DIR / "edstays.csv.gz", usecols=["hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    ).dropna(subset=["hadm_id"])
    edstays["hadm_id"] = edstays["hadm_id"].astype("int64")
    edstays = edstays.sort_values(["hadm_id", "stay_id"]).drop_duplicates("hadm_id", keep="first")
    edstays = edstays.rename(
        columns={"stay_id": "ed_stay_id", "intime": "ed_intime", "outtime": "ed_outtime"})
    return admissions.merge(edstays, on="hadm_id", how="inner")


def require_complete_demographics(admissions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep adults with recorded age, gender, marital status and insurance."""
    valid = {
        "anchor_age": admissions["anchor_age"].notna() & (admissions["anchor_age"] >= MIN_AGE),
        "gender": is_valid_categorical(admissions["gender"]),
        "marital_status": is_valid_categorical(admissions["marital_status"]),
        "insurance": is_valid_categorical(admissions["insurance"]),
    }
    invalid = {field: int((~mask).sum()) for field, mask in valid.items()}
    keep = np.logical_and.reduce([valid[field].to_numpy() for field in REQUIRED_DEMOGRAPHICS])
    return admissions[keep].copy(), invalid


# Sort order used to select each patient's latest admission
SELECTION_SORT = ["subject_id", "edregtime", "edouttime", "admittime", "dischtime", "hadm_id"]


def select_last_and_previous(admissions: pd.DataFrame) -> pd.DataFrame:
    """Pair each patient's latest stay with their previous stay when available."""
    admissions = admissions.sort_values(SELECTION_SORT, kind="stable")
    last = admissions.drop_duplicates("subject_id", keep="last")
    remainder = admissions[~admissions["hadm_id"].isin(set(last["hadm_id"]))]
    previous = remainder.drop_duplicates("subject_id", keep="last")[
        ["subject_id", "hadm_id", "ed_stay_id", "admittime", "dischtime", "edregtime",
         "non_home_discharge", "discharge_location"]
    ].rename(
        columns={"hadm_id": "prev_hadm_id", "ed_stay_id": "prev_ed_stay_id",
                 "admittime": "prev_admittime", "dischtime": "prev_dischtime",
                 "edregtime": "prev_edregtime", "non_home_discharge": "prev_non_home_discharge",
                 "discharge_location": "prev_discharge_location"}
    )
    cohort = last.merge(previous, on="subject_id", how="left")
    return cohort.rename(columns={"hadm_id": "last_hadm_id"})


def add_window_bounds(cohort: pd.DataFrame) -> pd.DataFrame:
    """Add start and end times for both observation periods."""
    cohort = cohort.copy()
    cohort["curr_start"] = cohort["edregtime"]
    cohort["curr_end"] = cohort["admittime"]
    cohort["prev_start"] = cohort["prev_edregtime"]
    cohort["prev_end"] = cohort["prev_dischtime"]
    return cohort


def apply_lookback_band(cohort: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Mark recent previous stays and remove older ones."""
    cohort = cohort.copy()
    gap = (cohort["edregtime"] - cohort["prev_dischtime"]).dt.total_seconds() / 86_400
    cohort["days_since_last_discharge"] = gap
    cohort["has_prev_any"] = cohort["prev_hadm_id"].notna()
    cohort["has_prev_band"] = gap.between(0, lookback_days).fillna(False)
    out_of_band = cohort["has_prev_any"] & ~cohort["has_prev_band"]
    if out_of_band.any():
        for column in PREVIOUS_ADMISSION_COLUMNS:
            if column in cohort.columns:
                cohort.loc[out_of_band, column] = np.nan
    return cohort


def map_race_group(race: pd.Series) -> pd.Series:
    """Group race descriptions into standard groups."""
    lowered = race.astype("string").str.lower().fillna("")
    groups = pd.Series("Other", index=race.index, dtype="object")

    groups[lowered.str.contains("asian", na=False)] = "Asian"
    groups[lowered.str.contains("hispanic|south american", na=False, regex=True)] = (
        "Hispanic/Latino"
    )
    groups[lowered.str.contains("black|caribbean island", na=False, regex=True)] = "Black"
    groups[lowered.str.contains("white|middle eastern|portuguese", na=False, regex=True)] = (
        "White"
    )
    return groups


def load_omr_anthropometrics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Get the latest recorded weight, height and BMI before admission."""
    if not OMR_PATH.exists():
        raise FileNotFoundError(
            f"OMR table not found at {OMR_PATH}. It supplies the anthropometric static features."
        )
    omr = pd.read_csv(OMR_PATH, usecols=["subject_id", "chartdate", "result_name", "result_value"])
    omr = omr.loc[
        omr["subject_id"].isin(set(cohort["subject_id"]))
        & omr["result_name"].isin(OMR_MEASURES)
    ].copy()
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])
    omr["result_value"] = pd.to_numeric(omr["result_value"], errors="coerce")
    omr = omr.dropna(subset=["result_value"])

    # Convert units before applying limits
    for name, factor in OMR_UNIT_CONVERSIONS.items():
        omr.loc[omr["result_name"] == name, "result_value"] *= factor
    omr["measure"] = omr["result_name"].map(OMR_MEASURES)
    for measure, (low, high) in OMR_BOUNDS.items():
        rows = omr["measure"] == measure
        if low is not None:
            omr.loc[rows & (omr["result_value"] < low), "result_value"] = np.nan
        if high is not None:
            omr.loc[rows & (omr["result_value"] > high), "result_value"] = np.nan
    omr = omr.dropna(subset=["result_value"])

    anchors = cohort[["subject_id", "curr_start"]].copy()
    merged = omr.merge(anchors, on="subject_id", how="inner")
    # Compare dates at midnight because chart dates have no time
    merged["days_before"] = (
        merged["curr_start"].dt.normalize() - merged["chartdate"]
    ).dt.days
    floor = 0 if OMR_INCLUDE_ANCHOR_DAY else 1
    merged = merged.loc[
        merged["days_before"].between(floor, OMR_LOOKBACK_DAYS)
    ]
    # Keep the latest matching record for each patient and measure
    nearest = (
        merged.sort_values(["subject_id", "measure", "days_before"], kind="stable")
        .groupby(["subject_id", "measure"], as_index=False)
        .first()
    )
    values = nearest.pivot(index="subject_id", columns="measure", values="result_value")
    frame = cohort[["subject_id"]].copy()
    for measure, column in (
        ("weight", "omr_weight_kg"),
        ("height", "omr_height_cm"),
        ("bmi", "omr_bmi"),
    ):
        frame[column] = frame["subject_id"].map(
            values[measure] if measure in values.columns else pd.Series(dtype="float64")
        )
    return frame


def rename_static_column(column: str) -> str:
    """Return the output name for a patient field."""
    if column in STATIC_RENAMES:
        return STATIC_RENAMES[column]
    for source, target in CONDITION_RENAME_PREFIXES:
        if column.startswith(source):
            return f"{target}{column[len(source):]}"
    for source, suffix in MEDICATION_RENAME_SUFFIXES:
        if column.startswith(source):
            return f"medications_{column[len(source):]}_{suffix}"
    return column


def apply_static_prefixes(static: pd.DataFrame) -> pd.DataFrame:
    """Add a family name to each patient field."""
    renamed = static.rename(columns={column: rename_static_column(column) for column in static})
    duplicated = sorted({c for c in renamed.columns if list(renamed.columns).count(c) > 1})
    if duplicated:
        raise ValueError(f"Renaming static columns collided on: {duplicated}.")
    unclassified = [
        column for column in renamed.columns
        if column not in STATIC_KEY_COLUMNS and not column.startswith(STATIC_PREFIXES)
    ]
    if unclassified:
        raise ValueError(
            f"static.csv columns belong to no family: {unclassified}. Add each to STATIC_RENAMES "
            f"(or a patterned family) under one of {list(STATIC_PREFIXES)}, or to "
            f"STATIC_KEY_COLUMNS if it is an identifier."
        )
    return renamed


def build_static_block(cohort: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Build one row of patient information and outcomes for each patient."""
    triage = pd.read_csv(ED_DIR / "triage.csv.gz", usecols=["stay_id", "acuity"])
    edstays_transport = pd.read_csv(
        ED_DIR / "edstays.csv.gz", usecols=["stay_id", "arrival_transport"]
    )

    static = cohort[
        [
            "subject_id", "last_hadm_id", "ed_stay_id",
            *STATIC_TIMESTAMPS,
            "anchor_age", "gender", "marital_status", "insurance", "race", "admission_location",
            "days_since_last_discharge",
            "prev_non_home_discharge",
            "prev_discharge_location",
        ]
    ].copy()
    static["race_group"] = map_race_group(cohort["race"])
    static["marital_status"] = static["marital_status"].astype("string").str.lower().str.title()
    static = static.drop(columns=["race"])
    # Encode male as 1 and female as 0
    static["gender"] = static["gender"].eq(MALE_GENDER_CODE).astype("int8")
    static["prev_los_days"] = (
        cohort["prev_dischtime"] - cohort["prev_admittime"]).dt.total_seconds() / 86_400
    static["ed_phase_hours"] = (
        cohort["admittime"] - cohort["edregtime"]).dt.total_seconds() / 3600

    prior = admissions.groupby("subject_id").size().rename("n_prior_admissions") - 1
    static = static.merge(prior, on="subject_id", how="left")

    static = static.merge(
        triage.rename(columns={"stay_id": "ed_stay_id", "acuity": "vitals_acuity"}),
        on="ed_stay_id",
        how="left",
    )
    static["vitals_acuity"] = static["vitals_acuity"].clip(upper=4) - 1
    static = static.merge(
        edstays_transport.rename(columns={"stay_id": "ed_stay_id"}), on="ed_stay_id", how="left"
    )
    for column in ("arrival_transport", "admission_location"):
        null_count = int(static[column].isna().sum())
        if null_count:
            raise ValueError(
                f"{column} has {null_count:,} nulls; both are 100% present on this cohort and "
                "their UNKNOWN-style tokens are retained as levels, so a null indicates a lost "
                "merge rather than an unrecorded value."
            )

    static = static.merge(build_ltc_features(cohort), on="subject_id", how="left")
    static = static.merge(build_medication_features(cohort), on="subject_id", how="left")
    static = static.merge(build_care_contact_features(cohort), on="subject_id", how="left")
    static = static.merge(load_omr_anthropometrics(cohort), on="subject_id", how="left")
    static = static.merge(
        cohort[["subject_id", *INDEX_OUTCOME_LABELS]], on="subject_id", how="left"
    )
    return apply_static_prefixes(static)


def match_icd_prefixes(codes: pd.Series, prefixes: Iterable[str]) -> pd.Series:
    """Check whether each diagnosis code starts with a listed prefix."""
    lengths_to_prefixes: dict[int, set[str]] = {}
    for prefix in prefixes:
        text = str(prefix).replace(".", "").strip()
        if text:
            lengths_to_prefixes.setdefault(len(text), set()).add(text)
    if not lengths_to_prefixes:
        return pd.Series(False, index=codes.index)
    # Require at least three characters because shorter values match too many diagnoses
    shortest = min(lengths_to_prefixes)
    if shortest < 3:
        offenders = sorted(lengths_to_prefixes[shortest])[:5]
        raise ValueError(
            f"ICD prefix shorter than 3 characters would over-match: {offenders}. "
            "Check the code list rather than relaxing this guard."
        )
    text = codes.astype("string")
    matched = pd.Series(False, index=codes.index)
    for length, bucket in lengths_to_prefixes.items():
        matched |= text.str.slice(0, length).isin(bucket).fillna(False)
    return matched


def build_ltc_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build long-term condition and multimorbidity fields."""
    with open(LTC_MAPPING_PATH) as handle:
        ltc_mapping = json.load(handle)
    oncology_codes = {
        line.strip().replace(".", "")
        for line in ONCOLOGY_PATH.read_text().splitlines()
        if line.strip()
    }

    subject_ids = set(cohort["subject_id"])
    diagnoses = pd.read_csv(
        HOSP_DIR / "diagnoses_icd.csv",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )
    diagnoses = diagnoses[diagnoses["subject_id"].isin(subject_ids)].copy()
    # Restrict to prior admissions before the index ED registration to prevent label leakage
    prior = pd.read_csv(
        HOSP_DIR / "admissions.csv",
        usecols=["subject_id", "hadm_id", "edregtime"],
        parse_dates=["edregtime"],
    )
    prior = prior[prior["subject_id"].isin(subject_ids)].merge(
        cohort[["subject_id", "edregtime"]].rename(columns={"edregtime": "index_edregtime"}),
        on="subject_id",
        how="inner",
    )
    prior = prior[prior["edregtime"] < prior["index_edregtime"]]
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(set(prior["hadm_id"]))]
    diagnoses["code"] = diagnoses["icd_code"].astype("string").str.strip().str.replace(".", "", regex=False)

    icd_map = pd.read_csv(ICD9TO10_PATH, sep="\t", usecols=["icd9cm", "icd10cm"], dtype="string")
    icd_map = icd_map.dropna().drop_duplicates("icd9cm").set_index("icd9cm")["icd10cm"]
    is_icd9 = diagnoses["icd_version"].eq(9)
    diagnoses.loc[is_icd9, "code"] = diagnoses.loc[is_icd9, "code"].map(icd_map)
    diagnoses = diagnoses.dropna(subset=["code"])

    flags = pd.DataFrame(index=pd.Index(sorted(subject_ids), name="subject_id"))
    for ltc_name, codes in ltc_mapping.items():
    # Match the beginning of each diagnosis code
        hit = diagnoses.loc[match_icd_prefixes(diagnoses["code"], codes), "subject_id"].unique()
        flags[ltc_name] = flags.index.isin(hit).astype("int16")
    onc_hit = diagnoses.loc[
        match_icd_prefixes(diagnoses["code"], oncology_codes), "subject_id"
    ].unique()
    flags["physltc_malignant_tumor"] = (
        flags["physltc_malignant_tumor"] | flags.index.isin(onc_hit)
    ).astype("int16")

    phys_columns = [c for c in flags.columns if c.startswith("physltc_")]
    men_columns = [c for c in flags.columns if c.startswith("menltc_")]
    n_conditions = flags[phys_columns + men_columns].sum(axis=1)
    flags["phys_men_multimorbidity"] = (
        flags[phys_columns].any(axis=1) & flags[men_columns].any(axis=1)
    ).astype("int16")
    flags["n_unique_conditions"] = n_conditions.astype("int16")
    flags["is_multimorbid"] = (n_conditions > MM_CUTOFF).astype("int16")
    flags["is_complex_multimorbid"] = (n_conditions > CMM_CUTOFF).astype("int16")
    return flags.reset_index()


def normalise_medication_names(medications: pd.Series) -> pd.Series:
    """Standardise medication names for matching."""
    names = medications.astype("string").str.lower().str.strip()
    names = names.str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
    for needle, replacement in MEDICATION_COLLAPSES.items():
        names = names.mask(names.str.contains(needle, regex=False, na=False), replacement)
    return names.str.replace("(", "", regex=False).str.replace(")", "", regex=False)


def select_top_medications(names: pd.Series, n: int = TOP_MEDICATIONS) -> list[str]:
    """Choose the most common medications in a consistent order."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}. A non-positive top-N retains no drug.")
    counts = names.dropna().value_counts()
    return sorted(counts.index, key=lambda name: (-int(counts[name]), name))[:n]


def build_medication_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build medication counts and time-since-medication fields."""
    subject_ids = set(cohort["subject_id"])
    chunks = []
    for chunk in pd.read_csv(
        EMAR_PATH,
        usecols=["subject_id", "charttime", "medication", "event_txt"],
        parse_dates=["charttime"],
        chunksize=5_000_000,
    ):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)]
        chunk = chunk.dropna(subset=["medication", "event_txt", "charttime"])
        chunks.append(chunk[chunk["event_txt"].isin(EMAR_ADMINISTERED_EVENTS)])
    events = pd.concat(chunks, ignore_index=True)

    events = events.merge(cohort[["subject_id", "edregtime"]], on="subject_id", how="inner")
    events = events[events["charttime"] < events["edregtime"]].copy()

    events["medication"] = normalise_medication_names(events["medication"])

    retained = select_top_medications(events["medication"], TOP_MEDICATIONS)
    print(
        f"        computed the top {len(retained)} administered drugs from "
        f"{events['medication'].nunique():,} distinct names in this cohort's EMAR rows; the list "
        "is recorded in extract_summary.json as top_medications"
    )
    events = events[events["medication"].isin(retained)].copy()

    days = (events["edregtime"] - events["charttime"]).dt.days
    events["days_since"] = days
    grouped = events.groupby(["subject_id", "medication"])["days_since"]
    wide = pd.DataFrame(
        {
            "n_presc": grouped.size(),
            "dsf": grouped.max(),  # first administration == largest days-since
            "dsl": grouped.min(),  # last administration == smallest days-since
        }
    ).reset_index()

    pivot = wide.pivot(index="subject_id", columns="medication", values=["n_presc", "dsf", "dsl"])
    pivot.columns = [f"{stat}_{drug}" for stat, drug in pivot.columns]
    totals = wide.groupby("subject_id")["medication"].nunique().rename("total_n_presc")

    out = pd.DataFrame(index=pd.Index(sorted(subject_ids), name="subject_id"))
    out = out.join(totals).join(pivot)
    # Patients absent from the pivot have no medication history and retain missing day values
    absent_from_pivot = out["total_n_presc"].isna().to_numpy()
    for drug in retained:
        for stat in ("n_presc", "dsf", "dsl"):
            column = f"{stat}_{drug}"
            if column not in out.columns:
                out[column] = np.nan
    count_columns = [c for c in out.columns if c.startswith("n_presc_") or c == "total_n_presc"]
    day_columns = [c for c in out.columns if c.startswith(("dsf_", "dsl_"))]
    out[count_columns] = out[count_columns].fillna(0).astype("int32")
    out[day_columns] = out[day_columns].fillna(LEGACY_DAYS_SINCE_FILL)
    out[day_columns] = out[day_columns].astype("Int32")
    out.loc[absent_from_pivot, day_columns] = pd.NA
    ordered = ["total_n_presc"] + [
        f"{stat}_{drug}" for stat in ("dsf", "dsl", "n_presc") for drug in retained
    ]
    return out[ordered].reset_index()


def build_care_contact_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Build counts of care contacts recorded before admission."""
    poe = pd.read_csv(
        HOSP_DIR / "poe.csv.gz",
        usecols=["subject_id", "ordertime", "order_type"],
        parse_dates=["ordertime"],
    )
    poe = poe[
        poe["subject_id"].isin(set(cohort["subject_id"]))
        & poe["order_type"].isin(POE_ORDER_TYPES)
    ]
    merged = poe.merge(cohort[["subject_id", "edregtime"]], on="subject_id", how="inner")
    merged = merged[merged["ordertime"] < merged["edregtime"]]

    counts = (
        merged.groupby(["subject_id", "order_type"]).size().unstack("order_type", fill_value=0)
    )
    counts = counts.rename(columns=POE_ORDER_TYPES)
    out = pd.DataFrame(index=pd.Index(sorted(cohort["subject_id"]), name="subject_id"))
    out = out.join(counts)
    for column in POE_ORDER_TYPES.values():
        if column not in out.columns:
            out[column] = 0
    out = out[list(POE_ORDER_TYPES.values())].fillna(0).astype("int32")
    out["total_proc_count"] = out.gt(0).sum(axis=1).astype("int32")
    return out.reset_index()


def load_event_items(path: Path = EVENT_ITEMS_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the list of laboratory tests and vital signs to extract."""
    if not path.exists():
        raise FileNotFoundError(
            f"Lab panel not found at {path}. This file defines which lab assays are extracted and "
            "the plausibility bounds applied to each; it is version-controlled with the repository."
        )
    panel = pd.read_csv(path)
    required = {"itemid", "slug", "lower", "upper"}
    missing_columns = required.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing_columns)}.")
    for column in ("lower", "upper"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["itemid"] = pd.to_numeric(panel["itemid"], errors="coerce")

    if panel["slug"].duplicated().any():
        repeated = sorted(panel.loc[panel["slug"].duplicated(), "slug"].unique())
        raise ValueError(f"{path} repeats slug(s): {repeated}; feature names must be unique.")
    both = panel["lower"].notna() & panel["upper"].notna()
    inverted = panel.loc[both & (panel["lower"] >= panel["upper"])]
    if len(inverted):
        raise ValueError(f"{path} has lower >= upper for: {inverted['slug'].tolist()}.")
    unbounded = panel.loc[panel["lower"].isna() & panel["upper"].isna(), "slug"].tolist()
    if unbounded:
        raise ValueError(
            f"{path} leaves marker(s) {unbounded} with neither bound. Every marker needs at least "
            "one bound, or an explicit note that it is deliberately unbounded."
        )

    labs = panel.loc[panel["itemid"].notna()].copy()
    labs["itemid"] = labs["itemid"].astype("int64")
    if labs["itemid"].duplicated().any():
        repeated = sorted(labs.loc[labs["itemid"].duplicated(), "itemid"].unique())
        raise ValueError(f"{path} repeats itemid(s): {repeated}.")
    if len(labs) != EXPECTED_LAB_PANEL_SIZE:
        raise ValueError(
            f"{path} defines {len(labs)} lab markers but {EXPECTED_LAB_PANEL_SIZE} are expected. "
            "Update EXPECTED_LAB_PANEL_SIZE deliberately if the panel has changed; the downstream "
            "time-series block width changes with it."
        )

    vitals = panel.loc[panel["itemid"].isna()].copy()
    # Limits must cover exactly the six vital signs that are exported
    expected_vitals = set(ED_VITALS)
    named = set(vitals["slug"])
    if named != expected_vitals:
        raise ValueError(
            f"{path} vitals rows name {sorted(named)} but the extractor emits "
            f"{sorted(expected_vitals)}. Missing: {sorted(expected_vitals - named)}; "
            f"unexpected: {sorted(named - expected_vitals)}."
        )
    return labs.set_index("itemid"), vitals.set_index("slug")


def harmonise_lab_units(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert laboratory results to standard units where possible."""
    counts = {"n_unit_rows_converted": 0, "n_unit_rows_unmapped": 0}
    if "valueuom" not in events.columns or not len(events):
        return events, counts
    for itemid, mapping in LAB_UNIT_HARMONISATION.items():
        rows = events["itemid"] == itemid
        if not rows.any():
            continue
        canonical = {target for target, _ in mapping.values()}
        known = set(mapping) | canonical
        for source_unit, (target_unit, factor) in mapping.items():
            convert = rows & (events["valueuom"] == source_unit)
            n_convert = int(convert.sum())
            if not n_convert:
                continue
            if factor != 1.0:
                events.loc[convert, "value"] = (
                    pd.to_numeric(events.loc[convert, "value"], errors="coerce") * factor
                )
            events.loc[convert, "valueuom"] = target_unit
            counts["n_unit_rows_converted"] += n_convert
        unmapped = rows & ~events["valueuom"].isin(known) & events["valueuom"].notna()
        counts["n_unit_rows_unmapped"] += int(unmapped.sum())
    return events, counts


def apply_thresholds(events: pd.DataFrame, bounds: pd.DataFrame, key: str,
                     label: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mark values outside their allowed range as missing."""
    empty = {f"n_{label}_nulled_below": 0, f"n_{label}_nulled_above": 0,
             f"n_{label}_nulled_total": 0}
    if not len(events):
        return events, empty
    keys = events[key]
    lower = keys.map(bounds["lower"]).astype("float64")
    upper = keys.map(bounds["upper"]).astype("float64")
    values = pd.to_numeric(events["value"], errors="coerce")
    # Missing limits and missing measurements are not treated as violations
    below = values.lt(lower) & lower.notna()
    above = values.gt(upper) & upper.notna()
    violating = below | above
    counts = {f"n_{label}_nulled_below": int(below.sum()),
              f"n_{label}_nulled_above": int(above.sum()),
              f"n_{label}_nulled_total": int(violating.sum())}
    if counts[f"n_{label}_nulled_total"]:
        events.loc[violating[violating].index, "value"] = np.nan
    return events, counts


def stream_window_events(
    cohort: pd.DataFrame, test_mode: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Read laboratory results for both observation periods."""

    labels = pd.read_csv(HOSP_DIR / "d_labitems.csv", usecols=["itemid", "label", "fluid"])
    lab_bounds, _ = load_event_items()
    panel_itemids = set(lab_bounds.index)
    unresolved = sorted(panel_itemids.difference(labels["itemid"]))
    if unresolved:
        raise ValueError(
            f"Lab panel itemids absent from {HOSP_DIR / 'd_labitems.csv'}: {unresolved}. The panel "
            "and the MIMIC-IV release have diverged; re-curate config/event_items.csv."
        )

    windows = pl.from_pandas(
        cohort[["subject_id", "prev_start", "prev_end", "curr_start", "curr_end"]]
        .dropna(subset=["curr_start", "curr_end"])
    )
    scan = pl.scan_csv(LABEVENTS_PATH, try_parse_dates=True).select(
        ["subject_id", "charttime", "itemid", "valuenum", "valueuom"])
    if test_mode:
        scan = scan.head(TEST_ROW_LIMIT)

    joined = (
        scan.filter(pl.col("valuenum").is_not_null())
        .filter(pl.col("itemid").is_in(sorted(panel_itemids)))
        .join(windows.lazy(), on="subject_id", how="inner")
        .with_columns(
            # Patients without a previous stay are outside the previous window
            in_prev=pl.col("charttime")
            .is_between("prev_start", "prev_end", closed="both")
            .fill_null(False),
            in_curr=pl.col("charttime")
            .is_between("curr_start", "curr_end", closed="left")
            .fill_null(False),
        )
        .filter(pl.col("in_prev") | pl.col("in_curr"))
        .select(["subject_id", "charttime", "itemid", "valuenum", "valueuom", "in_prev", "in_curr"])
        .collect()
    )
    n_polars = joined.height
    frame = joined.to_pandas()
    assert len(frame) == n_polars, f"boundary row-count mismatch: {len(frame)} != {n_polars}"
    assert frame["valuenum"].notna().all(), "null valuenum survived the Polars filter"

    frame = frame.merge(labels, on="itemid", how="left")
    frame = frame.rename(columns={"valuenum": "value"})
    frame["linksto"] = LABS_SOURCE
    # Stop if a laboratory test has no known name
    unresolved = frame["label"].isna()
    if unresolved.any():
        missing_itemids = sorted(frame.loc[unresolved, "itemid"].unique())[:10]
        raise ValueError(
            f"{int(unresolved.sum()):,} labevents rows carry itemids absent from d_labitems.csv "
            f"(first: {missing_itemids}). The lab dictionary and labevents must come from the same "
            "MIMIC-IV release."
        )
    frame, unit_counts = harmonise_lab_units(frame)
    print(
        f"        unit harmonisation: converted {unit_counts['n_unit_rows_converted']:,} rows "
        f"({unit_counts['n_unit_rows_unmapped']:,} unmapped units left unchanged)"
    )
    frame, threshold_counts = apply_thresholds(frame, lab_bounds, key="itemid", label="lab")
    threshold_counts = {**threshold_counts, **unit_counts}
    print(
        f"        lab bounds: nulled {threshold_counts['n_lab_nulled_total']:,} of "
        f"{len(frame):,} values "
        f"({threshold_counts['n_lab_nulled_below']:,} below, "
        f"{threshold_counts['n_lab_nulled_above']:,} above)"
    )
    prev = frame.loc[frame["in_prev"], list(EVENT_COLUMNS)]
    curr = frame.loc[frame["in_curr"], list(EVENT_COLUMNS)]
    return prev, curr, threshold_counts


def fahrenheit_to_celsius(f):
    """Convert temperature values from Fahrenheit to Celsius."""
    return (f - 32) * 5.0/9.0


def load_ed_vitals(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Read emergency department vital signs for both observation periods."""
    vitals = pd.read_csv(
        ED_DIR / "vitalsign.csv", usecols=["subject_id", "stay_id", "charttime", *ED_VITALS],
        parse_dates=["charttime"],
    )
    long = vitals.melt(
        id_vars=["subject_id", "charttime"], value_vars=list(ED_VITALS), var_name="label",
        value_name="value",
    ).dropna(subset=["value"])
    long["linksto"] = VITALS_SOURCE
    # Vital sign names identify the measurements because they have no test IDs
    long["itemid"] = pd.NA
    # Vital signs are measured on the patient, not from a specimen
    long["fluid"] = "Vital Sign"
    long["valueuom"] = long["label"].map(ED_VITALS_UNITS)
    if long["valueuom"].isna().any():
        unmapped = sorted(long.loc[long["valueuom"].isna(), "label"].unique())
        raise ValueError(f"ED vitals with no unit in ED_VITALS_UNITS: {unmapped}.")

    # Apply limits before selecting the observation window
    _, vitals_bounds = load_event_items()
    long, threshold_counts = apply_thresholds(long, vitals_bounds, key="label", label="vitals")
    print(
        f"        vitals bounds: nulled {threshold_counts['n_vitals_nulled_total']:,} of "
        f"{len(long):,} values "
        f"({threshold_counts['n_vitals_nulled_below']:,} below, "
        f"{threshold_counts['n_vitals_nulled_above']:,} above)"
    )

    # Convert Fahrenheit to Celsius
    is_temp = long["label"] == "temperature"
    long.loc[is_temp, "value"] = long.loc[is_temp, "value"].apply(fahrenheit_to_celsius)
    long.loc[is_temp, "valueuom"] = ED_VITALS_TEMPERATURE_UNIT

    window_bounds = cohort[["subject_id", "prev_start", "prev_end", "curr_start", "curr_end"]]
    merged = long.merge(window_bounds, on="subject_id", how="inner")
    in_prev = merged["charttime"].between(merged["prev_start"], merged["prev_end"])
    in_curr = (merged["charttime"] >= merged["curr_start"]) & (
        merged["charttime"] < merged["curr_end"])
    return (
        merged.loc[in_prev, list(EVENT_COLUMNS)],
        merged.loc[in_curr, list(EVENT_COLUMNS)],
        threshold_counts,
    )


def build_ts_feature_columns(labs: pd.DataFrame, vitals: pd.DataFrame) -> list[str]:
    """List the measurement columns in their output order."""
    columns = [f"{TS_SOURCE_PREFIXES[VITALS_SOURCE]}{slug}" for slug in ED_VITALS]
    columns += [f"{TS_SOURCE_PREFIXES[LABS_SOURCE]}{slug}" for slug in labs["slug"]]
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            f"Feature column names are not unique: {duplicates}. A vitals slug collides with a lab "
            f"slug in {EVENT_ITEMS_PATH}; the two share one column namespace once prefixed."
        )
    missing_vitals = sorted(set(ED_VITALS).difference(vitals.index))
    if missing_vitals:
        raise ValueError(f"ED vitals absent from the panel: {missing_vitals}.")
    return columns


def assign_feature_column(events: pd.DataFrame, labs: pd.DataFrame) -> pd.Series:
    """Choose the output column for one measurement."""
    slug_by_itemid = {int(itemid): slug for itemid, slug in labs["slug"].items()}
    is_lab = events["linksto"].eq(LABS_SOURCE)
    slug = pd.Series(pd.NA, index=events.index, dtype="object")
    slug[is_lab] = (
        pd.to_numeric(events.loc[is_lab, "itemid"], errors="coerce")
        .astype("Int64")
        .map(slug_by_itemid)
    )
    slug[~is_lab] = events.loc[~is_lab, "label"].astype("object")
    unresolved = slug.isna()
    if unresolved.any():
        sample = events.loc[unresolved, ["itemid", "label", "linksto"]].head(5).to_dict("records")
        raise ValueError(
            f"{int(unresolved.sum()):,} event rows resolve to no panel slug (first: {sample}). "
            f"The events and {EVENT_ITEMS_PATH} have diverged."
        )
    return events["linksto"].map(TS_SOURCE_PREFIXES) + slug


def build_wide_events(
    events: pd.DataFrame, cohort: pd.DataFrame, window: str, feature_columns: list[str],
    labs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Turn measurements from one period into a wide table."""
    if window not in WINDOW_SPECS:
        raise ValueError(f"window must be one of {sorted(WINDOW_SPECS)}; got {window!r}.")
    spec = WINDOW_SPECS[window]
    events = events.copy()
    events["feature"] = assign_feature_column(events, labs)

    values = (
        events.sort_values(["subject_id", "charttime"], kind="stable")
        .groupby(["subject_id", "charttime", "feature"], sort=False, observed=True)["value"]
        .first()
    )
    wide = values.unstack("feature").reindex(columns=feature_columns).reset_index()

    metadata = cohort[["subject_id", spec["hadm_id"], spec["window_start"], spec["window_end"]]]
    wide = wide.merge(metadata, on="subject_id", how="left")
    # Record elapsed time in minutes for each window
    wide[spec["elapsed"]] = (
        wide["charttime"] - wide[spec["window_start"]]
    ) / pd.Timedelta(minutes=1)
    wide[spec["hadm_id"]] = wide[spec["hadm_id"]].astype("Int64")

    ordered = [
        "subject_id", spec["hadm_id"], spec["window_start"], spec["window_end"], "charttime",
        spec["elapsed"], *feature_columns,
    ]
    wide = wide[ordered].sort_values(
        ["subject_id", spec["elapsed"]], kind="stable"
    ).reset_index(drop=True)

    measured = wide[feature_columns].notna()
    counts = {
        "n_rows": len(wide),
        "n_patients": int(wide["subject_id"].nunique()),
        "n_values": int(measured.to_numpy().sum()),
        "n_all_null_timepoints": int((~measured).all(axis=1).sum()),
        "n_features_never_measured": int((~measured.any()).sum()),
        "density": round(float(measured.to_numpy().mean()), 4) if len(wide) else 0.0,
    }
    return wide, counts


def check_feature_units(
    events: pd.DataFrame, feature_columns: list[str], labs: pd.DataFrame
) -> dict[str, int]:
    """Check that each output measurement uses one unit and count its values."""
    events = events.copy()
    events["feature"] = assign_feature_column(events, labs)
    units = events.dropna(subset=["valueuom"]).groupby("feature")["valueuom"].unique()
    conflicting = {
        feature: sorted(found) for feature, found in units.items() if len(found) > 1
    }
    if conflicting:
        raise ValueError(
            f"Feature columns carry more than one unit: {conflicting}. A wide column is one unit by "
            "construction, so harmonise them in LAB_UNIT_HARMONISATION rather than emitting a "
            "column whose rows are not comparable."
        )
    counts = events.groupby("feature").size()
    return {column: int(counts.get(column, 0)) for column in feature_columns}


def main() -> None:
    args = parse_args()
    # Create the output directory before writing files
    output_dir = args.output_dir / HISTORY_SUBDIRS[args.history]
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_counts: dict[str, int] = {}

    print("[1/8] Reading admissions and patients; validating timestamps; adults only.")
    admissions = read_admissions(nrows=TEST_ROW_LIMIT if args.test_mode else None)
    stage_counts["adult_ed_flagged"] = len(admissions)

    print("[2/8] Requiring an edstays join and complete demographics; selecting last + previous.")
    admissions = require_ed_linkage(admissions)
    stage_counts["ed_linked"] = len(admissions)
    admissions, invalid = require_complete_demographics(admissions)
    stage_counts["complete_demographics"] = len(admissions)
    for field, count in invalid.items():
        print(f"        invalid {field}: {count:,}")

    cohort = select_last_and_previous(admissions)
    cohort = add_window_bounds(cohort)
    cohort = apply_lookback_band(cohort, args.lookback_days)
    stage_counts[COHORT_SIZE_STAGES["optional"]] = len(cohort)
    stage_counts["has_prev_any"] = int(cohort["has_prev_any"].sum())
    stage_counts[COHORT_SIZE_STAGES["required"]] = int(cohort["has_prev_band"].sum())
    if args.history == "required":
        # Keep only patients with a previous admission for this cohort
        cohort = cohort[cohort["has_prev_band"]].copy()
        stage_counts["extracted"] = len(cohort)
        print(
            f"        --history required: narrowed to {len(cohort):,} band-eligible patients. "
            "This extraction cannot produce the optional-history superset; re-run with "
            "--history optional for it."
        )

    print("[3/8] Writing cohort keys and reporting cohort sizes.")
    cohort.to_csv(output_dir / "cohort_keys.csv", index=False)
    for stage, count in stage_counts.items():
        print(f"        {stage}: {count:,}")

    print("[4/8] Building the static block and the three index-stay labels.")
    static = build_static_block(cohort, admissions)
    static.to_csv(output_dir / "static.csv", index=False)
    top_medications = [
        column[len("medications_"):-len("_n")]
        for column in static.columns
        if column.startswith("medications_") and column.endswith("_n")
        and column != STATIC_RENAMES["total_n_presc"]
    ]
    label_prevalence = {
        label: {"n_positive": int(static[label].sum()),
                "rate": round(float(static[label].mean()), 4)}
        for label in STATIC_OUTCOME_LABELS
    }
    for label, counts in label_prevalence.items():
        print(f"        {label}: {counts['n_positive']:,} of {len(static):,} "
              f"({counts['rate'] * 100:.2f}%)")

    summary = {
        "history": args.history, "stage_counts": stage_counts, "invalid_demographics": invalid,
        "lookback_days": args.lookback_days, "current_window": "admittime",
        "static_columns": len(static.columns), "label_columns": list(STATIC_OUTCOME_LABELS),
        "top_medications": top_medications,
        "label_prevalence": label_prevalence, "event_items_path": str(EVENT_ITEMS_PATH),
        "lab_panel_itemids": EXPECTED_LAB_PANEL_SIZE, "test_mode": args.test_mode,
    }
    if args.skip_events:
        summary["events"] = "skipped (--skip-events)"
        (output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=1))
        print("[--] --skip-events: cohort and static written; stopping before the labevents pass.")
        return

    print(
        f"[5/8] Streaming labevents once for both windows, restricted to the "
        f"{EXPECTED_LAB_PANEL_SIZE}-itemid panel (this is the slow stage)."
    )
    lab_prev, lab_curr, threshold_counts = stream_window_events(cohort, test_mode=args.test_mode)

    print("[6/8] Appending ED vitals (both windows).")
    vit_prev, vit_curr, vitals_threshold_counts = load_ed_vitals(cohort)
    summary["thresholds"] = {**threshold_counts, **vitals_threshold_counts}
    long_prev = pd.concat([lab_prev, vit_prev], ignore_index=True)
    long_curr = pd.concat([lab_curr, vit_curr], ignore_index=True)

    print("[7/8] Pivoting both windows to the wide layout.")
    labs, vitals = load_event_items()
    feature_columns = build_ts_feature_columns(labs, vitals)
    event_counts = {}
    for window, long_events in (("prev", long_prev), ("curr", long_curr)):
        wide, counts = build_wide_events(long_events, cohort, window, feature_columns, labs)
        wide.to_csv(output_dir / WINDOW_SPECS[window]["filename"], index=False)
        event_counts[window] = counts
        print(
            f"        {WINDOW_SPECS[window]['filename']}: {counts['n_rows']:,} rows x "
            f"{len(feature_columns)} features | {counts['n_patients']:,} patients | "
            f"{counts['n_values']:,} values | density {counts['density'] * 100:.1f}% | "
            f"{counts['n_all_null_timepoints']:,} all-null timepoints | "
            f"{counts['n_features_never_measured']} features never measured"
        )
    # Check values from both observation windows
    values_per_feature = check_feature_units(
        pd.concat([long_prev, long_curr], ignore_index=True), feature_columns, labs
    )

    print("[8/8] Writing the extraction summary.")
    summary["events"] = {
        "layout": "wide (one row per subject_id x charttime, one column per marker)",
        "feature_columns": len(feature_columns),
        "meta_columns": {window: [
            spec["hadm_id"], spec["window_start"], spec["window_end"], "charttime", spec["elapsed"]
        ] for window, spec in WINDOW_SPECS.items()},
        "duplicate_rule": "first non-null reading per (subject_id, charttime, feature)",
        "values_per_feature": values_per_feature,
        **{window: counts for window, counts in event_counts.items()},
    }
    (output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
