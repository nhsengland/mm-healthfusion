"""Build baseline characteristics tables for the deterioration and hospitalisation cohorts.

Usage::

    python src/exploration/create_tableone.py
    python src/exploration/create_tableone.py --task hospitalisation --output-dir /data/outputs/tableone
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from tableone import TableOne

OUTPUTS_DIR = Path("/data/outputs")
DEFAULT_BUNDLE_DIRS = {
    "deterioration": OUTPUTS_DIR
    / "deterioration"
    / "3.1_complete"
    / "deterioration_allStay_averageRecord_allCohort",
    "hospitalisation": OUTPUTS_DIR
    / "hospitalisation"
    / "3.1"
    / "hospitalisation_dualWindow_averageRecord_optionalHistory",
}
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "tableone"
# Split directory inside each bundle, named after the bundle plus this suffix
SPLIT_SUFFIX = "normalised_ohe"

# One table row per identifier value
ID_COLUMNS = {"deterioration": "ed_stay_id", "hospitalisation": "subject_id"}
UNITS = {"deterioration": "ED stays", "hospitalisation": "patients"}

# Outcome columns and their column headings, in table order
OUTCOMES = {
    "deterioration": (
        ("deterioration_mechanical_ventilation", "MV"),
        ("deterioration_cardiac_arrest", "CA"),
        ("deterioration_icu_24h", "ICU-24h"),
        ("deterioration_icu_stay", "ICU"),
        ("deterioration_mortality_1d", "MORT-1D"),
    ),
    "hospitalisation": (
        ("hospitalisation_ext_stay_7", "EXT-ST"),
        ("hospitalisation_in_hosp_death", "IHD"),
        ("hospitalisation_non_home_discharge", "NHD"),
    ),
}

ETHNICITY_PREFIX = "demographics_ethnicity_"
ETHNICITY_LEVELS = ("Asian", "Black", "Hispanic/Latino", "White", "Other")
# Deterioration stores ethnicity as one indicator column per group
DETERIORATION_ETHNICITY = {
    "asian": "Asian",
    "black/african": "Black",
    "hispanic/latino": "Hispanic/Latino",
    "white": "White",
    "other": "Other",
}
MARITAL_LEVELS = ("Divorced", "Married", "Single", "Widowed")
INSURANCE_LEVELS = ("Medicaid", "Medicare", "Private", "Other")
# Too few patients to report as a separate insurance group
INSURANCE_MERGED = {"No charge": "Other"}

NOT_RECORDED = "Missing"
# Triage acuity is shown in its original Emergency Severity Index (ESI) labels, with levels 4 and 5
# as one group. Both bundles store it minus one, and hospitalisation merges 4 and 5 at extraction
ACUITY_LABELS = {
    "deterioration": {0: "1", 1: "2", 2: "3", 3: "4 or 5", 4: "4 or 5"},
    "hospitalisation": {0: "1", 1: "2", 2: "3", 3: "4 or 5"},
}

# Raw columns read from each bundle file
TABULAR_COLUMNS = {
    "deterioration": ("subject_id", "demographics_age", "demographics_gender", "biometrics_bmi",
                      "vitals_acuity"),
    "hospitalisation": ("demographics_age", "demographics_gender", "demographics_ethnicity",
                        "demographics_marital_status", "demographics_insurance", "biometrics_bmi",
                        "vitals_acuity", "conditions_is_multimorbid",
                        "conditions_is_complex_multimorbid", "conditions_phys_men_multimorbidity",
                        "history_n_prior_admissions", "encounter_ed_phase_hours"),
}
KEY_COLUMNS = {
    "deterioration": ("has_ecg_embedding", "has_cxr_embedding", "has_ts"),
    "hospitalisation": ("has_prev_band", "has_ts_curr", "has_ecg_curr", "has_cxr_curr",
                        "has_ts_prev", "has_notes_prev", "has_cxr_prev"),
}

GROUP = "group"
OVERALL = "All"
FLAG_LEVELS = ("Yes", "No")

# tableone appends the summary statistic to each row label
STATISTICS = {"continuous": "median [Q1,Q3]", "categorical": "n (%)", "flag": "n (%)"}
# A tableone count cell, such as "1234 (5.6)"
COUNT_CELL = re.compile(r"(\d+) \(([\d.]+)\)")


class Variable(NamedTuple):
    """One table variable.

    ``kind`` is continuous (median [Q1,Q3]), categorical (n (%) per level) or flag (n (%) of the
    Yes level only). ``section`` and ``item`` nest the variable under a shared heading in the
    LaTeX table.
    """

    column: str
    label: str
    kind: str
    section: str | None = None
    item: str | None = None


MODALITY = "Modality availability, n (%)"
COMORBIDITY = "Comorbidity history, n (%)"
PREVIOUS_ADMISSION = "Previous admission, n (%)"
CURRENT_WINDOW = "Current window"
PREVIOUS_WINDOW = "Previous window, n (%)"

SHARED_VARIABLES = (
    Variable("age", "Age", "continuous"),
    Variable("women", "Women", "flag"),
    Variable("ethnicity", "Ethnicity", "categorical"),
)
BMI_VARIABLES = (
    Variable("bmi_recorded", "BMI recorded", "flag"),
    Variable("bmi", "BMI", "continuous"),
)
VARIABLES = {
    "deterioration": (
        *SHARED_VARIABLES,
        Variable("acuity", "Triage acuity", "categorical"),
        *BMI_VARIABLES,
        Variable("has_ecg_embedding", "ECG available", "flag", MODALITY, "ECG"),
        Variable("has_cxr_embedding", "Chest X-ray available", "flag", MODALITY, "Chest X-ray"),
        Variable("has_ts", "Vital signs or lab tests available", "flag", MODALITY,
                 "Vital signs or lab tests"),
    ),
    "hospitalisation": (
        *SHARED_VARIABLES,
        Variable("marital_status", "Marital status", "categorical"),
        Variable("insurance", "Insurance", "categorical"),
        Variable("acuity", "Triage acuity", "categorical"),
        *BMI_VARIABLES,
        Variable("conditions_is_multimorbid", "≥2 conditions", "flag", COMORBIDITY, "≥2 conditions"),
        Variable("conditions_is_complex_multimorbid", "≥4 conditions", "flag", COMORBIDITY,
                 "≥4 conditions"),
        Variable("conditions_phys_men_multimorbidity", "≥1 physical and ≥1 mental condition", "flag",
                 COMORBIDITY, "≥1 physical and ≥1 mental"),
        Variable("any_previous_admission", "Any previous admission", "flag", PREVIOUS_ADMISSION, "Any"),
        Variable("has_prev_band", "Previous admission within the 90-day lookback band", "flag",
                 PREVIOUS_ADMISSION, "Within the 90-day lookback band"),
        Variable("encounter_ed_phase_hours", "Current window length (hours)", "continuous",
                 CURRENT_WINDOW, "Length in hours, median [Q1,Q3]"),
        Variable("has_ts_curr", "Current window: vital signs or lab tests", "flag", CURRENT_WINDOW,
                 "Vital signs or lab tests, n (%)"),
        Variable("has_ecg_curr", "Current window: ECG", "flag", CURRENT_WINDOW, "ECG, n (%)"),
        Variable("has_cxr_curr", "Current window: chest X-ray", "flag", CURRENT_WINDOW,
                 "Chest X-ray, n (%)"),
        Variable("has_ts_prev", "Previous window: vital signs or lab tests", "flag", PREVIOUS_WINDOW,
                 "Vital signs or lab tests"),
        Variable("has_notes_prev", "Previous window: discharge note", "flag", PREVIOUS_WINDOW,
                 "Discharge note"),
        Variable("has_cxr_prev", "Previous window: chest X-ray", "flag", PREVIOUS_WINDOW, "Chest X-ray"),
    ),
}
# Decimal places for continuous rows. Percentages keep the tableone default of one
DECIMALS = {"age": 0, "bmi": 1, "encounter_ed_phase_hours": 1}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--task",
        choices=("deterioration", "hospitalisation", "both"),
        default="both",
        help="Cohort to tabulate. Default: both.",
    )
    parser.add_argument(
        "--deterioration-bundle",
        type=Path,
        default=DEFAULT_BUNDLE_DIRS["deterioration"],
        help="Deterioration bundle directory written by prepare_deterioration.py.",
    )
    parser.add_argument(
        "--hospitalisation-bundle",
        type=Path,
        default=DEFAULT_BUNDLE_DIRS["hospitalisation"],
        help="Hospitalisation bundle directory written by prepare_hospitalisation.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def read_split_ids(split_dir: Path, id_column: str) -> pd.Series:
    """Read the identifier of every row that reaches a split."""
    paths = sorted(split_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No split parquet files found in {split_dir}")
    ids = pd.concat(
        [pd.read_parquet(path, columns=[id_column])[id_column] for path in paths],
        ignore_index=True,
    )
    if ids.duplicated().any():
        raise ValueError(f"{split_dir} holds the same {id_column} in more than one row or split.")
    return ids


def load_bundle(task: str, bundle_dir: Path) -> pd.DataFrame:
    """Join the raw tabular and key columns the table needs, one row per identifier."""
    id_column = ID_COLUMNS[task]
    wanted = {id_column, *TABULAR_COLUMNS[task], *(column for column, _ in OUTCOMES[task])}
    tabular = pd.read_csv(
        bundle_dir / f"{bundle_dir.name}_tabular.csv",
        usecols=lambda name: name in wanted
        or (task == "deterioration" and name.startswith(ETHNICITY_PREFIX)),
        low_memory=False,
    )
    missing = sorted(wanted - set(tabular.columns))
    if missing:
        raise ValueError(f"{bundle_dir.name}_tabular.csv is missing columns: {missing}")
    keys = pd.read_csv(
        bundle_dir / f"{bundle_dir.name}_keys.csv",
        usecols=[id_column, *KEY_COLUMNS[task]],
        low_memory=False,
    )
    return tabular.merge(keys, on=id_column, validate="one_to_one")


def _ordered(values: pd.Series, levels: tuple[str, ...], name: str) -> pd.Series:
    """Fix the display order of a categorical variable and reject unknown values."""
    unexpected = sorted(set(values.dropna()) - set(levels))
    if unexpected or values.isna().any():
        raise ValueError(f"Unexpected or missing {name} values: {unexpected or 'NaN'}")
    return pd.Series(pd.Categorical(values, categories=list(levels)), index=values.index)


def _acuity(values: pd.Series, task: str) -> pd.Series:
    """Restore the original ESI labels and keep missing acuity as its own category."""
    labels = ACUITY_LABELS[task]
    unexpected = sorted(set(values.dropna()) - set(labels))
    if unexpected:
        raise ValueError(f"Unexpected vitals_acuity values for {task}: {unexpected}")
    levels = (*dict.fromkeys(labels.values()), NOT_RECORDED)
    return _ordered(values.map(labels).fillna(NOT_RECORDED), levels, "acuity")


def prepare_frame(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the display columns named in ``VARIABLES`` from the raw bundle columns."""
    out = frame.copy()
    out["age"] = out["demographics_age"]
    # Gender is stored as 1 for male and 0 for female
    out["women"] = out["demographics_gender"].eq(0)
    out["bmi"] = out["biometrics_bmi"]
    out["bmi_recorded"] = out["bmi"].notna()
    out["acuity"] = _acuity(out["vitals_acuity"], task)

    if task == "deterioration":
        indicators = out[[c for c in out.columns if c.startswith(ETHNICITY_PREFIX)]]
        if not indicators.sum(axis=1).eq(1).all():
            raise ValueError("Each deterioration stay must have exactly one ethnicity indicator set.")
        ethnicity = indicators.idxmax(axis=1).str.removeprefix(ETHNICITY_PREFIX)
        out["ethnicity"] = _ordered(ethnicity.map(DETERIORATION_ETHNICITY), ETHNICITY_LEVELS, "ethnicity")
        return out

    out["ethnicity"] = _ordered(out["demographics_ethnicity"], ETHNICITY_LEVELS, "ethnicity")
    out["marital_status"] = _ordered(out["demographics_marital_status"], MARITAL_LEVELS, "marital status")
    out["insurance"] = _ordered(
        out["demographics_insurance"].replace(INSURANCE_MERGED), INSURANCE_LEVELS, "insurance"
    )
    out["any_previous_admission"] = out["history_n_prior_admissions"].gt(0)
    return out


def _format_cell(cell: str) -> str:
    """Add thousands separators to counts and a percent sign to percentages."""
    count = COUNT_CELL.fullmatch(cell)
    if count:
        return f"{int(count[1]):,} ({count[2]}%)"
    return f"{int(cell):,}" if cell.isdigit() else cell


def build_table(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise every variable for the whole cohort and for each outcome-positive group."""
    # Stack one copy of the cohort per column so tableone summarises overlapping groups at once
    groups = [frame.assign(**{GROUP: OVERALL})]
    for column, heading in OUTCOMES[task]:
        # Missing labels count in neither the positive nor the negative group
        groups.append(frame[frame[column].eq(1)].assign(**{GROUP: heading}))
    stacked = pd.concat(groups, ignore_index=True)

    # tableone leaves out a group with no rows, so only observed groups are ordered
    order = {GROUP: list(stacked[GROUP].unique())}
    limit = {}
    for column, kind in ((variable.column, variable.kind) for variable in VARIABLES[task]):
        if kind == "flag":
            stacked[column] = np.where(stacked[column].astype(bool), *FLAG_LEVELS)
            order[column] = [level for level in FLAG_LEVELS if stacked[column].eq(level).any()]
            limit[column] = 1
        elif kind == "categorical":
            observed = set(stacked[column])
            order[column] = [level for level in stacked[column].cat.categories if level in observed]
            stacked[column] = stacked[column].astype(str)

    table = TableOne(
        stacked,
        columns=[variable.column for variable in VARIABLES[task]],
        categorical=[variable.column for variable in VARIABLES[task] if variable.kind != "continuous"],
        nonnormal=[variable.column for variable in VARIABLES[task] if variable.kind == "continuous"],
        groupby=GROUP,
        overall=False,
        missing=False,
        pval=False,
        order=order,
        limit=limit,
        decimals=DECIMALS,
        rename={variable.column: variable.label for variable in VARIABLES[task]},
    ).tableone
    table.columns = table.columns.get_level_values(-1)
    table.index.names = ["Variable", "Level"]
    # The group sizes come back as integers and every other cell as text
    return table.astype(str).map(_format_cell)


def nest_rows(task: str, table: pd.DataFrame) -> list[tuple[str, bool, list[str]]]:
    """Lay the table out with levels and section items nested under bold headings.

    Each entry is the row label, whether it is a heading, and one cell per column.
    """
    blank = [""] * table.shape[1]
    rows: list[tuple[str, bool, list[str]]] = []
    section = None
    for variable in VARIABLES[task]:
        name = f"{variable.label}, {STATISTICS[variable.kind]}"
        levels = table.xs(name, level="Variable")
        if variable.kind == "flag":
            # tableone drops the Yes level when no row has it
            zero = _format_cell("0 (0.0)")
            cells = list(levels.loc["Yes"]) if "Yes" in levels.index else [zero] * table.shape[1]
        elif variable.kind == "continuous":
            cells = list(levels.iloc[0])

        if variable.section is not None:
            if variable.section != section:
                rows.append((variable.section, True, blank))
            rows.append((variable.item, False, cells))
        elif variable.kind == "categorical":
            rows.append((name, True, blank))
            rows.extend((level, False, list(values)) for level, values in levels.iterrows())
        else:
            rows.append((name, True, cells))
        section = variable.section
    return rows


def _latex_escape(text: str) -> str:
    for char in "&%#_":
        text = text.replace(char, f"\\{char}")
    return text.replace("≥", "$\\geq$")


def to_latex(task: str, table: pd.DataFrame) -> str:
    """Render the nested table as a LaTeX tabular with group sizes in the header."""
    sizes = table.loc[("n", "")]
    header = " & ".join(["\\textbf{Variable}", *(f"\\textbf{{{_latex_escape(c)}}}" for c in table.columns)])
    lines = [
        f"\\begin{{tabular}}{{l{'c' * table.shape[1]}}}",
        "\\toprule",
        f"{header} \\\\",
        " & ".join(["", *(f"(n={sizes[column]})" for column in table.columns)]) + " \\\\",
        "\\midrule",
    ]
    for label, heading, cells in nest_rows(task, table):
        text = _latex_escape(label)
        text = f"\\textbf{{{text}}}" if heading else f"\\quad {text}"
        lines.append(" & ".join([text, *map(_latex_escape, cells)]) + " \\\\")
    return "\n".join([*lines, "\\bottomrule", "\\end{tabular}", ""])


def write_table(task: str, table: pd.DataFrame, output_stem: Path) -> list[Path]:
    """Write the table as CSV and as a nested LaTeX tabular."""
    csv_path = output_stem.with_suffix(".csv")
    table.to_csv(csv_path)
    tex_path = output_stem.with_suffix(".tex")
    tex_path.write_text(to_latex(task, table))
    return [csv_path, tex_path]


def run_task(task: str, bundle_dir: Path, output_dir: Path) -> dict:
    """Build one cohort table and record which bundle rows it describes."""
    id_column = ID_COLUMNS[task]
    split_dir = bundle_dir / f"{bundle_dir.name}_{SPLIT_SUFFIX}"
    bundle = load_bundle(task, bundle_dir)
    split_ids = read_split_ids(split_dir, id_column)
    unknown = int((~split_ids.isin(bundle[id_column])).sum())
    if unknown:
        raise ValueError(f"{unknown} split rows in {split_dir} are not in the bundle.")

    frame = prepare_frame(task, bundle[bundle[id_column].isin(split_ids)].reset_index(drop=True))
    table = build_table(task, frame)
    paths = write_table(task, table, output_dir / f"cohort_table_{task}")

    summary = {
        "task": task,
        "unit": UNITS[task],
        "bundle_dir": str(bundle_dir),
        "split_dir": str(split_dir),
        "n_bundle_rows": len(bundle),
        "n_rows_in_split": len(frame),
        "n_rows_not_in_split": len(bundle) - len(frame),
        "n_patients": int(frame["subject_id"].nunique()),
        "outcomes": {
            heading: {
                "column": column,
                "n_positive": int(frame[column].eq(1).sum()),
                "n_positive_patients": int(frame.loc[frame[column].eq(1), "subject_id"].nunique()),
                "n_missing_label": int(frame[column].isna().sum()),
            }
            for column, heading in OUTCOMES[task]
        },
        "output_files": [str(path) for path in paths],
    }
    summary_path = output_dir / f"cohort_table_{task}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"\n{task}: {summary['n_rows_in_split']:,} of {summary['n_bundle_rows']:,} bundle "
        f"{UNITS[task]} reach a split ({summary['n_rows_not_in_split']:,} excluded)."
    )
    print(table.to_string())
    print(f"Wrote {', '.join(path.name for path in [*paths, summary_path])} to {output_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        "deterioration": args.deterioration_bundle,
        "hospitalisation": args.hospitalisation_bundle,
    }
    tasks = tuple(bundles) if args.task == "both" else (args.task,)
    for task in tasks:
        run_task(task, bundles[task], args.output_dir)


if __name__ == "__main__":
    main()
