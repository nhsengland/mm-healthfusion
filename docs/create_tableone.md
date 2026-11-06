# Cohort Tables

`create_tableone.py` writes a baseline characteristics table (a "Table 1") for the deterioration
cohort and for the hospitalisation cohort. Each table has an overall column and one column per
outcome-positive group, and is built with the [`tableone`](https://github.com/tompollard/tableone)
package.

Example:

```sh
python src/exploration/create_tableone.py
python src/exploration/create_tableone.py --task hospitalisation --output-dir /data/outputs/tableone
```

## Which rows are described

The script reads the raw `_tabular.csv` and `_keys.csv` of a bundle, not the split parquet files.
The parquet files hold imputed, scaled and one-hot encoded values, which are not suitable for
describing a cohort.

A bundle can hold rows that no split keeps. The deterioration splitter keeps only the earliest
stay of each validation and test patient, so later stays of those patients stay in the bundle but
reach no model. The table therefore keeps only identifiers found in the split parquet files, and
the summary JSON records how many bundle rows were left out.

The unit of analysis differs between the two tables. Deterioration counts ED stays, so one patient
can contribute several rows. Hospitalisation counts patients.

## Columns and rows

Columns are `All` followed by the outcome-positive groups. A row with a missing outcome label
belongs to `All` only. The groups overlap, so no hypothesis test is run across them.

| Kind | Summary | Variables |
|---|---|---|
| Continuous | median [Q1,Q3] | age, BMI, current window length |
| Categorical | n (%) per level | ethnicity, marital status, insurance, triage acuity |
| Flag | n (%) of the `Yes` level | women, BMI recorded, modality availability, comorbidity, previous admission |

Counts carry thousands separators and percentages carry a percent sign, as in `62,188 (53.4%)`.

Marital status, insurance, comorbidity, previous admission and the two observation windows exist
only in the hospitalisation bundle.

Display rules applied before summarising:

- Triage acuity is presented in its original Emergency Severity Index (ESI) labels, from 1 (most
  urgent) to 5 (least urgent).
  ESI levels 4 and 5 are shown as one level, `4 or 5`, in both tables. Hospitalisation merges them
  at extraction and the deterioration table merges them for display. Missing acuity is shown as
  `Missing`.
- Gender is stored as 1 for male and 0 for female.
- Deterioration ethnicity is rebuilt from its indicator columns.
- The `No charge` insurance group is merged into `Other`.
- An unexpected category raises an error instead of being dropped.

## Main options

- `--task {deterioration,hospitalisation,both}`: cohort to tabulate. Default: `both`.
- `--deterioration-bundle`: deterioration bundle directory. Default:
  `/data/outputs/deterioration/3.1_complete/deterioration_allStay_averageRecord_allCohort`.
- `--hospitalisation-bundle`: hospitalisation bundle directory. Default:
  `/data/outputs/hospitalisation/3.1/hospitalisation_dualWindow_averageRecord_optionalHistory`.
- `--output-dir`: where the tables are written. Default: `/data/outputs/tableone`.

The dual-window hospitalisation bundle is the default because its key table carries availability
flags for both the current and the previous window.

The split is always read from the bundle's `{bundle_name}_normalised_ohe` directory.

## Outputs

For each task the script writes:

| File | Purpose |
|---|---|
| `cohort_table_{task}.csv` | The table as `tableone` lays it out, with a `Variable` and a `Level` column |
| `cohort_table_{task}.tex` | The same values as a nested LaTeX `tabular` (needs `booktabs`) |
| `cohort_table_{task}_summary.json` | Bundle and split paths, rows kept and left out, patients, and per outcome the positive rows, positive patients and missing labels |

The LaTeX table has one label column. Each variable is a bold heading, the levels of a categorical
variable are indented under it, and a flag shows only its `Yes` count. Related flags share one
heading, such as `Modality availability, n (%)`. The size of each group sits under its column
heading instead of in an `n` row.
