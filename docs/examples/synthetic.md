# Synthetic data example

This example generates fictional data for both pathways and runs the full workflow on it without
access to MIMIC: data exploration, model training, evaluation, explanation and fairness evaluation.
The files follow the parquet schema of the real average-record split bundles, the input of the
static fusion models, so the existing datasets, collators and static model configurations read them
unchanged. Follow the [installation steps](../getting_started.md#installation) first and run
commands from the repository root in the activated environment.

**The data is artificial.** Every record, identifier and outcome is fictional and has no clinical
interpretation. The generator does not imitate real clinical distributions. It builds learnable
relationships between a few hidden factors and the outcome, so models train and the formats can be
explored, but results on these files say nothing about clinical performance.

The workflow has two routes. The [demo notebook](https://github.com/xinzhao13/mm-healthfusion/blob/main/notebooks/01%20Synthetic%20Data%20Demo.ipynb) 
runs every step on the deterioration pathway with the static MoE model. The sections after it give
the same steps as commands, for every static model and both pathways.

## Generate the data

```sh
python -m src.preprocessing.generate_synthetic
```

The command writes 1,000 fictional records per pathway. Each pathway is split 80/10/10, stratified
by outcome, into 800 training, 100 validation and 100 test records. Every split holds both outcome
classes and no fictional patient appears in more than one split. The same arguments always produce
identical files. See [synthetic data generation](../api_reference.md#synthetic-data-generation) in
the API reference for every option, including the record count, missing rate and feature widths.

The output folder holds one directory per pathway:

```text
../outputs/synthetic/
├── deterioration/
│   ├── train_synthetic.parquet      # 800 records
│   ├── val_synthetic.parquet        # 100 records
│   ├── test_synthetic.parquet       # 100 records
│   ├── synthetic_manifest.json
│   ├── synthetic_split_summary.json # tabular feature names for explainability
│   └── synthetic_demographics.csv   # fictional demographics for fairness evaluation
└── hospitalisation/
    ├── train_synthetic.parquet      # 800 records
    ├── val_synthetic.parquet        # 100 records
    ├── test_synthetic.parquet       # 100 records
    ├── synthetic_manifest.json
    ├── synthetic_split_summary.json
    └── synthetic_demographics.csv
```

`synthetic_manifest.json` records the generator version and settings, the counts and outcome
prevalence per split, feature widths and names, modality availability, missingness rates, a
description of the synthetic signal and missingness, and the planted demographic biases with
per-group counts, prevalence and modality availability.

---

## What the files contain

Deterioration predicts `deterioration_mechanical_ventilation` from `tabular`, `ecg` and `cxr`.
Hospitalisation predicts `hospitalisation_ext_stay_7` from `static`, `ts_prev`, `ts_curr`,
`ecg_curr`, `cxr_prev`, `cxr_curr` and `notes_prev`.

The columns, types and missingness convention match average-record splits from
`split_deterioration` and `split_hospitalisation`, where 0 means observed and 1 means missing:

- **Embeddings:** a present ECG, CXR or notes modality holds one averaged embedding, 768 wide for
  ECG and notes and 1024 wide for CXR, as from the real extractors. An absent one has an empty array
  and a modality flag of 1.
- **Measurement summaries:** each vital-sign or lab measurement becomes nine statistics (mean,
  median, min, max, std, first, last, rate_change and coeff). They sit inside `tab_feats` in
  deterioration and form the one-row `ts_prev` and `ts_curr` in hospitalisation, which are empty
  when absent.
- **Missing values:** each tabular or summary value has a matching flag that is 1 when the value is
  missing, and the missing value itself is stored as 0. A measurement that was never taken has all
  nine statistics missing. In hospitalisation, a measurement taken only once also has no
  rate_change or coeff, because one reading has no trend.
- **No observation times:** the `*_time` columns are always empty as only static fusion models are supported in synthetic data.

Unlike the real data, each pathway has one outcome column and fewer base features and measurements,
and deterioration files also carry `subject_id` so the patient split can be checked.

---

## Demo notebook: from data to fairness

[`notebooks/01 Synthetic Data Demo.ipynb`](https://github.com/xinzhao13/mm-healthfusion/blob/main/notebooks/01%20Synthetic%20Data%20Demo.ipynb)
runs the full workflow end to end, from the stored records to a trained, evaluated and explained
model:

- **Data:** the parquet records of both pathways, example patients, each modality's format,
  missingness plots, PyTorch samples from `DeteriorationDataset` and `HospitalisationDataset`, and
  collated batches.
- **Model training:** a full run of the static MoE model on the deterioration files with
  `config/smp.yaml`, with its training curves.
- **Model evaluation:** the test predictions and the metrics of the best checkpoint on each split.
- **Model interpretation:** MM-SHAP modality contributions, the tabular feature SHAP heatmap and a
  decision plot for one test patient.
- **Fairness evaluation:** fairness ratios and per-group selection and false negative rates by sex,
  age, ethnicity and CXR availability, compared with the
  [planted biases](#evaluate-fairness).

---

## Train and evaluate

Six static model configurations run on synthetic data:

| Pathway | Model | Configuration |
|---|---|---|
| Deterioration | Concatenation | `config/concat.yaml` |
| Deterioration | M3Care | `config/m3care.yaml` |
| Deterioration | Static MoE model | `config/smp.yaml` |
| Hospitalisation | Concatenation | `config/concat_hospitalisation.yaml` |
| Hospitalisation | M3Care | `config/m3care_hospitalisation.yaml` |
| Hospitalisation | Static MoE model | `config/smp_hospitalisation.yaml` |

Point a configuration at a synthetic pathway with `data.split_dir`. Use `--output-root` to keep
these runs apart from runs on real data:

```sh
# Deterioration, static MoE model
python -m src.train --config config/smp.yaml \
    --set data.split_dir=../outputs/synthetic/deterioration --output-root ../logs/synthetic

# Hospitalisation, static MoE model
python -m src.train --config config/smp_hospitalisation.yaml \
    --set data.split_dir=../outputs/synthetic/hospitalisation --output-root ../logs/synthetic
```

FuseMoE and mTAND-M3Care are not supported. They read observation times and the deterioration `ts`
time series, which only an all-record bundle (`--record all`) provides.

Training also evaluates the model. It prints the run directory, which holds the best checkpoint,
`metrics.csv` with the metrics of that checkpoint on each split, and `test_predictions.csv` with one
row per test record. The explanation and fairness steps below read these runs. See
[Getting started](../getting_started.md#multi-modal-learning) for the model configurations and the
[API reference](../api_reference.md) for all options.

---

## Explain the predictions

Explain the synthetic runs with the same script as real runs:

```sh
python -m src.explanation.explain --path ../logs/synthetic
```

Each run gains an `explanation/` folder with three kinds of output:

- **Modality contributions:** each modality's share of every prediction, from MM-SHAP.
- **Tabular feature SHAP:** the importance of each tabular feature, with a summary plot, a heatmap
  by default.
- **Patient decision plots:** one per patient, for the first ten test patients or the stay IDs
  passed with `--stays`.

See [tabular and modality SHAP](../api_reference.md#tabular-and-modality-shap) for the options.

**Compare explanations with how the data was generated.** In real data, how each input relates to
the outcome is unknown. In synthetic data, the generator sets it. Two hidden patient traits, illness
severity and frailty, influence every input type, so each input type is expected to contribute to
the predictions. Some input types also carry extra information about the outcome: the ECG and chest
X-ray for deterioration, and the notes and chest X-ray for hospitalisation. Feature names such as
`base_00` and `lab_1_min` are labels without clinical meaning, and plots show feature values scaled
from 0 to 1 rather than in measurement units.

**Spatial explanation is not supported.** `src.explanation.explain_cxr` draws Grad-CAM heatmaps on
the chest X-ray images. Synthetic CXR records are fictional embeddings with no image behind them, so
there is nothing to draw a heatmap on.

---

## Evaluate fairness

Each pathway folder also holds `synthetic_demographics.csv`, with fictional sex, age, ethnicity and,
for hospitalisation, insurance in the format of the real cohort tables. The demographics are not
model inputs. They carry known, deliberately planted biases, so the fairness script has gaps to find
and the expected direction of each one is documented:

| Attribute | Planted bias | Expected effect |
|---|---|---|
| Sex | None, drawn independently as a null control | Fairness ratios close to 1, apart from sampling noise |
| Age | Age rises with frailty, which raises outcome risk | Higher prevalence and selection rate in older groups, so a lower demographic parity ratio |
| Ethnicity | Patients with unrecorded ethnicity, reported as Other, have sparser records measured with extra error, with unchanged risk | Weaker discrimination and higher error rates for Other, so lower equalised odds and equal opportunity ratios once groups are large enough |
| Insurance (hospitalisation) | Medicare covers most patients aged 65 or over | Inherits the age gap, with a higher prevalence for Medicare |
| CXR availability (deterioration) | CXR absence is more likely at low acuity | Lower prevalence in the CXR absent group |

The manifest's `demographic_bias` and `demographic_groups` entries record these mechanisms and the
resulting per-group counts, prevalence and modality availability.

Run the fairness script on each pathway with its synthetic demographics:

```sh
python -m src.evaluation.fairness --path ../logs/synthetic/deterioration \
    --static ../outputs/synthetic/deterioration/synthetic_demographics.csv

python -m src.evaluation.fairness --path ../logs/synthetic/hospitalisation \
    --static ../outputs/synthetic/hospitalisation/synthetic_demographics.csv
```

Each run gains a `fairness/` folder to compare with the expected effects above. With the default
1,000 records the groups are small, so generate more with `--n-samples` for clearer estimates. See
[fairness evaluation](../api_reference.md#fairness-evaluation) for the options and outputs.
