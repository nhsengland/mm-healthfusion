# Getting started

Follow either the deterioration or hospitalisation pathway below, from source data to model
analysis. Run the commands from the repository root in an activated Python environment. For more
options and commands of each task, see the [deterioration](examples/deterioration.md) and
[hospitalisation](examples/hospitalisation.md) examples.

**Without MIMIC access.** After [installation](#installation), the
[synthetic data example](examples/synthetic.md) generates fictional records for both pathways in
the same format as the average-record prepared datasets. Its demo notebook runs the full workflow on
them: the data formats, then training, evaluation, explanation and fairness evaluation of the
static MoE model. The example also gives the commands to train every static fusion model.

## Data access

Obtain access through [PhysioNet](https://physionet.org/) and complete each dataset's access requirements.

| Dataset | Release | Required for |
|---|---|---|
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | Both tasks: `hosp`. Deterioration also uses `icu` |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 2.2 | Both tasks |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) | 1.0 | Both tasks |
| [MIMIC-IV-ECG-Ext-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/) | 1.0.1 | Deterioration cohort |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | 2.1.0 | Both tasks |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) and [MIMIC-IV-Ext-BHC](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/) | 2.2 and 1.2.0 | Hospitalisation notes |

The scripts resolve their default paths from the repository location. Source data, prepared
outputs and training logs sit in folders beside the repository, so a clone at
`/path/mm-healthfusion` uses:

```text
/path/
├── mm-healthfusion/  # this repository, run commands from here
├── data/MIMIC-IV/    # PhysioNet source releases
├── outputs/          # cohort tables, embeddings and prepared datasets
└── logs/             # training runs and W&B files
```

Paths in these guides are written relative to the repository root, for example `../outputs`. The
source data under `../data/MIMIC-IV` follows this layout:

```text
../data/MIMIC-IV/
├── mimiciv/3.1/
│   ├── hosp/
│   └── icu/
├── mimic-iv-ed/2.2/
│   └── ed/
├── mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/
│   ├── record_list.csv
│   └── files/
├── mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.1/
│   └── records_w_diag_icd10.csv
├── mimic-cxr-jpg/
│   ├── mimic-cxr-2.0.0-metadata.csv
│   └── files/
├── mimic-iv-note-2.2/
│   └── note/
│       └── discharge.csv
└── mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0/
    └── mimic-iv-bhc.csv  # Brief Hospital Course segments
```

Keep the repository separate from the source data and run commands from the repository root. The
cohort builders have no source path options, so their inputs must follow this layout. Check the
[deterioration](api_reference.md#inputs) and [hospitalisation](api_reference.md#hospitalisation-cohort)
sections for required filenames and compression. The embedding extractors accept path overrides. See
the [API reference](api_reference.md#modality-embedding-extraction).

## Installation

Create the project environment with uv, conda or pip, following the
[README](https://github.com/xinzhao13/mm-healthfusion/blob/main/README.md#installation).
All three routes install the same locked packages, including the embedding extractors'
dependencies and the spaCy model. The commands in this guide assume an activated environment.
With uv, activate `.venv` with `source .venv/bin/activate` or prefix each command with `uv run`.

## Data curation

### Cohort curation

Choose a task. For deterioration, the default `ts` mode produces the inputs for preparation,
including the base table used for fairness analysis. For hospitalisation, the command produces
the cohort, static and event tables used for preparation. Its static table is also used for
fairness analysis. See the [deterioration](api_reference.md#deterioration-cohort) and
[hospitalisation](api_reference.md#hospitalisation-cohort) cohort sections.

```sh
# Deterioration
python -m src.preprocessing.build_deterioration

# Hospitalisation
python -m src.preprocessing.build_hospitalisation
```

### Modality embedding extraction

Extract [ECG](api_reference.md#ecg-embeddings) and
[chest X-ray](api_reference.md#chest-x-ray-embeddings) embeddings for either task. Hospitalisation
also needs [note embeddings](api_reference.md#note-embeddings). Matching full-source ECG and chest
X-ray embeddings can be reused across tasks. Hospitalisation uses notes from previous admissions,
avoiding the current admission's discharge summary.

```sh
python -m src.preprocessing.extract_ecg
python -m src.preprocessing.extract_cxr
python -m src.preprocessing.extract_notes # hospitalisation only
```

### Dataset preparation

Prepare a bundle and split it into training, validation and test datasets. For either task,
replace `path/to/bundle` with the directory containing the files printed by the preparation command.

**Deterioration:**

```sh
python -m src.preprocessing.prepare_deterioration --patient-stay all --record average --cohort all
python -m src.preprocessing.split_deterioration --bundle path/to/bundle --normalise --ohe
```

**Hospitalisation:**

```sh
python -m src.preprocessing.prepare_hospitalisation
python -m src.preprocessing.split_hospitalisation --bundle path/to/bundle --normalise --ohe
```

See the preparation and splitting reference for
[deterioration](api_reference.md#deterioration-preparation-and-splitting) or
[hospitalisation](api_reference.md#hospitalisation-preparation-and-splitting).

## Model training and evaluation

### Multi-modal learning

Choose a model and replace `path/to/split` with the split directory printed during dataset
preparation. The following deterioration commands use the average-record split above for **static**
fusion models:

```sh
# Concatenation
python -m src.train --config config/concat.yaml --set data.split_dir=path/to/split

# M3Care
python -m src.train --config config/m3care.yaml --set data.split_dir=path/to/split

# Static MoE model
python -m src.train --config config/smp.yaml --set data.split_dir=path/to/split
```

For FuseMoE and mTAND-M3Care **temporal** modelling, first prepare and split a deterioration bundle
with `--record all` instead of `--record average`, then use that split directory:

```sh
# FuseMoE
python -m src.train --config config/fusemoe.yaml --set data.split_dir=path/to/all_record_split

# mTAND-M3Care
python -m src.train --config config/m3care/mtand_m3care_deterioration.yaml --set data.split_dir=path/to/all_record_split
```

For hospitalisation (static fusion only), use the hospitalisation split with one of these
configurations:

```sh
# Concatenation
python -m src.train --config config/concat_hospitalisation.yaml --set data.split_dir=path/to/split

# M3Care
python -m src.train --config config/m3care_hospitalisation.yaml --set data.split_dir=path/to/split

# Static MoE model
python -m src.train --config config/smp_hospitalisation.yaml --set data.split_dir=path/to/split
```

The deterioration examples predict mechanical ventilation. The hospitalisation examples predict
an extended stay over seven days. Each run trains one binary outcome and prints its run directory.
For other outcomes, see the [general usage tips](#general-usage-tips). For model variants, see
[fusion models](developer_guide.md#fusion-models) in the developer guide.

### Model run logs

The run directory contains the checkpoint and results used by the analysis steps below:

```text
../logs/<task>/<data settings>/<outcome>/runs/<model>_<run_id>/
├── config.yaml
├── checkpoint_best.ckpt
├── test_predictions.csv
├── metrics.csv
└── history.csv
```

See [run logs and checkpoints](api_reference.md#run-logs-and-checkpoints) for more details.

### Multi-modal evaluation

Training writes test predictions and metrics. To estimate confidence intervals for those metrics,
replace `/path/to/your/run` with the printed run directory and run
[bootstrap evaluation](api_reference.md#evaluation):

```sh
python -m src.evaluation.bootstrap_metric --path /path/to/your/run
```

### Fairness evaluation

Measure subgroup performance using the test predictions and the task's cohort static data. See
the [fairness reference](api_reference.md#evaluation).

```sh
python -m src.evaluation.fairness --path /path/to/your/run
```

### Model explainability

Generate tabular and modality attributions for the baseline checkpoint. The
[explainability guide](api_reference.md#explainability) also covers chest X-ray overlays and model
support.

```sh
python -m src.explanation.explain --path /path/to/your/run
```

## General usage tips

- Replace example paths when using a different layout. Consult each guide for supported overrides.
- Use `--help` on a script for its options. Prepared data defaults to `../outputs`. Training
  runs are written under `../logs`. Analysis results are saved with the run or at the
  documented output path.
- **Add a fusion method:** Implement a model in `src/models/`, register it in the model factory and
  add a configuration. Follow the [fusion model guide](developer_guide.md#adding-a-new-fusion-model)
  for the batch and output contracts.
- **Run names:** Name `--run-id` with the model followed by values and capitalised setting names:
  `<model>_<value><Setting>_<value><Setting>_...`, for example
  `fusemoe_0_1SharedWeight_0_1ModalityDropout_1e_1Lb`. Set the corresponding options in the
  configuration or with `--set`. The name does not configure them. The local directory is named
  `<fusion_method>_<run_id>`, under the data and outcome path. For W&B, `--run-name` optionally
  sets a display name without changing the local directory.
- **Outcomes:** Set `outcome` in the configuration or use, for example,
  `--set outcome=deterioration_icu_24h`. Select one binary outcome available in the split:
  - Deterioration: prefix `deterioration_` to `severe_hypoxemia`, `ecmo`, `vasopressors`,
    `inotropes`, `mechanical_ventilation`, `cardiac_arrest`, `icu_24h`, `icu_stay` or
    `mortality_1d`.
  - Hospitalisation: `hospitalisation_ext_stay_7`, `hospitalisation_in_hosp_death` or
    `hospitalisation_non_home_discharge`.
