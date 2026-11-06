# mm-healthfusion: building multi-modal patient representations for clinical prediction

## NHSE PhD internship project

## 📖 About this project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENCE)

**mm-healthfusion** explores fairness and explainability in multi-modal healthcare prediction. It
builds on the MM-HealthFair project and supports two workflows using linked MIMIC datasets:

- **Deterioration:** prediction from the first 90 minutes of an emergency department stay, using
  tabular features, ECG and CXR embeddings, with raw vitals/labs sequences for temporal models.
- **Hospitalisation:** prediction for the current admission, using its pre-admission ED window
  and eligible previous-admission history, including tabular data, vitals/labs, ECG, CXR and notes.

The pipeline covers cohort extraction, embedding extraction, dataset preparation, fusion-model
training, performance and fairness evaluation, and SHAP, MM-SHAP and CXR Grad-CAM explanations.
Available fusion methods are `concat`, `m3care`, `smp`, `fusemoe` and `mtand_m3care`, with
task-specific data and configuration requirements.

Start with [Getting started](docs/getting_started.md) for data requirements and the two workflows.
The [documentation home](docs/index.md) provides overview figures and links to every active guide.
<!-- See the [original project proposal](https://nhsx.github.io/nhsx-internship-projects/advances-modalities-explainability/)
for the research background. -->

## 📁 Project structure

| Directory | Contents |
|---|---|
| `src/` | Preprocessing, data loading, encoders, models, training, evaluation and explanation code |
| `config/` | Experiment YAML files and preprocessing lookup tables |
| `docs/` | User and developer guides |
| `notebooks/` | End-to-end synthetic data demo (data, training, evaluation, explanation and fairness), model diagnostics and report figures |
| `report/` | Project report |

## 🧰 Built with

In the latest iteration, the framework was developed using [Python 3.10](https://www.python.org/) (3.10.20), with 3.11 also supported. Model training and evaluation were performed on a Microsoft Azure virtual machine running Ubuntu 22.04 LTS with 1 × NVIDIA Tesla T4 GPU (16 GiB VRAM) and 4 × vCPUs (28 GiB memory).

<!--  Models are built and trained with
[PyTorch 2.5.1](https://pytorch.org/) (CUDA 12.1) and [Lightning 2.6](https://lightning.ai/docs/pytorch/stable/).
Cohort extraction and data preparation use [pandas](https://pandas.pydata.org/) and
[Polars](https://pola.rs/), explanations use [SHAP](https://shap.readthedocs.io/), and
[uv](https://docs.astral.sh/uv/) manages dependencies, with exact versions pinned in `uv.lock`. -->

## 🚀 Getting started

### Clone the repository

```sh
git clone https://github.com/xinzhao13/mm-healthfusion
cd mm-healthfusion
```

Run all commands from the repository root.

### Installation

Python 3.10 and 3.11 are supported. The three options below all install the same package
versions, including the packages needed under [Extraction prerequisites](#extraction-prerequisites).

#### uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), for example with
`curl -LsSf https://astral.sh/uv/install.sh | sh`. Then, from the repository folder, run:

```sh
uv sync
```

This sets up everything the project needs in a `.venv` folder and downloads a supported Python
version if you don't already have one. To use Python 3.10, the version behind the reported
results, run `uv sync --python 3.10` instead. You can then run commands in either of two ways:

```sh
uv run python -m src.train ...      # no activation needed

source .venv/bin/activate           # or activate once
python -m src.train ...
```

`uv run` keeps the environment up to date for you. If you activate the environment instead, run
`uv sync` again whenever you pull changes that update `uv.lock`.

<details>
<summary>Alternative: conda</summary>

`environment.yml` provides Python 3.10 and uv only. uv then installs the locked packages into the
conda environment:

```sh
conda env create -f environment.yml
conda activate mm-healthfusion
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --frozen --inexact
```

`--inexact` leaves the packages conda installed in place. Run commands with `python ...` in the
activated environment.

</details>

<details>
<summary>Alternative: pip and a virtual environment</summary>

`requirements.txt` is generated from `uv.lock` and pins the same packages, including the PyTorch
package index:

```sh
python3.10 -m venv _env
source _env/bin/activate
python -m pip install --no-deps --use-pep517 -r requirements.txt
```

`--no-deps` installs exactly the pinned set and skips the spaCy model's outdated spaCy requirement
(see [spaCy version warning](#extraction-prerequisites)). `--use-pep517` is needed for
`confidenceinterval`, which is published only as a source archive. Without it, older pip versions
build that package without its code. Run commands with `python ...` in the activated environment.

</details>

<details>
<summary>PyTorch wheel selection</summary>

The lock selects CUDA 12.1 PyTorch wheels. The `torch` and `torchvision` entries use the explicit
`pytorch-cu121` index in `[tool.uv.sources]` in `pyproject.toml`. CPU-only installations and other
platforms need a different wheel selection. Change or remove those sources, then run `uv lock`.
Passing `--device cpu` to `src.preprocessing.extract_cxr` does not change which wheels are installed.

</details>

<details>
<summary>C compiler for fairseq-signals</summary>

`fairseq-signals` is not on PyPI, so it is installed from a commit of the
[Jwoo5/fairseq-signals](https://github.com/Jwoo5/fairseq-signals) repository pinned in
`pyproject.toml`. It builds a
compiled extension during installation, which needs a C compiler on the machine. Without one,
installation stops at `fairseq-signals`. This applies whether you install with uv, conda or pip.
On Ubuntu, install a compiler with:

```sh
sudo apt install build-essential
```

</details>

### Updating dependencies

Edit `pyproject.toml`, then run `uv lock`. Never edit `uv.lock` or `requirements.txt` by hand.
With the pre-commit hooks installed (`pre-commit install`), each commit re-locks and regenerates
`requirements.txt`. CI fails when either file is out of date with `pyproject.toml`.

### Extraction prerequisites

This repo provides built-in embedding extractors for three modalities. Once set up, the environment already contains every package
below. The pretrained models are downloaded on each extractor's first run, which needs internet
access.

| Modality | Extractor | Packages required | Pretrained model downloaded on first run | Guide |
|---|---|---|---|---|
| ECG | `src.preprocessing.extract_ecg` | `fairseq-signals` | ECG-FM checkpoint `mimic_iv_ecg_physionet_pretrained.pt` from the Hugging Face repository `wanglab/ecg-fm` | [ECG extraction](docs/api_reference.md#ecg-embeddings) |
| CXR | `src.preprocessing.extract_cxr` | `torchxrayvision`, `opencv-python`, `scikit-image` | TorchXRayVision DenseNet weights, `densenet121-res224-chex` by default | [CXR extraction](docs/api_reference.md#chest-x-ray-embeddings) |
| Notes | `src.preprocessing.extract_notes` | `spacy` with the `en_core_sci_md` model for sentence splitting, `transformers` | `emilyalsentzer/Bio_Discharge_Summary_BERT` from Hugging Face | [Notes extraction](docs/api_reference.md#note-embeddings) |

<details>
<summary>spaCy version warning</summary>

When notes extraction starts, spaCy may warn that the `en_core_sci_md` model was built for an
older version of spaCy. This is expected and does not change the results.

</details>

## 💻 Usage

### Datasets

Only public or synthetic data are shared in this repository. Request access to the source datasets
through [PhysioNet](https://physionet.org/), complete each dataset's access requirements, then
download the releases below.

| Dataset | Release | Required for |
|---|---|---|
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | Both tasks: `hosp`. Deterioration also uses `icu` |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 2.2 | Both tasks |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) | 1.0 | Both tasks |
| [MIMIC-IV-ECG-Ext-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/) | 1.0.1 | Deterioration cohort |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | 2.1.0 | Both tasks |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) and [MIMIC-IV-Ext-BHC](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/) | 2.2 and 1.2.0 | Hospitalisation notes |

The scripts read the source data from `data/MIMIC-IV` beside the repository folder. See
[Data access](docs/getting_started.md#data-access) for the expected folder layout.

### Synthetic data

Without MIMIC access, generate fictional data for both tasks:

```sh
python -m src.preprocessing.generate_synthetic
```

These files have the same layout as the real prepared data, so you can train models, check
fairness and explain predictions without MIMIC access. The demo notebook
[`notebooks/01 Synthetic Data Demo.ipynb`](notebooks/01%20Synthetic%20Data%20Demo.ipynb) walks
through the whole process with one model, from training to fairness checks. The
[synthetic data example](docs/examples/synthetic.md) gives the same steps as commands for every
static fusion model and both tasks.

### Running the pipeline

Follow the [Getting started guide](docs/getting_started.md) to select a task, prepare train/validation/test
parquet files and train a baseline. Detailed configuration and model interfaces are described in
the [Developer guide](docs/developer_guide.md). Prepared data and embeddings are written under `outputs`, and training runs under `logs`, both
**beside** the cloned repository. See [Model run logs](docs/getting_started.md#model-run-logs) for
what each run contains.

- [Evaluation](docs/api_reference.md#evaluation): prediction generation, confidence intervals, paired model
  comparisons, modality tests, expert diagnostics and subgroup fairness.
- [Explainability](docs/api_reference.md#explainability): feature SHAP, MM-SHAP modality contributions and CXR
  Grad-CAM overlays.

## 👥 Contributors

**mm-healthfusion** was jointly developed by **Xin Zhao** and **Zhiyan Bo** during the
[NHS England Data Science PhD Internship](https://nhsx.github.io/nhsx-internship-projects/) in 2026.

Xin Zhao · [@xinzhao13](https://github.com/xinzhao13) · 380 commits on `main` (excluding merges)

- Fusion architectures: Developed multimodal fusion architectures, including concatenation
  baselines and FuseMoE.
- Data pipelines: Developed data preparation, cohort construction and multimodal integration
  pipelines.
- Experiment infrastructure: Implemented shared infrastructure for training, evaluation and
  experimentation.
- Evaluation and explainability: Developed model evaluation, fairness assessment and
  explainability methods.
- Code and documentation: Integrated, tested and refactored code, and contributed documentation.

Zhiyan Bo · [@ZhiyanBo](https://github.com/ZhiyanBo) · 87 commits on `main` (excluding merges)

- CXR embeddings: Developed chest X-ray embedding extraction pipelines.
- M3Care architecture: Implemented and extended the M3Care multimodal architecture.
- Training methods: Developed contrastive learning and two-stage training procedures for M3Care.
- Training infrastructure: Extended model training infrastructure to support pretrained weight
  loading and encoder freezing.
- Evaluation: Developed imputation quality evaluation for M3Care.
- Code and documentation: Integrated and tested models, and contributed documentation.

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## 📜 Licence

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government Licence v3.0][ogl].

[mit]: LICENCE
[copyright]: https://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## 📬 Contact

To find out more about the Analytics Unit, visit our
[project website](https://nhsx.github.io/nhsx-internship-projects/) or get in touch at
[datascience@nhs.net](mailto:datascience@nhs.net).
