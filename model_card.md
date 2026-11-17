# Model card: mm-healthfusion multimodal fusion models

## Model details

mm-healthfusion provides a shared framework for comparing multimodal clinical prediction models.
It extends [MM-HealthFair](https://github.com/nhsengland/mm-healthfair) to ECG waveforms, chest
X-rays and incomplete clinical records.
The models combine structured EHR data, vital signs and laboratory measurements with available ECG,
CXR and clinical note representations. The inputs depend on the prediction pathway.

Five fusion methods are available through the current model factory:

| Configuration name | Approach | Supplied pathway configurations |
|---|---|---|
| `concat` | Concatenate encoded modalities and pass them to a prediction head. | Deterioration and hospitalisation |
| `m3care` | Construct patient-similarity graphs for optional missing-modality imputation, then fuse the representations. | Deterioration and hospitalisation |
| `smp` | Route static modality representations through a static sparse MoE, with an optional shared expert. | Deterioration and hospitalisation |
| `fusemoe` | Use mTAND temporal encoding and mixture-of-experts routing within a transformer encoder. | Deterioration |
| `mtand_m3care` | Combine mTAND temporal encoding with M3Care imputation and fusion. | Deterioration |

Here and in our [report](report/mm_healthfusion.pdf), FuseMoE refers to both the static sparse MoE
and its temporal extension. In the code, the static variant is `smp`, while `fusemoe` names the
temporal variant. The main comparisons in report Tables 5.4 and 5.5 use the static variant.
Temporal experiments were restricted to mechanical ventilation in the deterioration pathway.
Factory support alone does not establish that a method has been evaluated for every pathway.

ECG, CXR and note embeddings are extracted with pretrained models. These extractors run in inference
mode, and their weights are not updated during fusion-model training:

| Modality | Pretrained extractor | Representation |
|---|---|---|
| ECG | ECG-FM, `wanglab/ecg-fm`, checkpoint `mimic_iv_ecg_physionet_pretrained.pt` | Five-second segment embeddings averaged into one vector per recording. |
| CXR | TorchXRayVision DenseNet-121, default weights `densenet121-res224-chex` | Spatial feature maps pooled into one vector per image. |
| Notes | `emilyalsentzer/Bio_Discharge_Summary_BERT`, with scispaCy sentence splitting | Sentence representations averaged into one vector per note. |

Trainable modality encoders map the prepared inputs into the representations used by each fusion
model. Missingness handling, M3Care contrastive objectives and encoder freezing depend on the
configuration. See the [developer guide](docs/developer_guide.md#fusion-models) for architecture
details and the [getting started guide](docs/getting_started.md#multimodal-learning) for supported
training configurations.

## Model use

### Intended use

These models are intended for retrospective research into multimodal risk prediction. They are
not intended for clinical use. The framework
supports comparisons of fusion architectures, missing-modality handling, fairness and explanations
across two clinical prediction pathways. Each training run predicts one binary outcome.

| Characteristic | Deterioration | Hospitalisation |
|---|---|---|
| Unit of analysis | ED stay | One current admission per patient |
| Prediction point | 90 minutes after ED arrival. | Transition from ED to inpatient care, operationalised using the recorded hospital admission time. |
| Observation window | First 90 minutes of the ED stay. | Current window from ED arrival to hospital admission, optionally with a previous window. |
| Inputs | Tabular EHR, ECG and CXR, with separate vital-sign and laboratory sequences for temporal models. | Static EHR, current and previous time series, current ECG, current and previous CXR, and previous notes. |
| Outcomes studied | Mechanical ventilation, cardiac arrest, one-day mortality, ICU admission within 24 hours and ICU admission during the hospital stay. | Extended stay over seven days, in-hospital death and non-home discharge. |

The hospitalisation previous window covers the next most recent eligible ED-linked admission, from
its ED arrival to discharge, provided that discharge was within 90 days before the current ED
arrival. Patients without such an admission remain eligible in the optional-history cohort.
Notes come from that previous admission. The current admission's discharge summary is excluded
to avoid revealing the outcome being predicted.

Outcome names do not imply identical prediction horizons. Deterioration ICU labels use admissions
at least 90 minutes after ED arrival, and the 24-hour ICU limit is measured from ED arrival.
The implemented one-day mortality label uses the integer day difference from arrival to the
recorded date of death, excluding deaths recorded before the 90-minute point. Mechanical
ventilation uses procedure codes dated no later than the calendar day after arrival, while cardiac
arrest uses diagnosis codes and ED or hospital discharge timing as proxies. These labels therefore
do not all establish an exact event time within 24 hours after prediction. See the
[cohort builder](src/preprocessing/build_deterioration.py) for the implemented definitions.

Hospitalisation outcomes refer to the current admission. Extended stay is measured from hospital
admission to discharge, in-hospital death uses the admission's recorded death timestamp, and
non-home discharge excludes patients labelled as having died in hospital.

## Training data

Data was downloaded from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) and the linked
datasets listed below. MIMIC-IV is a publicly accessible repository (subject to data usage
agreements and mandatory training) containing multimodal secondary care data from a single medical
centre in the US.

| Dataset | Release | Use |
|---|---|---|
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | Hospital records for both pathways and ICU records for deterioration. |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 2.2 | ED encounters for both pathways. |
| [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) | 1.0 | ECG recordings for both pathways. |
| [MIMIC-IV-ECG-Ext-ICD](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/) | 1.0.1 | Deterioration cohort linkage and inherited folds. |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | 2.1.0 | CXR images for both pathways. |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | 2.2 | Previous-admission discharge notes and identifiers for hospitalisation. |
| [MIMIC-IV-Ext-BHC](https://physionet.org/content/labelled-notes-hospital-course/1.2.0/) | 1.2.0 | Brief Hospital Course sections for hospitalisation notes. |

The reported deterioration cohort contained **116,468 ED stays from 71,098 patients**. Eligible
stays required static EHR information, an ECG within the first 90 minutes and linked diagnostic
codes. CXR could be missing, and no minimum number of vital-sign or laboratory measurements was
required. The builder applies no additional adult-age filter. CXR was available for 21.7% of stays
in the reported cohort.

The reported hospitalisation cohort contained **82,107 patients**, each represented by their most
recent eligible ED-linked admission. Inclusion required age of at least 18 years, valid encounter
chronology and recorded age, sex, marital status and insurance. Only static EHR was mandatory.
There were 11,979 patients with a qualifying admission within the 90-day lookback, so
previous-window modalities were sparse. Other modalities could also be absent in the current window.
Both cohort counts may change with future data releases or new cohort settings.

Each patient appears in only one of the training, validation and test sets. Deterioration inherits
the MDS-ED folds supplied with the ECG diagnostic extension and, by default, retains only the
earliest eligible stay per patient in validation and test. Training may include multiple stays per
patient. Hospitalisation uses a shared stratified 80/10/10 training, validation and test partition
across outcomes, stratifying by outcome labels, sex and ethnicity. The same split labels can be
reused across observation-window configurations.

Imputation statistics, normalisation parameters and categorical encoding mappings are fitted on
training data and applied to validation and test data. Missingness indicators are retained.
See [data access and preparation](docs/getting_started.md#data-access) and the
[cohort reference](docs/api_reference.md#cohort-building) for requirements and configuration options.

**Synthetic data.** Users without MIMIC access can generate fictional records for both pathways in
the same format as the average-record prepared datasets. The demo notebook
[`notebooks/01 Synthetic Data Demo.ipynb`](notebooks/01%20Synthetic%20Data%20Demo.ipynb) runs the
full workflow on them. It trains the static MoE model, evaluates it on the test split, explains it
with modality and feature SHAP, and checks its fairness against known, deliberately planted
demographic biases. The [synthetic data example](docs/examples/synthetic.md) covers every static
fusion model. The records and outcomes are artificial, so results on them say nothing about
clinical performance.

## Performance and limitations

### Predictive performance

The tables below reproduce test-set Matthews correlation coefficients (MCC) and 95% bootstrap
confidence intervals from Tables 5.4 and 5.5 of the [report](report/mm_healthfusion.pdf).
These are exploratory results. More complex fusion did not consistently outperform concatenation.

**Deterioration**

| Outcome | Concatenation | M3Care | FuseMoE |
|---|---|---|---|
| Cardiac arrest | 0.349 [0.240–0.463] | 0.440 [0.335–0.519] | 0.318 [0.233–0.390] |
| ICU admission within 24 hours | 0.597 [0.560–0.635] | 0.548 [0.511–0.587] | 0.573 [0.539–0.610] |
| ICU admission during the stay | 0.543 [0.508–0.580] | 0.547 [0.514–0.584] | 0.552 [0.518–0.587] |
| Mechanical ventilation | 0.498 [0.443–0.538] | 0.542 [0.488–0.589] | 0.574 [0.509–0.624] |
| One-day mortality | 0.206 [0.144–0.257] | 0.197 [0.131–0.256] | 0.223 [0.151–0.291] |

**Hospitalisation**

| Observation windows | Outcome | Concatenation | M3Care | FuseMoE |
|---|---|---|---|---|
| Current only | Extended stay over seven days | 0.301 [0.281–0.322] | 0.300 [0.278–0.323] | 0.288 [0.267–0.310] |
| Current only | In-hospital death | 0.346 [0.317–0.373] | 0.354 [0.324–0.382] | 0.339 [0.313–0.367] |
| Current only | Non-home discharge | 0.424 [0.404–0.445] | 0.416 [0.394–0.437] | 0.424 [0.404–0.448] |
| Current and previous | Extended stay over seven days | 0.300 [0.277–0.322] | 0.299 [0.275–0.320] | 0.300 [0.277–0.323] |
| Current and previous | In-hospital death | 0.355 [0.324–0.384] | 0.336 [0.305–0.367] | 0.332 [0.307–0.358] |
| Current and previous | Non-home discharge | 0.434 [0.413–0.455] | 0.421 [0.400–0.442] | 0.418 [0.397–0.441] |

For each training run, the checkpoint was selected by validation MCC. Standard evaluation used a
probability threshold of 0.5. Reported intervals used 1,000 bias-corrected and accelerated (BCa)
bootstrap resamples with seed 42. Paired MCC comparisons aligned encounters and used 10,000
prediction-swap permutations, with Benjamini–Hochberg correction for multiple comparisons.

M3Care and FuseMoE both beat concatenation on mechanical ventilation in the report's
adjusted comparisons. Concatenation did better on ICU admission within 24 hours. Across the
hospitalisation outcomes, it was never more than 0.01 MCC below the best method. The report gives
the full comparison results, alongside AUROC, sensitivity, specificity and balanced accuracy.

For reproducibility, retain the code revision, dataset releases, cohort and window settings,
record aggregation mode, split identifiers, seed, resolved configuration and checkpoint for each
experiment. The [run logs and checkpoints](docs/api_reference.md#run-logs-and-checkpoints) section
describes saved artefacts, and the [evaluation reference](docs/api_reference.md#evaluation) covers
analysis commands.

### Fairness and explainability

Fairness was evaluated across sex, age and ethnicity in both pathways. Hospitalisation also used
insurance groups, and deterioration used CXR availability. The report presents demographic parity,
equal opportunity and equalised odds ratios, each with bootstrap confidence intervals. More complex
fusion did not make the models consistently fairer. The project measured differences between
groups but did not try to reduce them.

Feature-level SHAP and modality-level MM-SHAP show how much each input contributes to a prediction.
Modality permutation shuffles a modality's values between patients while keeping each patient's
availability fixed. This separates what the values tell the model from whether the modality was
recorded at all. CXR Grad-CAM highlights the image regions behind a prediction. These methods show
what the model relied on, not what causes the outcome. See the
[explainability reference](docs/api_reference.md#explainability) for supported analyses.

### Limitations

- **Generalisability and selection.** All reported experiments use retrospective data from one US
  medical centre. There is no external NHS validation, and ECG eligibility or the availability of
  previous records selects particular patient groups. Findings may change with local workflows,
  admission practices and discharge policies.
- **Informative missingness.** Investigations are ordered for clinical reasons. Their availability
  can carry predictive information, so changes in investigation practices may alter model behaviour
  even when the recorded clinical values appear similar.
- **Training variability and tuning.** Most final comparisons and ablations used a single random
  seed. The bootstrap intervals do not cover variation between training runs. Hyperparameter tuning
  did check stability across three seeds. Tuning focused on mechanical ventilation and extended
  stay, and the chosen settings were reused for the other outcomes.
- **Model selection.** Architectural configurations were selected using test-set MCC during
  ablations. Reusing that test partition for final reporting can make performance estimates
  optimistic. The bootstrap intervals do not account for this selection bias.
- **Representations.** The pretrained extractors were used as they are, without fine-tuning for
  these outcomes. Averaging each recording, image or note into one vector loses detail, which
  limits what the fusion models can learn.
- **Batch-dependent imputation.** M3Care with imputation compares each patient with the others in
  the same batch. A patient's prediction can therefore change with who else is in the batch. How
  much it changes was not fully tested. Explanations need the same batch as the prediction.
- **Temporal and contrastive experiments.** Temporal runs used individual timestamped records
  instead of pooled inputs. Any performance difference cannot be put down to temporal encoding
  alone. Contrastive-learning experiments covered only some settings. In some hospitalisation
  comparisons, the training strategy and the set of modalities changed together.
- **Subgroup estimates and explanations.** Rare outcomes and small subgroups give wide fairness
  intervals, and some ratios cannot be computed at all. Intersectional groups were not studied in
  depth. There are no explanations for individual time points or note text. CXR heatmaps only
  roughly locate what the model used, and they have not been checked against anatomy or findings.
- **Calibration and clinical utility.** The 0.5 threshold was not tuned per outcome. Calibration,
  treatment thresholds and clinical benefit were not evaluated in depth.

Further discussion is provided in Section 6.3 of the [report](report/mm_healthfusion.pdf).

## Acknowledgements

mm-healthfusion was jointly developed by [Xin Zhao](https://github.com/xinzhao13) and
[Zhiyan Bo](https://github.com/ZhiyanBo) during the NHS England Data Science
PhD Internship in 2026. Xin contributed fusion architectures, data pipelines, training infrastructure,
evaluation and explainability. Zhiyan contributed CXR embedding extraction, M3Care development,
contrastive learning and two-stage training. Both contributed integration, testing and documentation.

The project builds on the earlier [MM-HealthFair](https://github.com/nhsengland/mm-healthfair) work
by Sophie Martin and Konstantin Georgiev during an NHS England PhD internship.
See the [citations guide](docs/citations.md) for the research methods and pretrained models used
here.
Code and documentation licensing is described in the [README](README.md#-licence). Source datasets
and pretrained models remain subject to their respective access and licence terms.
