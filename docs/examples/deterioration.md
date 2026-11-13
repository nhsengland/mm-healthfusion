# Deterioration example

This example trains a concatenation model to predict mechanical ventilation. Follow the
[data access and installation steps](../getting_started.md) first. Run commands from the
repository root in the activated environment.

## Build the data

Build the cohort and extract ECG and chest X-ray embeddings. Existing full-source embeddings
can be reused.

```sh
python -m src.preprocessing.build_deterioration
python -m src.preprocessing.extract_ecg
python -m src.preprocessing.extract_cxr
```

Prepare and split the data. Replace `path/to/bundle` with the directory containing the files
printed by the preparation command.

```sh
python -m src.preprocessing.prepare_deterioration --patient-stay all --record average --cohort all
python -m src.preprocessing.split_deterioration --bundle path/to/bundle --normalise --ohe
```

## Train and evaluate

Use the split directory printed by the previous command:

```sh
python -m src.train --config config/concat.yaml --set data.split_dir=path/to/split
```

Training prints the run directory. Use its checkpoint in the
[evaluation and explainability example](analysis.md). For FuseMoE and mTAND-M3Care, prepare
and split a separate bundle using `--record all`.

See the [API reference](../api_reference.md#deterioration-preparation-and-splitting) for
options and outputs, and [Getting started](../getting_started.md#multimodal-learning) for other
model configurations.
