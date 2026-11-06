# Hospitalisation example

This example trains a concatenation model to predict a hospital stay longer than seven days.
Follow the [data access and installation steps](../getting_started.md) first. Run commands from
the repository root in the activated environment.

## Build the data

Build the cohort and extract ECG, chest X-ray and note embeddings. Existing full-source ECG
and chest X-ray embeddings can be reused from the deterioration workflow.

```sh
python -m src.preprocessing.build_hospitalisation
python -m src.preprocessing.extract_ecg
python -m src.preprocessing.extract_cxr
python -m src.preprocessing.extract_notes
```

Prepare and split the data. Replace `path/to/bundle` with the directory containing the files
printed by the preparation command.

```sh
python -m src.preprocessing.prepare_hospitalisation
python -m src.preprocessing.split_hospitalisation --bundle path/to/bundle --normalise --ohe
```

## Train and evaluate

Use the split directory printed by the previous command:

```sh
python -m src.train --config config/concat_hospitalisation.yaml --set data.split_dir=path/to/split
```

Training prints the run directory. Use its checkpoint in the
[evaluation and explainability example](analysis.md). The
[API reference](../api_reference.md#hospitalisation-preparation-and-splitting) explains the current
and legacy windows, and how to reuse a split across them.
