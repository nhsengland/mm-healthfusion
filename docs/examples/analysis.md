# Evaluation and explainability example

Training writes test predictions and metrics. Replace `/path/to/your/run` with the run directory
printed during training. These commands work with a compatible run from either task. `--path`
searches the directory for every `checkpoint_best.ckpt`, so pointing it at a parent folder
processes all the runs inside it.

```sh
# Generate validation predictions
python -m src.evaluation.generate_prediction --path /path/to/your/run --split val

# Estimate confidence intervals and measure subgroup fairness
python -m src.evaluation.bootstrap_metric --path /path/to/your/run
python -m src.evaluation.fairness --path /path/to/your/run

# Explain the concatenation baseline
python -m src.explanation.explain --path /path/to/your/run
```

Fairness analysis also reads the task's cohort table. See the
[evaluation reference](../api_reference.md#evaluation) for defaults and the
[explainability reference](../api_reference.md#explainability) for model support and image overlays.
