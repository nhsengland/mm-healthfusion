"""Measure how predictions change when modality content is shuffled.

Keep modality availability fixed and write results to modality_permutation.

Usage:
    python -m src.evaluation.modality_permutation --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.modality_permutation --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import MultimodalSplitDataset
from src.data.splits import get_datasets, get_permuted_test_dataloader
from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    _scenario_torchmetrics,
    _usable_device,
    discover_checkpoints,
    load_config,
    predict_batches,
)
from src.models.base import canonical_modalities
from src.training.lightning import MMModel

BASELINE_SCENARIO = "baseline"
PERMUTE_ALL_SCENARIO = "permute_all"
METRIC_KEYS = ("auroc", "recall", "specificity", "mcc", "ba")

# Compare permute_all scores with these chance values
NULL_SKILL_VALUES: dict[str, float] = {
    "auroc": 0.5,
    "recall": 0.5,
    "specificity": 0.5,
    "mcc": 0.0,
    "ba": 0.5,
}
DEFAULT_DRAWS = 100
DEFAULT_BASE_SEED = 42


def permutation_scenarios_for_active_modalities(
    active_modalities: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Return baseline, single-modality, and all-modality permutation scenarios."""
    active = canonical_modalities(tuple(str(modality) for modality in active_modalities))
    scenarios: dict[str, tuple[str, ...]] = {BASELINE_SCENARIO: ()}
    scenarios.update(
        {f"permute_{modality}": (modality,) for modality in active}
    )
    scenarios[PERMUTE_ALL_SCENARIO] = active
    return scenarios


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run modality-information permutation for a single checkpoint "
            "(--checkpoint) or every run under a directory tree (--path)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a single trained Lightning checkpoint.",
    )
    source.add_argument(
        "--path",
        default=None,
        help=(
            "Root directory searched recursively for run checkpoints. Every "
            "checkpoint_best.ckpt file found is evaluated in place."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "YAML config used to build the test dataloader. Defaults to "
            "config.yaml in each checkpoint's run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the permutation CSVs. Defaults to the checkpoint's run "
            "directory. Ignored in --path mode (each run writes its own)."
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_DRAWS,
        help=f"Seeded permutation draws per scenario (default: {DEFAULT_DRAWS}).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"Seed of the first draw (default: {DEFAULT_BASE_SEED}).",
    )
    parser.add_argument(
        "--emit-predictions",
        action="store_true",
        help=(
            "Also write per-scenario {scenario}_predictions.csv (first draw) for "
            "paired testing with src.evaluation.compare_prediction."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --path mode, skip runs that already have the summary CSV.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In --path mode, keep going on failure and report failures at the end.",
    )
    return parser.parse_args(argv)


def _n_draws_for(scenario: str, draws: int) -> int:
    """Use one draw for the unchanged baseline and the requested count for other scenarios."""
    return 1 if scenario == BASELINE_SCENARIO else draws


def _score_predictions(
    prediction_batches: list[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, float], dict[str, int], torch.Tensor, torch.Tensor]:
    """Concatenate prediction batches and score with train.py's metric classes."""
    labels = torch.cat([torch.atleast_1d(b["label"]) for b in prediction_batches])
    probabilities = torch.cat(
        [torch.atleast_1d(b["probability"]) for b in prediction_batches]
    )
    predictions = torch.cat(
        [torch.atleast_1d(b["prediction"]) for b in prediction_batches]
    )
    targets = labels.to(dtype=torch.int64).reshape(-1)
    probabilities = probabilities.to(dtype=torch.float32).reshape(-1)
    positives = int(targets.sum().item())
    counts = {
        "n": int(targets.numel()),
        "positive_n": positives,
        "negative_n": int(targets.numel() - positives),
    }
    metrics = _scenario_torchmetrics(probabilities, targets)
    return metrics, counts, probabilities, predictions.reshape(-1)


def _draw_metric_row(
    scenario: str,
    modalities: tuple[str, ...],
    seed: int,
    metrics: Mapping[str, float],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    row = {
        "scenario": scenario,
        "seed": seed,
        "permuted_modalities": "+".join(modalities) if modalities else "none",
        "n": counts["n"],
        "positive_n": counts["positive_n"],
        "negative_n": counts["negative_n"],
    }
    row.update({key: float(metrics[key]) for key in METRIC_KEYS})
    return row


def run_modality_permutation(
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    output_dir: str | Path,
    *,
    base_dataset: MultimodalSplitDataset,
    scenarios: Mapping[str, tuple[str, ...]] | None = None,
    draws: int = DEFAULT_DRAWS,
    base_seed: int = DEFAULT_BASE_SEED,
    emit_predictions: bool = False,
) -> dict[str, Path]:
    """Score each scenario and draw, then write individual metrics and a summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if scenarios is None:
        scenarios = permutation_scenarios_for_active_modalities(
            base_dataset.modality_names
        )
    # Keep stay IDs in dataset order to match the unshuffled predictions
    ed_stay_ids = base_dataset.frame["ed_stay_id"].to_numpy()

    metric_rows: list[dict[str, Any]] = []
    first_draw_predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    total_draws = sum(_n_draws_for(name, draws) for name in scenarios)
    progress = tqdm(total=total_draws, desc="permutation draws", unit="draw")
    for scenario, modalities in scenarios.items():
        for draw in range(_n_draws_for(scenario, draws)):
            seed = base_seed + draw
            loader: DataLoader = get_permuted_test_dataloader(
                cfg, modalities, seed=seed, base=base_dataset
            )
            batches = predict_batches(model, loader)
            metrics, counts, probabilities, predictions = _score_predictions(batches)
            metric_rows.append(
                _draw_metric_row(scenario, modalities, seed, metrics, counts)
            )
            if draw == 0:
                first_draw_predictions[scenario] = (probabilities, predictions)
            progress.set_postfix_str(scenario)
            progress.update(1)
    progress.close()

    metrics_path = output_dir / "test_modality_permutation_metrics.csv"
    summary_path = output_dir / "test_modality_permutation_summary.csv"
    _write_metrics_csv(metric_rows, metrics_path)
    _write_summary_csv(metric_rows, summary_path)
    paths = {
        "test_modality_permutation_metrics_path": metrics_path,
        "test_modality_permutation_summary_path": summary_path,
    }
    if emit_predictions:
        paths["prediction_paths"] = _write_prediction_csvs(
            first_draw_predictions,
            ed_stay_ids,
            base_dataset,
            output_dir,
        )
    return paths


def _write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "scenario", "seed", "permuted_modalities", "n", "positive_n", "negative_n", *METRIC_KEYS,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_draws(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise draws with one row per scenario and metric.

    Delta is baseline minus the mean score. For individual modality permutations,
    empirical_p counts draws at least as high as baseline. A small value indicates
    a drop in performance after shuffling.

    For permute_all, empirical_p counts draws at or below the metric's chance value
    in NULL_SKILL_VALUES. A small value indicates performance above chance despite
    shuffling all content. Add one to each count and draw total to avoid zero p-values.
    """
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)
    baseline_values = {
        metric: float(by_scenario[BASELINE_SCENARIO][0][metric])
        for metric in METRIC_KEYS
    } if BASELINE_SCENARIO in by_scenario else {}

    summary: list[dict[str, Any]] = []
    for scenario, scenario_rows in by_scenario.items():
        permute_all = scenario == PERMUTE_ALL_SCENARIO
        for metric in METRIC_KEYS:
            values = np.array([float(r[metric]) for r in scenario_rows], dtype=float)
            baseline = baseline_values.get(metric, float("nan"))
            mean = float(values.mean())
            if permute_all:
                # Count draws at or below chance when testing for performance above chance
                null_value = NULL_SKILL_VALUES[metric]
                n_tail = int(np.sum(values <= null_value))
            else:
                # Count draws reaching baseline when testing for a drop in performance
                null_value = baseline
                n_tail = int(np.sum(values >= null_value))
            summary.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n_draws": int(values.size),
                    "mean": mean,
                    "std": float(values.std(ddof=0)),
                    "ci_lo": float(np.percentile(values, 2.5)),
                    "ci_hi": float(np.percentile(values, 97.5)),
                    "baseline": baseline,
                    "null_value": null_value,
                    "delta": baseline - mean,
                    "empirical_p": (n_tail + 1) / (values.size + 1),
                }
            )
    return summary


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "scenario", "metric", "n_draws", "mean", "std", "ci_lo", "ci_hi", "baseline", "null_value",
        "delta", "empirical_p",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summarise_draws(rows))


def _write_prediction_csvs(
    first_draw_predictions: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ed_stay_ids: np.ndarray,
    base_dataset: MultimodalSplitDataset,
    output_dir: Path,
) -> list[Path]:
    """Write the first draw for each scenario in the shared prediction CSV format."""
    labels = np.array(
        [int(base_dataset[i]["label"].item()) for i in range(len(base_dataset))],
        dtype=np.int64,
    )
    paths: list[Path] = []
    for scenario, (probabilities, predictions) in first_draw_predictions.items():
        path = output_dir / f"{scenario}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["ed_stay_id", "label", "probability", "prediction"])
            for stay_id, label, probability, prediction in zip(
                ed_stay_ids,
                labels,
                probabilities.tolist(),
                predictions.tolist(),
                strict=True,
            ):
                writer.writerow(
                    [int(stay_id), int(label), float(probability), int(prediction)]
                )
        paths.append(path)
    return paths


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    draws: int = DEFAULT_DRAWS,
    base_seed: int = DEFAULT_BASE_SEED,
    emit_predictions: bool = False,
) -> dict[str, Path]:
    """Run the permutation experiment for one checkpoint and return output paths.

    Read config.yaml from the run directory by default. Write outputs below
    output_dir/modality_permutation.
    """
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    config_path = (
        run_dir / "config.yaml" if config_path is None else Path(config_path)
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    model = MMModel.load_from_checkpoint(
        str(checkpoint), map_location=_usable_device()
    )
    base_dataset = get_datasets(cfg, splits=("test",))["test"]
    scenarios = permutation_scenarios_for_active_modalities(
        base_dataset.modality_names
    )

    output_dir = run_dir if output_dir is None else Path(output_dir)
    return run_modality_permutation(
        model,
        cfg,
        output_dir / "modality_permutation",
        base_dataset=base_dataset,
        scenarios=scenarios,
        draws=draws,
        base_seed=base_seed,
        emit_predictions=emit_predictions,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.checkpoint is not None:
        paths = evaluate_checkpoint(
            args.checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
            draws=args.n_permutations,
            base_seed=args.base_seed,
            emit_predictions=args.emit_predictions,
        )
        run_dir = Path(args.checkpoint).parent
        output_dir = run_dir if args.output_dir is None else Path(args.output_dir)
        print(
            "Modality permutation outputs stored at: "
            f"{output_dir / 'modality_permutation'}"
        )
        for path in paths.values():
            print(path)
        return

    checkpoints = discover_checkpoints(args.path)
    if not checkpoints:
        print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {args.path}.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {args.path}.")
    failures: list[tuple[Path, str]] = []
    evaluated = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        existing = (
            run_dir / "modality_permutation" / "test_modality_permutation_summary.csv"
        )
        if args.skip_existing and existing.exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
        try:
            evaluate_checkpoint(
                checkpoint,
                config_path=args.config,
                draws=args.n_permutations,
                base_seed=args.base_seed,
                emit_predictions=args.emit_predictions,
            )
            evaluated += 1
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not args.continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(f"Done. Evaluated {evaluated}/{len(checkpoints)} run(s).")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


if __name__ == "__main__":
    main()
