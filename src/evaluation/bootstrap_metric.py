"""Estimate confidence intervals for saved validation and test predictions.

Generate predictions first with python -m src.evaluation.generate_prediction.

Usage:
    python -m src.evaluation.bootstrap_metric --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.bootstrap_metric --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    bias_corrected_ci,
    discover_checkpoints,
    read_prediction_arrays,
)
from src.evaluation.recompute_metric import read_existing_metrics_row
from src.metrics import build_binary_metrics

METRIC_NAMES = ("auroc", "recall", "specificity", "mcc", "ba")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrapped 95% CIs (BCa) for the five build_binary_metrics and "
            "write bootstrap_metrics.csv next to each run's prediction CSVs."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="Checkpoint in the run directory holding the prediction CSVs.",
    )
    source.add_argument(
        "--path",
        help=(
            "Root directory to search recursively for run checkpoints. Every "
            "run holding a checkpoint_best.ckpt file is bootstrapped in place "
            "(runs without prediction CSVs are skipped)."
        ),
    )
    parser.add_argument(
        "--val-predictions",
        default=None,
        help="Override val predictions path. Only valid with --checkpoint. "
        "Defaults to <run_dir>/val_predictions.csv.",
    )
    parser.add_argument(
        "--test-predictions",
        default=None,
        help="Override test predictions path. Only valid with --checkpoint. "
        "Defaults to <run_dir>/test_predictions.csv.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --path mode, skip runs that already have bootstrap_metrics.csv "
            "(useful to resume a batch)."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "In --path mode, keep going when a run fails and report the "
            "failures at the end instead of aborting the whole batch."
        ),
    )
    args = parser.parse_args(argv)
    if args.path is not None and (args.val_predictions or args.test_predictions):
        parser.error("--val-predictions/--test-predictions require --checkpoint.")
    return args


def _compute_metrics(collection, y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Score one sample using the training metric collection.

    Pass probabilities directly to the metrics, which apply their own thresholds.
    Mark AUROC as NaN when only one class is present so it is excluded from the interval.
    """
    collection.reset()
    preds = torch.as_tensor(prob, dtype=torch.float32)
    target = torch.as_tensor(y_true, dtype=torch.int64)
    collection.update(preds, target)
    out = {name: float(value.item()) for name, value in collection.compute().items()}

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        out["auroc"] = float("nan")  # AUROC undefined on a single-class resample
    return out


def bootstrap_metric_cis(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    metrics: tuple[str, ...] = METRIC_NAMES,
) -> dict[str, dict[str, float]]:
    """Return point estimates, bootstrap means, confidence bounds and valid sample counts.

    Sample rows with replacement using a fixed seed. Calculate bias-corrected and accelerated
    (BCa) 95% confidence bounds from the valid samples for each metric.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collection = build_binary_metrics()

    point = _compute_metrics(collection, y_true, prob)
    boot = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample all rows with replacement
        sample = _compute_metrics(collection, y_true[idx], prob[idx])
        for m in metrics:
            if not np.isnan(sample[m]):
                boot[m].append(sample[m])

    results: dict[str, dict[str, float]] = {}
    for m in metrics:
        samples = boot[m]
        if len(samples) < 2:
            results[m] = {
                "point": point[m],
                "mean": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_valid": len(samples),
            }
            continue
        lower, upper = bias_corrected_ci(samples, point[m])
        results[m] = {
            "point": float(point[m]),
            "mean": float(np.mean(samples)),
            "ci_lower": lower,
            "ci_upper": upper,
            "n_valid": len(samples),
        }
    return results


def _format_table(split: str, res: dict[str, dict[str, float]]) -> str:
    lines = [
        f"[{split}]  metric        point     mean      95% CI (BCa)",
        "-" * 58,
    ]
    for m in METRIC_NAMES:
        if m not in res:
            continue
        r = res[m]
        lines.append(
            f"          {m:<12} {r['point']:>7.4f}  {r['mean']:>7.4f}  "
            f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        )
    return "\n".join(lines)


METADATA_FIELDS = ("run_id", "model", "outcome", "bundle")
OUTPUT_FIELDS = (
    *METADATA_FIELDS, "split", "metric", "point", "mean", "ci_lower", "ci_upper", "n_valid",
)


def _resolve_metadata(existing_row: dict[str, str] | None, run_dir: Path) -> dict[str, str]:
    """Read run identifiers from metrics.csv, using the folder name when run_id is missing."""
    defaults = {field: "" for field in METADATA_FIELDS}
    defaults["run_id"] = run_dir.name
    if not existing_row:
        return defaults
    return {field: existing_row.get(field) or defaults[field] for field in METADATA_FIELDS}


def bootstrap_run(
    run_dir: Path,
    *,
    val_path: Path | None = None,
    test_path: Path | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[Path, dict[str, dict[str, dict[str, float]]]]:
    """Write bootstrap_metrics.csv from a run's validation and test predictions.

    Read {split}_predictions.csv by default. Return the output path and results for
    each available split.
    """
    val_path = val_path or run_dir / "val_predictions.csv"
    test_path = test_path or run_dir / "test_predictions.csv"
    output_path = run_dir / "bootstrap_metrics.csv"

    candidates = [("val", val_path), ("test", test_path)]
    present = [(split, path) for split, path in candidates if path.exists()]
    if not present:
        raise FileNotFoundError(
            f"No prediction CSVs found in {run_dir} "
            f"(looked for {val_path.name}, {test_path.name})."
        )

    metadata = _resolve_metadata(read_existing_metrics_row(run_dir / "metrics.csv"), run_dir)

    results: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    for split, path in present:
        y_true, prob = read_prediction_arrays(path)
        res = bootstrap_metric_cis(y_true, prob, n_boot=n_boot, seed=seed)
        results[split] = res
        for metric in METRIC_NAMES:
            rows.append({**metadata, "split": split, "metric": metric, **res[metric]})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path, results


def _has_predictions(run_dir: Path) -> bool:
    """True when the run has at least one ``{split}_predictions.csv`` to bootstrap."""
    return (run_dir / "val_predictions.csv").exists() or (
        run_dir / "test_predictions.csv"
    ).exists()


def run_batch(
    root: str | Path,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> None:
    """Write bootstrap metrics for each checkpoint found under root.

    Skip runs without prediction CSVs.
    """
    checkpoints = discover_checkpoints(root)
    if not checkpoints:
        print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {root}.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {root}.")
    failures: list[tuple[Path, str]] = []
    evaluated = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        if skip_existing and (run_dir / "bootstrap_metrics.csv").exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        if not _has_predictions(run_dir):
            print(f"[{index}/{len(checkpoints)}] skip (no predictions): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] bootstrapping: {run_dir}")
        try:
            output_path, results = bootstrap_run(run_dir, n_boot=n_boot, seed=seed)
            evaluated += 1
            print(f"    Saved at: {output_path}: {_summary(results)}")
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(f"Done. Bootstrapped {evaluated}/{len(checkpoints)} run(s).")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


def _summary(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Format MCC estimates and confidence bounds for each available split."""
    parts = []
    for split in ("val", "test"):
        res = results.get(split)
        if not res:
            continue
        mcc = res["mcc"]
        parts.append(
            f"{split}_mcc={mcc['point']:.4f} [{mcc['ci_lower']:.4f}, {mcc['ci_upper']:.4f}]"
        )
    return " ".join(parts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.path is not None:
        run_batch(
            args.path,
            n_boot=args.n_boot,
            seed=args.seed,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
        )
        return
    run_dir = Path(args.checkpoint).parent
    output_path, results = bootstrap_run(
        run_dir,
        val_path=Path(args.val_predictions) if args.val_predictions else None,
        test_path=Path(args.test_predictions) if args.test_predictions else None,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    for split in ("val", "test"):
        if split in results:
            print(_format_table(split, results[split]))
            print()
    print(f"Saved at: {output_path}: {_summary(results)}")


if __name__ == "__main__":
    main()
