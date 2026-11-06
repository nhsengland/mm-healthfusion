"""Recalculate training, validation and test metrics from saved checkpoints.

Write recompute_metrics.csv beside the original metrics.csv.

Usage:
    python -m src.evaluation.recompute_metric --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.recompute_metric --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import lightning as L
from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    discover_checkpoints,
    usable_accelerator,
)
from src.train import (
    SUMMARY_STAGES,
    build_grouped_metric_summary,
    load_config,
    merge_grouped_metric_summaries,
    resolve_bundle_name,
    write_metrics_csv,
)
from src.training.lightning import MMModel


METADATA_FIELDS = ("run_id", "model", "outcome", "bundle", "checkpoint_path")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute train/val/test on the best checkpoint and write "
            "recompute_metrics.csv next to the original metrics.csv."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained Lightning checkpoint.",
    )
    source.add_argument(
        "--path",
        default=None,
        help=(
            "Root directory to search recursively for run checkpoints. Every "
            "checkpoint_best.ckpt file found under it is recomputed in place."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Override config path. In --path mode it is used only when a run lacks "
            "config.yaml. Defaults to <run_dir>/config.yaml."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --path mode, skip runs that already have recompute_metrics.csv.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In --path mode, continue after a run fails and report failures at the end.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Lightning accelerator for the eval passes. Defaults to the shared "
        "eval device resolver (usable_accelerator -> GPU when CUDA initialises, "
        "else a loud CPU fall-back), matching generate_prediction / "
        "modality_ablation / output_expert. The forward is not bit-identical "
        "CPU-vs-GPU for masked/missing-modality rows, so GPU is required to match "
        "train.py's reported numbers. Pass an explicit value (e.g. cpu) to "
        "override.",
    )
    args = parser.parse_args(argv)
    if args.skip_existing or args.continue_on_error:
        if args.path is None:
            parser.error("--skip-existing/--continue-on-error require --path.")
    return args


def read_existing_metrics_row(metrics_path: Path) -> dict[str, str] | None:
    """Return the single data row of an existing metrics.csv, if present."""
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return rows[0] if rows else None


def _relabel_stage(
    results: list[dict[str, Any]] | None,
    from_stage: str,
    to_stage: str,
) -> dict[str, Any]:
    """Rename metric prefixes in the first results row.

    Lightning validation returns val_* keys even for the training split. Rename
    these to train_* when evaluating training data.
    """
    if not results:
        return {}
    prefix = f"{from_stage}_"
    return {
        (f"{to_stage}_{key[len(prefix):]}" if key.startswith(prefix) else key): value
        for key, value in results[0].items()
    }


def _coerce_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def resolve_metadata(
    existing_row: dict[str, str] | None,
    cfg: dict[str, Any],
    run_dir: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Build metric identifiers, preferring values from the existing metrics row."""
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fusion_method = str(
        model_cfg.get("fusion_method", model_cfg.get("name", "concat"))
    ).lower()
    outcome = str(data_cfg.get("outcome", cfg.get("outcome", "unknown")))
    defaults = {
        "run_id": run_dir.name,
        "model": fusion_method,
        "outcome": outcome,
        "bundle": resolve_bundle_name(data_cfg),
        "checkpoint_path": str(checkpoint_path),
    }
    if not existing_row:
        return defaults
    # Keep training identifiers and record the checkpoint actually scored
    resolved = {field: existing_row.get(field) or defaults[field] for field in METADATA_FIELDS}
    resolved["checkpoint_path"] = str(checkpoint_path)
    return resolved


def recompute_run(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    checkpoint_path: Path | None = None,
    accelerator: str | None = None,
) -> Path:
    """Recalculate all three splits and write recompute_metrics.csv.

    Use the shared device check unless accelerator is supplied. CPU and GPU
    results can differ, so use the training device for comparisons.
    """
    accelerator = accelerator or usable_accelerator()
    config_path = config_path or run_dir / "config.yaml"
    checkpoint_path = checkpoint_path or run_dir / "checkpoint_best.ckpt"
    # Keep the original metrics.csv and write the recomputed values separately
    source_metrics_path = run_dir / "metrics.csv"
    output_metrics_path = run_dir / "recompute_metrics.csv"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)
    # Keep every training row in a fixed batch order for models that depend on batch neighbours
    data_cfg = cfg.setdefault("data", {})
    data_cfg["drop_last"] = False
    data_cfg["shuffle_train"] = False
    dataloaders = get_dataloaders(cfg, splits=("train", "val", "test"))
    model = MMModel.load_from_checkpoint(checkpoint_path, map_location="cpu")

    trainer = L.Trainer(
        accelerator=accelerator,
        precision="32-true",
        logger=False,
        enable_progress_bar=False,
    )
    # Validate training data in evaluation mode, then rename val_* keys to train_*
    train_results = trainer.validate(model=model, dataloaders=dataloaders["train"])
    val_results = trainer.validate(model=model, dataloaders=dataloaders["val"])
    test_results = trainer.test(model=model, dataloaders=dataloaders["test"])

    existing_row = read_existing_metrics_row(source_metrics_path)
    grouped_metrics = merge_grouped_metric_summaries(
        build_grouped_metric_summary(_relabel_stage(train_results, "val", "train")),
        build_grouped_metric_summary(val_results[0] if val_results else {}),
        build_grouped_metric_summary(test_results[0] if test_results else {}),
    )
    metadata = resolve_metadata(existing_row, cfg, run_dir, checkpoint_path)
    write_metrics_csv(output_metrics_path, metadata, grouped_metrics)
    return output_metrics_path


def run_batch(
    root: str | Path,
    *,
    config_override: Path | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
    accelerator: str | None = None,
) -> None:
    """Recompute metrics for every checkpoint found under ``root``."""
    checkpoints = discover_checkpoints(root)
    if not checkpoints:
        print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {root}.")
        return

    failures: list[tuple[Path, str]] = []
    evaluated = 0
    print(f"Found {len(checkpoints)} checkpoint(s) under {root}.")
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        output_path = run_dir / "recompute_metrics.csv"
        if skip_existing and output_path.exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
        per_run_config = (
            config_override
            if config_override is not None and not (run_dir / "config.yaml").exists()
            else None
        )
        try:
            recompute_run(
                run_dir,
                config_path=per_run_config,
                checkpoint_path=checkpoint,
                accelerator=accelerator,
            )
            evaluated += 1
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(f"Done. Evaluated {evaluated}/{len(checkpoints)} run(s).")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


def _format_stage(grouped: dict[str, dict[str, Any]], stage: str) -> str:
    stage_metrics = grouped.get(stage, {})
    mcc = stage_metrics.get("mcc")
    return f"{stage}_mcc={mcc:.4f}" if isinstance(mcc, float) else f"{stage}_mcc=n/a"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.path is not None:
        run_batch(
            args.path,
            config_override=Path(args.config) if args.config else None,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
            accelerator=args.accelerator,
        )
        return

    checkpoint = Path(args.checkpoint)
    metrics_path = recompute_run(
        checkpoint.parent,
        config_path=Path(args.config) if args.config else None,
        checkpoint_path=checkpoint,
        accelerator=args.accelerator,
    )
    row = read_existing_metrics_row(metrics_path) or {}
    grouped = build_grouped_metric_summary(
        {key: _coerce_float(value) for key, value in row.items()}
    )
    summary = " ".join(_format_stage(grouped, stage) for stage in SUMMARY_STAGES)
    print(f"Saved at: {metrics_path}: {summary}")


if __name__ == "__main__":
    main()
