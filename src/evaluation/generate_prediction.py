"""Generate validation and test predictions from saved checkpoints.

Write prediction CSVs in the same format as training.

Usage:
    python -m src.evaluation.generate_prediction --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.generate_prediction --path <runs-root>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import (
    CHECKPOINT_SEARCH_ROOTS,
    DEFAULT_CHECKPOINT_NAME,
    _usable_device,
    discover_checkpoints,
    load_config,
    predict_batches,
    resolve_run_dir_by_id_via_checkpoint,
)
from src.train import write_predictions_csv
from src.training.lightning import MMModel

VALID_SPLITS = ("val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validation and/or test predictions for a checkpoint.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "Path to the YAML config used to build the dataloaders. Defaults "
            "to config.yaml in the checkpoint's run directory (its parent "
            "folder), which training writes there. With --path this is a "
            "fallback used only for runs missing their own config.yaml."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="Path to a trained Lightning checkpoint.",
    )
    source.add_argument(
        "--run-id",
        dest="run_id",
        help=(
            "Training run_id to score instead of a checkpoint path. The run "
            f"directory (holding {DEFAULT_CHECKPOINT_NAME} and config.yaml) is located "
            "by searching "
            + " and ".join(str(root) for root in CHECKPOINT_SEARCH_ROOTS)
            + "."
        ),
    )
    source.add_argument(
        "--path",
        default=None,
        help=(
            "Root directory to search recursively for run checkpoints. Every "
            "checkpoint_best.ckpt file found under it is scored in place, "
            "writing {split}_predictions.csv to each run's own directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=(
            "Directory where {split}_predictions.csv files are written. "
            "Defaults to the checkpoint's run directory (its parent folder). "
            "Ignored in --path mode, where each run writes to its own directory."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --path mode, skip runs that already have {split}_predictions.csv "
            "for every requested split (useful to resume a batch)."
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
    parser.add_argument(
        "--split",
        dest="splits",
        choices=VALID_SPLITS,
        nargs="+",
        default=list(VALID_SPLITS),
        help="Which split(s) to score. Defaults to both val and test.",
    )
    return parser.parse_args(argv)


# Keep the training writer and eval_utils prediction columns aligned
def generate_predictions(
    model: MMModel,
    dataloaders: Mapping[str, "object"],
    output_dir: str | Path,
    splits: tuple[str, ...] = VALID_SPLITS,
) -> dict[str, Path]:
    """Write ``{split}_predictions.csv`` for each requested split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in splits:
        prediction_batches = predict_batches(model, dataloaders[split])
        predictions_path = output_dir / f"{split}_predictions.csv"
        write_predictions_csv(prediction_batches, predictions_path)
        paths[f"{split}_predictions_path"] = predictions_path
    return paths


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    splits: tuple[str, ...] = VALID_SPLITS,
) -> dict[str, Path]:
    """Generate predictions for one checkpoint and return the output paths.

    Read config.yaml from the run directory and write predictions there unless
    config_path or output_dir is supplied.
    """
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    config_path = (
        run_dir / "config.yaml" if config_path is None else Path(config_path)
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    # Check that CUDA can start before loading the checkpoint
    map_location = _usable_device()
    model = MMModel.load_from_checkpoint(str(checkpoint), map_location=map_location)
    output_dir = run_dir if output_dir is None else Path(output_dir)
    dataloaders = get_dataloaders(cfg, splits=splits)
    return generate_predictions(model, dataloaders, output_dir, splits)


def _predictions_exist(output_dir: str | Path, splits: tuple[str, ...]) -> bool:
    """True when ``{split}_predictions.csv`` already exists for every split."""
    output_dir = Path(output_dir)
    return all((output_dir / f"{split}_predictions.csv").exists() for split in splits)


def run_batch(
    root: str | Path,
    splits: tuple[str, ...],
    *,
    config_override: Path | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> None:
    """Write predictions for each checkpoint found under root.

    Use config_override only when a run has no config.yaml of its own.
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
        if skip_existing and _predictions_exist(run_dir, splits):
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
        # A CLI --config is only a fallback for runs without their own config.yaml
        per_run_config = (
            config_override
            if config_override is not None and not (run_dir / "config.yaml").exists()
            else None
        )
        try:
            evaluate_checkpoint(checkpoint, config_path=per_run_config, splits=splits)
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    splits = tuple(dict.fromkeys(args.splits))

    if args.path is not None:
        run_batch(
            args.path,
            splits,
            config_override=Path(args.config) if args.config else None,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
        )
        return

    if args.run_id is not None:
        run_dir = resolve_run_dir_by_id_via_checkpoint(args.run_id)
        checkpoint = run_dir / DEFAULT_CHECKPOINT_NAME
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    else:
        checkpoint = Path(args.checkpoint)
    paths = evaluate_checkpoint(
        checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        splits=splits,
    )
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
