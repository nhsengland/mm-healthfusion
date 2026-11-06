"""Share helpers for finding runs, reading predictions and evaluating models.

Use these helpers through the evaluation commands.

Usage:
    python -m src.evaluation.generate_prediction --checkpoint <run>/checkpoint_best.ckpt
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader

from src.metrics import build_binary_metrics

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
# Search locations for checkpoints
CHECKPOINT_SEARCH_ROOTS = (WORKSPACE_ROOT / "outputs", WORKSPACE_ROOT / "logs")
DEFAULT_CHECKPOINT_NAME = "checkpoint_best.ckpt"
CONFIG_NAME = "config.yaml"


def resolve_run(path: str | Path) -> tuple[str, str]:
    """Return the checkpoint and configuration paths in a run directory."""
    run = Path(path)
    ckpt = run / DEFAULT_CHECKPOINT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(f"{DEFAULT_CHECKPOINT_NAME} not found under {run}.")
    return str(ckpt), str(run / CONFIG_NAME)


def discover_checkpoints(root: str | Path) -> list[Path]:
    """Find matching checkpoint files and sort them by path."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Search path not found: {root}")
    return sorted(root.rglob(DEFAULT_CHECKPOINT_NAME))


def resolve_run_dir_by_id_via_checkpoint(
    run_id: str,
    search_roots: tuple[Path, ...] = CHECKPOINT_SEARCH_ROOTS,
) -> Path:
    """Find a run that contains the requested checkpoint.

    Match the manifest ID, or use the _<run_id> folder suffix if no ID can be read.
    """
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint in root.rglob(DEFAULT_CHECKPOINT_NAME):
            run_dir = checkpoint.parent
            manifest_path = run_dir / "manifest.json"
            manifest_id = None
            if manifest_path.exists():
                try:
                    manifest_id = json.loads(manifest_path.read_text()).get("run_id")
                except (json.JSONDecodeError, OSError):
                    manifest_id = None
            if manifest_id == run_id or (
                manifest_id is None and run_dir.name.endswith(f"_{run_id}")
            ):
                candidates.append(run_dir)

    if not candidates:
        roots = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"No run directory with run_id {run_id!r} found under: {roots}"
        )
    if len(candidates) > 1:
        found = "\n  ".join(str(path) for path in sorted(candidates))
        raise ValueError(
            f"Multiple run directories match run_id {run_id!r}. Pass --checkpoint "
            f"to disambiguate:\n  {found}"
        )
    return candidates[0]


def bias_corrected_ci(
    bootstrap_samples: list[float] | np.ndarray, observed_value: float
) -> tuple[float, float]:
    """Calculate adjusted 95% confidence bounds from bootstrap samples.

    Estimate bias from samples below the observed value and acceleration from
    means calculated with each bootstrap sample left out.
    """
    sorted_samples = np.sort(np.asarray(bootstrap_samples, dtype=np.float64))
    z0 = norm.ppf(
        (np.sum(sorted_samples < observed_value) + 0.5) / len(sorted_samples)
    )
    n = len(sorted_samples)
    jackknife = (sorted_samples.sum() - sorted_samples) / (n - 1)
    deviations = np.mean(jackknife) - jackknife
    denom = 6 * (np.sum(deviations ** 2) ** 1.5)
    a = np.sum(deviations ** 3) / denom if denom else 0.0
    alpha = np.array([0.025, 0.975])
    adjusted = norm.cdf(z0 + (z0 + norm.ppf(alpha)) / (1 - a * (z0 + norm.ppf(alpha))))
    if np.isnan(adjusted[0]) or np.isnan(adjusted[1]):
        adjusted = alpha
    lower, upper = np.percentile(sorted_samples, adjusted * 100)
    return float(lower), float(upper)


# Keep these columns aligned with src.train.write_predictions_csv
PREDICTION_ID_COLUMNS = ("hadm_id", "ed_stay_id")
PREDICTION_VALUE_COLUMNS = ("label", "probability", "prediction")


def prediction_identifier_column(frame: pd.DataFrame) -> str:
    """Return the only admission or ED stay ID column, raising an error otherwise."""
    columns = [column for column in PREDICTION_ID_COLUMNS if column in frame.columns]
    if len(columns) != 1:
        raise ValueError(
            "Predictions must contain exactly one of hadm_id or ed_stay_id."
        )
    return columns[0]


def predictions_path(run_dir: str | Path, split: str) -> Path:
    """Build the path to the saved predictions for a split."""
    return Path(run_dir) / f"{split}_predictions.csv"


def read_predictions(path: str | Path) -> pd.DataFrame:
    """Read prediction columns from a CSV and check the required fields.

    Use hadm_id for hospitalisation and ed_stay_id for deterioration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    frame = pd.read_csv(path)
    id_column = prediction_identifier_column(frame)
    missing = [column for column in PREDICTION_VALUE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame = frame[[id_column, *PREDICTION_VALUE_COLUMNS]].copy()
    frame[id_column] = frame[id_column].astype("int64")
    return frame


def read_prediction_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return labels as int64 and probabilities as float64 arrays."""
    frame = read_predictions(path)
    return (
        frame["label"].to_numpy(dtype=np.int64),
        frame["probability"].to_numpy(dtype=np.float64),
    )


def _warn_cpu_fallback(reason: str) -> None:
    """Warn that CPU results may differ from GPU results."""
    warnings.warn(
        f"Evaluation is falling back to CPU ({reason}). "
        "Results may differ from GPU evaluation.",
        RuntimeWarning,
        stacklevel=3,
    )


def _usable_device() -> str:
    """Use CUDA if it starts successfully, otherwise warn and use the CPU."""
    if not torch.cuda.is_available():
        _warn_cpu_fallback("CUDA is not available")
        return "cpu"
    try:
        torch.cuda.current_device()
    except Exception as error:  # Warn and fall back to CPU
        _warn_cpu_fallback(f"CUDA failed to initialise ({error})")
        return "cpu"
    return "cuda"


def usable_accelerator() -> str:
    """Choose gpu or cpu for Lightning using the same CUDA check."""
    return "gpu" if _usable_device() == "cuda" else "cpu"


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def predict_batches(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> list[dict[str, torch.Tensor]]:
    """Collect prediction tensors on the CPU without tracking gradients."""
    device = _model_device(model)
    was_training = model.training
    model.eval()
    prediction_batches: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_batch_to_device(batch, device)
            prediction = model.predict_step(batch, batch_idx)
            prediction_batches.append(
                {
                    key: value.detach().cpu()
                    for key, value in prediction.items()
                    if isinstance(value, torch.Tensor)
                }
            )
    if was_training:
        model.train()
    return prediction_batches


def _scenario_torchmetrics(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Score a scenario with the training metrics on the available device."""
    device = _usable_device()
    metrics = build_binary_metrics().to(device)
    metrics.update(
        probabilities.reshape(-1).to(device),
        targets.reshape(-1).to(device),
    )
    return {key: float(value) for key, value in metrics.compute().items()}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML using the training loader, including inherited settings."""
    # Keep this local because training imports the expert-output helpers
    from src.train import load_config as train_load_config

    return train_load_config(path)
