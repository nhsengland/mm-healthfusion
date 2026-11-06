"""Train a binary classifier, score its best checkpoint and save the run outputs.

Save configuration, checkpoints, predictions, metrics and optional model-specific reports together.
Run from the repository root, for example::

    python -m src.train --config config/concat.yaml --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data.datasets import modality_prefix
from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import _usable_device
from src.evaluation.modality_ablation import (
    resolve_modality_ablation_scenarios,
    run_modality_ablation,
)
from src.evaluation.output_expert import (
    EXPERT_METHODS,
    collect_modality_availability_batches,
    moe_expert_names,
    resolve_expert_output_dir,
    shared_expert_pool_modalities,
    uses_expert_modality_availability,
    write_moe_expert_outputs,
)
from src.training.lightning import MMModel

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "logs"
DEFAULT_RUNS_DIR = DEFAULT_LOG_DIR / "runs"
DEFAULT_WANDB_DIR = DEFAULT_LOG_DIR
SUMMARY_STAGES = ("train", "val", "test")
SUMMARY_METRICS = ("loss", "auroc", "recall", "specificity", "mcc", "ba")
TRAIN_EPOCH_LOG_METRICS = ("train_loss", "train_aux_loss", "train_mcc", "train_ba")
VAL_EPOCH_LOG_METRICS = ("val_loss", "val_aux_loss", "val_mcc", "val_ba")
M3CARE_METHODS = {"m3care", "mtand_m3care"}


def parse_args() -> argparse.Namespace:
    """Read the training configuration path, overrides and logging options."""
    parser = argparse.ArgumentParser(description="Train an MM HealthFusion model.")
    parser.add_argument(
        "--config", "-c", default="config/concat.yaml", help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override train.epochs from the config.",
    )
    parser.add_argument(
        "--set", dest="config_overrides", action="append", default=None, metavar="key=value",
        help=(
            "Override a config value with a dotted key. Can be repeated, "
            "for example --set train.learning_rate=0.0005."
        ),
    )
    parser.add_argument(
        "--output-root", default=None,
        help=(
            "Optional parent directory for run outputs. Defaults to mirrored "
            f"{DEFAULT_LOG_DIR} paths or {DEFAULT_RUNS_DIR}/{{bundle_name}}."
        ),
    )
    parser.add_argument(
        "--run-id", default=None,
        help=(
            "Optional run identifier. Defaults to a W&B-generated id when "
            "--wandb is enabled, otherwise Greenwich time YYYYMMDDTHHMMSS."
        ),
    )
    parser.add_argument(
        "--no-class-weights", dest="use_class_weights", action="store_false", default=True,
        help="Disable positive-class weighting in BCE loss (enabled by default).",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Log metrics and artifacts to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project", default="mm-healthfusion", help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity", default=os.environ.get("WANDB_ENTITY", "mm-healthfusion"),
        help=(
            "Weights & Biases entity or team. Defaults to WANDB_ENTITY if set, "
            "otherwise mm-healthfusion."
        ),
    )
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Optional W&B display name. If omitted, defaults to "
            "<outcome>_<fusion_method>_<run_id>. Local artefacts use the "
            "run_id in their folder name."
        ),
    )
    parser.add_argument(
        "--wandb-init-timeout", type=int,
        default=int(os.environ.get("WANDB_INIT_TIMEOUT", 120)),
        help=(
            "W&B run initialisation timeout in seconds. Defaults to "
            "WANDB_INIT_TIMEOUT if set, otherwise 120."
        ),
    )
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested overrides into the base dictionary and return it."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML settings and apply them over any configuration named by extends."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file) or {}

    parent_path = cfg.pop("extends", None)
    if parent_path is None:
        return cfg

    parent_config_path = Path(parent_path)
    if not parent_config_path.is_absolute():
        parent_config_path = config_path.parent / parent_config_path
        if not parent_config_path.exists():
            parent_config_path = config_path.parent.parent / parent_path

    parent_cfg = load_config(parent_config_path)
    return deep_update(parent_cfg, cfg)


def parse_config_override(raw: str) -> tuple[list[str], Any]:
    """Split a dotted key=value override and parse its value as YAML."""
    if "=" not in raw:
        raise ValueError(f"Config override must use key=value syntax: {raw!r}.")

    key, value = raw.split("=", 1)
    if not key:
        raise ValueError(f"Config override key must not be empty: {raw!r}.")

    parts = key.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Config override key has an empty nested part: {raw!r}.")

    return parts, yaml.safe_load(value)


def apply_config_overrides(cfg: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply overrides in order, creating nested dictionaries as needed."""
    if not overrides:
        return cfg

    for raw in overrides:
        parts, value = parse_config_override(raw)
        target = cfg
        for part in parts[:-1]:
            current = target.get(part)
            if not isinstance(current, dict):
                current = {}
                target[part] = current
            target = current
        target[parts[-1]] = value
    return cfg


def slugify(value: object, *, lower: bool = True) -> str:
    """Turn a value into a name containing only letters, digits and underscores."""
    text = str(value).strip()
    if lower:
        text = text.lower()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def resolve_log_parent_path(split_dir: str | Path | None, outcome: str) -> tuple[str, list[str]]:
    """Build the outcome log path and path labels from the bundle directory name."""
    bundle_name = Path(split_dir).name if split_dir is not None else "unknown"
    dir_layers = bundle_name.split("_")
    log_path_main = "/".join(dir_layers[:4])
    if len(dir_layers) > 4:
        log_path_sub = "_".join(dir_layers[4:])
        dir_layers = dir_layers[:4] + [log_path_main]
    else:
        log_path_sub = "base"  # Use base when the bundle name has no preprocessing suffix
        dir_layers.append("base")
    return os.path.join(log_path_main, log_path_sub, outcome), dir_layers


def generate_run_id() -> str:
    """Create a run ID from the current UTC date and time."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def resolve_run_id(explicit_run_id: str | None, *, use_wandb: bool) -> str:
    """Use the supplied ID, otherwise generate one with W&B or the current UTC time."""
    if explicit_run_id:
        return explicit_run_id
    if use_wandb:
        import wandb

        return wandb.util.generate_id()
    return generate_run_id()


def resolve_run_name(
    explicit_run_name: str | None, *, outcome: str, fusion_method: str, run_id: str,
) -> str:
    """Use the supplied display name or combine the outcome, model and run ID."""
    if explicit_run_name:
        return explicit_run_name
    return "_".join([slugify(outcome), slugify(fusion_method), slugify(run_id, lower=False)])


def print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def resolve_bundle_name(data_cfg: Mapping[str, Any]) -> str:
    """Read the bundle name from the first configured data directory."""
    return (
        Path(
            data_cfg.get("split_dir")
            or data_cfg.get("bundle_dir")
            or data_cfg.get("data_dir")
            or ""
        ).name
        or "unknown"
    )


def resolve_run_dir(
    data_cfg: Mapping[str, Any], outcome: str, fusion_method: str, run_id: str | None, *,
    use_wandb: bool, output_root: str | Path | None = None,
) -> tuple[str, Path, list[str]]:
    """Return the run ID, output directory and path labels used for logging."""
    resolved_run_id = resolve_run_id(run_id, use_wandb=use_wandb)
    bundle_name = slugify(resolve_bundle_name(data_cfg))
    source_dir = data_cfg.get("split_dir") or data_cfg.get("bundle_dir") or data_cfg.get("data_dir")
    log_parent_path, path_layers = resolve_log_parent_path(source_dir, outcome)
    run_name = "_".join([slugify(fusion_method), slugify(resolved_run_id, lower=False)])
    if output_root is not None:
        return (
            resolved_run_id,
            (Path(output_root) / log_parent_path / "runs" / run_name),
            path_layers,
        )

    split_dir = data_cfg.get("split_dir")
    if split_dir:
        try:
            log_split_path = DEFAULT_LOG_DIR / log_parent_path
        except ValueError:
            log_split_path = DEFAULT_RUNS_DIR / bundle_name
        return resolved_run_id, (log_split_path / "runs" / run_name), path_layers

    return resolved_run_id, (DEFAULT_RUNS_DIR / log_parent_path / run_name), path_layers


def batch_feature_dims(batch: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Read feature widths in batch modality order, using the standard modalities when absent."""
    modalities = tuple(batch.get("modality_names", ()))
    if not modalities:
        modalities = ("tabular", "ecg", "cxr")
        if "ts_feats" in batch:
            modalities = (*modalities, "ts")
    return {
        modality: int(batch[f"{modality_prefix(modality)}_feats"].shape[-1])
        for modality in modalities
    }


def compute_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """Divide the negative label count by the positive count, returning one if none are positive."""
    targets = targets.to(dtype=torch.float32)
    positives = targets.sum()
    if positives <= 0.0:
        return torch.ones((), dtype=targets.dtype, device=targets.device)
    negatives = targets.numel() - positives
    return negatives / positives


def compute_train_pos_weight(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Collect training labels and calculate the positive-class loss weight."""
    labels: list[torch.Tensor] = []
    for batch in dataloader:
        labels.append(batch["label"])

    return compute_class_weights(torch.cat(labels, dim=0))


def write_predictions_csv(
    prediction_batches: list[dict[str, torch.Tensor]], output_path: str | Path,
) -> None:
    """Write identifiers, labels and predictions in batch and row order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identifier_column = next(
        (
            column
            for column in ("hadm_id", "ed_stay_id")
            if all(column in batch for batch in prediction_batches)
        ),
        None,
    )
    if identifier_column is None:
        raise ValueError("Prediction batches must consistently contain hadm_id or ed_stay_id.")

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=[identifier_column, "label", "probability", "prediction"],
        )
        writer.writeheader()

        for batch in prediction_batches:
            labels = batch["label"]
            probabilities = batch["probability"]
            predictions = batch["prediction"]
            identifiers = batch[identifier_column]

            for index in range(labels.shape[0]):
                writer.writerow(
                    {
                        identifier_column: int(identifiers[index].item()),
                        "label": int(labels[index].item()),
                        "probability": float(probabilities[index].item()),
                        "prediction": int(predictions[index].item()),
                    }
                )


def write_m3care_imputation_attentions_csv(
    prediction_batches: list[dict[str, torch.Tensor]], imputation_attentions_path: str | Path,
    modality_names: tuple | list,
) -> Path | None:
    """Write imputed and original attention values in the requested modality order."""
    imputation_attentions_path = Path(imputation_attentions_path)

    if "imputation_attentions" not in prediction_batches[0]:
        print("Imputation attentions are not available in the model outputs.")
        return None

    imputation_attentions_path.parent.mkdir(parents=True, exist_ok=True)
    modality_names = list(modality_names)
    keys = list(prediction_batches[0].get("imputation_attentions").keys())
    if set(modality_names) != set(keys):
        raise KeyError(
            f"The output modality set ({keys}) does not match the selected modality set ({modality_names}).",
        )

    imputation_fieldnames = []
    for m in modality_names:
        imputation_fieldnames.extend([f"{m}_imputed", f"{m}_original"])

    rows: list[dict[str, float]] = []

    for batch in prediction_batches:
        imputed_attentions = batch.get("imputation_attentions")

        if not imputed_attentions:
            continue

        per_modality: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        batch_size: int | None = None

        for modality in modality_names:
            if modality not in imputed_attentions:
                raise KeyError(f"Missing {modality!r} in imputation_attentions.")

            values = imputed_attentions[modality]
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(
                    f"Expected imputation_attentions[{modality!r}] to be [imputed, original].",
                )

            imputed = values[0].detach().cpu().reshape(-1)
            original = values[1].detach().cpu().reshape(-1)

            if batch_size is None:
                batch_size = int(imputed.shape[0])
            elif int(imputed.shape[0]) != batch_size:
                raise ValueError("All modality attention tensors must have the same batch size.")

            per_modality[modality] = (imputed, original)

        for index in range(batch_size or 0):
            row: dict[str, float] = {}
            for modality in modality_names:
                imputed, original = per_modality[modality]
                row[f"{modality}_imputed"] = float(imputed[index].item())
                row[f"{modality}_original"] = float(original[index].item())
            rows.append(row)

    with imputation_attentions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=imputation_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"M3Care imputation attentions stored at: {imputation_attentions_path}")

    return imputation_attentions_path


def write_m3care_fusion_attentions_csv(
    prediction_batches: list[dict[str, torch.Tensor]], fusion_attentions_path: str | Path,
    modality_names: tuple | list,
) -> Path | None:
    """Write fusion attention by stage and output modality order, averaging attention heads."""
    if "fusion_attentions" not in prediction_batches[0]:
        print("Fusion attentions are not available in the model outputs.")
        return None

    fusion_attentions_path = Path(fusion_attentions_path)
    fusion_attentions_path.parent.mkdir(parents=True, exist_ok=True)
    modality_names = list(modality_names)
    modalities = list(prediction_batches[0].get("fusion_attentions").get("modalities"))
    if set(modality_names) != set(modalities):
        raise KeyError(
            f"The output modality set ({modalities}) does not match the selected modality set ({modality_names}).",
        )

    stage_names = ("stage_1", "stage_2")
    fusion_fieldnames = [
        f"{stage}_{query}_to_{key}"
        for stage in stage_names
        for query in modalities
        for key in modalities
    ]
    rows: list[dict[str, float]] = []

    for batch in prediction_batches:
        attentions = batch.get("fusion_attentions")
        if not attentions:
            continue

        stage_tensors: dict[str, torch.Tensor] = {}
        batch_size: int | None = None

        for stage in stage_names:
            if stage not in attentions:
                raise KeyError(f"Missing {stage!r} in fusion_attentions.")

            attn = attentions[stage].detach().cpu()

            # Average attention heads to get one modality-to-modality matrix per sample
            if attn.ndim == 4:
                attn = attn.mean(dim=1)
            elif attn.ndim != 3:
                raise ValueError(
                    f"Expected {stage} attention to have shape (B, heads, M, M) or (B, M, M).",
                )

            if batch_size is None:
                batch_size = int(attn.shape[0])
            elif int(attn.shape[0]) != batch_size:
                raise ValueError("All fusion attention tensors must have the same batch size.")

            stage_tensors[stage] = attn

        for index in range(batch_size or 0):
            row: dict[str, float] = {}
            for stage in stage_names:
                sample_attn = stage_tensors[stage][index]
                if sample_attn.shape != (len(modalities), len(modalities)):
                    raise ValueError(
                        f"{stage} attention must be a {len(modalities)}x{len(modalities)} matrix per sample.",
                    )
                for q_idx, query_modality in enumerate(modalities):
                    for k_idx, key_modality in enumerate(modalities):
                        row[f"{stage}_{query_modality}_to_{key_modality}"] = float(
                            sample_attn[q_idx, k_idx].item(),
                        )
            rows.append(row)

    with fusion_attentions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fusion_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"M3Care fusion attentions stored at: {fusion_attentions_path}")

    return fusion_attentions_path


def extract_m3care_imputation_params(lit_model: MMModel) -> dict[str, Any] | None:
    """Read learned imputation parameters and their sigmoid values in model modality order."""
    if lit_model.fusion_method not in M3CARE_METHODS:
        return None
    if not getattr(lit_model.model, "perform_imputation", False):
        return None

    imputation = getattr(lit_model.model, "imputation", None)
    if imputation is None:
        return None

    eps_params = imputation.eps_params
    eps_records = []
    if isinstance(eps_params, nn.ParameterDict):
        iterator = ((name, eps_params[name]) for name in imputation.modality_names)
    elif isinstance(eps_params, nn.ParameterList):
        iterator = zip(imputation.modality_names, eps_params)
    else:
        raise TypeError("eps_params must be either nn.ParameterList or nn.ParameterDict.")

    for modality_name, p in iterator:
        raw = float(p.detach().cpu().item())
        eps_records.append(
            {
                "modality": modality_name,
                "raw": raw,
                "sigmoid": float(torch.sigmoid(p.detach()).cpu().item()),
            }
        )

    threshold_raw = float(imputation.threshold.detach().cpu().item())
    threshold_sigmoid = float(torch.sigmoid(imputation.threshold.detach()).cpu().item())
    return {
        "eps_params": eps_records,
        "threshold": {"raw": threshold_raw, "sigmoid": threshold_sigmoid},
    }


def _serialise_value(value: Any) -> Any:
    """Convert tensors, paths and nested values to types that can be saved in JSON or CSV."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _serialise_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _serialise_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialise_value(item) for item in value]
    return value


def flatten_grouped_metric_summary(
    grouped_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Flatten available metrics in the configured stage and metric order."""
    flat_metrics: dict[str, Any] = {}
    for stage in SUMMARY_STAGES:
        stage_metrics = grouped_metrics.get(stage, {})
        for metric in SUMMARY_METRICS:
            if metric in stage_metrics:
                flat_metrics[f"{stage}_{metric}"] = _serialise_value(stage_metrics[metric])
    return flat_metrics


def write_metrics_csv(
    output_path: str | Path, metadata: Mapping[str, Any],
    grouped_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write one summary row with metadata columns first, followed by sorted metric columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = flatten_grouped_metric_summary(grouped_metrics)
    row = {
        **{key: _serialise_value(value) for key, value in metadata.items()},
        **{key: _serialise_value(value) for key, value in metrics.items()},
    }
    fieldnames = list(metadata.keys()) + sorted(metrics.keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return row


def write_history_csv(
    metrics_path: str | Path, output_path: str | Path, *, include_m3care_cl: bool = False,
) -> None:
    """Merge CSV logger rows by epoch, keeping the latest non-empty value for each column."""
    metrics_path = Path(metrics_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "train_loss", "train_aux_loss", "train_mcc", "train_ba", "val_loss",
        "val_aux_loss", "val_mcc", "val_ba",
    ]
    if include_m3care_cl:
        train_insert = fieldnames.index("train_mcc")
        fieldnames[train_insert:train_insert] = [
            "train_observed_contrastive_loss", "train_missing_contrastive_loss",
            "train_weighted_observed_contrastive_loss", "train_weighted_missing_contrastive_loss",
            "train_weighted_contrastive_loss", "train_observed_contrastive_sample_count",
            "train_missing_contrastive_sample_count", "train_observed_contrastive_sample_fraction",
            "train_missing_contrastive_sample_fraction", "train_learned_target_reconstruction_loss",
            "train_learned_target_contrastive_loss",
            "train_weighted_learned_target_reconstruction_loss",
            "train_weighted_learned_target_contrastive_loss", "train_weighted_learned_target_loss",
            "train_learned_target_reconstruction_sample_count",
            "train_learned_target_reconstruction_sample_fraction",
            "train_learned_target_contrastive_sample_count",
            "train_learned_target_contrastive_sample_fraction",
            "train_observed_contrastive_enabled", "train_missing_contrastive_enabled",
            "train_learned_target_supervision_enabled",
        ]
        val_insert = fieldnames.index("val_mcc")
        fieldnames[val_insert:val_insert] = [
            "val_observed_contrastive_loss", "val_missing_contrastive_loss",
            "val_weighted_observed_contrastive_loss", "val_weighted_missing_contrastive_loss",
            "val_weighted_contrastive_loss", "val_observed_contrastive_sample_count",
            "val_missing_contrastive_sample_count", "val_observed_contrastive_sample_fraction",
            "val_missing_contrastive_sample_fraction", "val_learned_target_reconstruction_loss",
            "val_learned_target_contrastive_loss",
            "val_weighted_learned_target_reconstruction_loss",
            "val_weighted_learned_target_contrastive_loss", "val_weighted_learned_target_loss",
            "val_learned_target_reconstruction_sample_count",
            "val_learned_target_reconstruction_sample_fraction",
            "val_learned_target_contrastive_sample_count",
            "val_learned_target_contrastive_sample_fraction", "val_observed_contrastive_enabled",
            "val_missing_contrastive_enabled", "val_learned_target_supervision_enabled",
        ]

    rows_by_epoch: dict[str, dict[str, Any]] = {}
    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                compact_row = {key: row.get(key, "") for key in fieldnames}
                if not any(compact_row[key] != "" for key in fieldnames[1:]):
                    continue

                epoch_key = compact_row.get("epoch", "")
                if epoch_key == "":
                    continue

                merged = rows_by_epoch.setdefault(epoch_key, {key: "" for key in fieldnames})
                for key in fieldnames:
                    value = compact_row[key]
                    if value != "":
                        merged[key] = value

    rows = list(rows_by_epoch.values())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(data: Mapping[str, Any], output_path: str | Path) -> None:
    """Save YAML settings while keeping the dictionary's key order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(dict(data), output_file, sort_keys=False)


def write_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> None:
    """Save run metadata and output paths as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(_serialise_value(manifest), output_file, indent=2)
        output_file.write("\n")


def build_grouped_metric_summary(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Group the configured metrics under their training, validation or test stage."""
    grouped: dict[str, dict[str, Any]] = {}
    for stage in SUMMARY_STAGES:
        stage_metrics = {}
        for metric in SUMMARY_METRICS:
            key = f"{stage}_{metric}"
            if key in metrics:
                stage_metrics[metric] = _serialise_value(metrics[key])
        if stage_metrics:
            grouped[stage] = stage_metrics
    return grouped


def merge_grouped_metric_summaries(
    *summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge stage summaries in order, letting later values replace earlier ones."""
    merged: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for stage in SUMMARY_STAGES:
            stage_metrics = summary.get(stage, {})
            if stage_metrics:
                merged.setdefault(stage, {}).update(stage_metrics)
    return merged


def relabel_val_metrics_to_train(results: list[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Relabel the first validation result as training metrics after scoring the training split."""
    if not results:
        return {}
    return {
        (f"train_{key[len('val_'):]}" if key.startswith("val_") else key): value
        for key, value in results[0].items()
    }


def build_eval_dataloader(loader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    """Reuse a loader's dataset and batch settings without shuffling or dropping the last batch."""
    return torch.utils.data.DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers,
        collate_fn=loader.collate_fn, pin_memory=loader.pin_memory, drop_last=False,
    )


def remove_flat_stage_metrics(summary: Any) -> None:
    """Remove flat stage-prefixed metrics from a summary."""
    for key in list(summary.keys()):
        if any(str(key).startswith(f"{stage}_") for stage in SUMMARY_STAGES):
            del summary[key]


def update_wandb_summary(
    logger: WandbLogger, grouped_metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    """Store grouped metrics in W&B and remove their old flat entries."""
    summary = logger.experiment.summary
    summary["metrics"] = _serialise_value(grouped_metrics)
    remove_flat_stage_metrics(summary)


class EpochSummaryPrinter(Callback):
    """Print available training and validation metrics at the end of each training epoch."""

    def __init__(self, include_m3care_cl: bool = False) -> None:
        super().__init__()
        self.include_m3care_cl = bool(include_m3care_cl)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        train_names = TRAIN_EPOCH_LOG_METRICS
        val_names = VAL_EPOCH_LOG_METRICS

        # Finalise M3Care epoch values here because this callback runs before the module's epoch hook
        if self.include_m3care_cl and getattr(pl_module, "fusion_method", None) == "m3care":
            finalise = getattr(pl_module, "_finalise_m3care_epoch_stats", None)

            if callable(finalise):
                finalise("val")
                finalise("train")

        if self.include_m3care_cl:
            extra_train = (
                "train_observed_contrastive_loss", "train_missing_contrastive_loss",
                "train_weighted_observed_contrastive_loss",
                "train_weighted_missing_contrastive_loss", "train_weighted_contrastive_loss",
                "train_observed_contrastive_sample_fraction",
                "train_missing_contrastive_sample_fraction",
                "train_learned_target_reconstruction_loss", "train_learned_target_contrastive_loss",
                "train_weighted_learned_target_loss",
                "train_learned_target_reconstruction_sample_fraction",
                "train_learned_target_contrastive_sample_fraction",
            )
            extra_val = (
                "val_observed_contrastive_loss", "val_missing_contrastive_loss",
                "val_weighted_observed_contrastive_loss", "val_weighted_missing_contrastive_loss",
                "val_weighted_contrastive_loss", "val_observed_contrastive_sample_fraction",
                "val_missing_contrastive_sample_fraction", "val_learned_target_reconstruction_loss",
                "val_learned_target_contrastive_loss", "val_weighted_learned_target_loss",
                "val_learned_target_reconstruction_sample_fraction",
                "val_learned_target_contrastive_sample_fraction",
            )
            # M3Care stores epoch averages on the module and _format_metrics skips unavailable values
            train_names = train_names + extra_train
            val_names = val_names + extra_val

        train_metrics = self._format_metrics(trainer, train_names, pl_module)
        val_metrics = self._format_metrics(trainer, val_names, pl_module)
        if train_metrics and val_metrics:
            metrics = f"{train_metrics} | {val_metrics}"
        else:
            metrics = train_metrics or val_metrics

        if metrics:
            print(f"Epoch {trainer.current_epoch}: {metrics}", flush=True)

    @staticmethod
    def _format_metrics(
        trainer: L.Trainer, metric_names: tuple[str, ...],
        pl_module: L.LightningModule | None = None,
    ) -> str:
        metrics = []
        m3care_values: dict[str, float] = {}
        if (
            pl_module is not None
            and hasattr(pl_module, "_m3care_epoch_reported_values")
            and metric_names
        ):
            stage = "train" if metric_names[0].startswith("train_") else "val"
            m3care_values = pl_module._m3care_epoch_reported_values.get(stage, {})

        for key in metric_names:
            value = m3care_values.get(key)
            # Use Lightning's callback metrics when the module has no saved epoch value
            if value is None:
                value = trainer.callback_metrics.get(key)
            if value is None:
                continue

            if torch.is_tensor(value):
                value = float(value.detach().cpu().item())
            metrics.append(f"{key}={float(value):.4f}")

        return " ".join(metrics)


def main() -> None:
    """Train the configured model, score checkpoints and write metrics and model reports."""
    args = parse_args()
    cfg = load_config(args.config)
    apply_config_overrides(cfg, args.config_overrides)

    seed = int(cfg.get("seed", 42))
    L.seed_everything(seed, workers=True)

    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    fusion_method = model_cfg.get("fusion_method", model_cfg.get("name", "concat")).lower()
    # Copy top-level mTAND settings into the model config so checkpoints retain them
    if isinstance(cfg.get("mtand"), Mapping):
        model_cfg = {**model_cfg, "mtand": dict(cfg["mtand"])}
    # Keep raw time-series settings in the saved model config as well
    if isinstance(cfg.get("ts_modality"), Mapping):
        model_cfg = {**model_cfg, "ts_modality": dict(cfg["ts_modality"])}
    if bool(data_cfg.get("MTL", False)):
        raise ValueError(
            "train.py currently supports only single-task training; set data.MTL=false.",
        )
    if int(model_cfg.get("output_dim", 1)) != 1:
        raise ValueError("train.py currently supports only model.output_dim=1.")

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    epochs = int(train_cfg.get("epochs", 1))
    accelerator = "auto"
    outcome = str(data_cfg.get("outcome", cfg.get("outcome", "unknown")))
    bundle_name = resolve_bundle_name(data_cfg)
    run_id, run_dir, run_layers = resolve_run_dir(
        data_cfg=data_cfg, outcome=outcome, fusion_method=fusion_method, run_id=args.run_id,
        use_wandb=args.wandb, output_root=args.output_root,
    )

    # A pretrained checkpoint supplies model weights for a new training run
    pretrained_checkpoint = model_cfg.get("load_from")

    if pretrained_checkpoint:
        pretrained_checkpoint = Path(pretrained_checkpoint)
        if not pretrained_checkpoint.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint_best.ckpt"
    predictions_path = run_dir / "test_predictions.csv"
    if fusion_method in EXPERT_METHODS:
        expert_output_dir = resolve_expert_output_dir(run_dir, fusion_method)
        moe_expert_gate_weights_path = expert_output_dir / "test_expert_gate_weights.csv"
        moe_expert_summary_path = expert_output_dir / "test_expert_summary.csv"
        moe_modality_expert_summary_path = expert_output_dir / "test_modality_expert_summary.csv"
        moe_router_contribution_summary_path = (
            expert_output_dir / "test_router_contribution_summary.csv"
        )
    if fusion_method in M3CARE_METHODS:
        attentions_dir = run_dir / "attentions"
        m3care_imputation_attentions_path = attentions_dir / "imputation_attentions.csv"
        m3care_fusion_attentions_path = attentions_dir / "fusion_attentions.csv"
        m3care_imputation_params_path = run_dir / "imputation" / "imputation_learned_params.json"

    metrics_path = run_dir / "metrics.csv"
    history_path = run_dir / "history.csv"
    manifest_path = run_dir / "manifest.json"
    write_yaml(cfg, config_path)

    print("Use the complete set:", cfg.get("data", {}).get("use_complete_set"))
    print("Filter the complete set based on:", cfg.get("data", {}).get("complete_set_modalities"))

    dataloaders = get_dataloaders(cfg)
    dataset_sizes = {split: len(loader.dataset) for split, loader in dataloaders.items()}
    batch_sizes = {split: len(loader) for split, loader in dataloaders.items()}
    print("\n============ Dataset Sizes ============")
    for split, loader in dataloaders.items():
        print(
            f"{split.upper():5s}: {len(loader.dataset):,} samples, {len(loader):,} batches",
        )
    print("=======================================\n")

    print("Available dataloaders:", list(dataloaders.keys()))

    first_batch = next(iter(dataloaders["train"]))
    feature_dims = batch_feature_dims(first_batch)
    pos_weight = compute_train_pos_weight(dataloaders["train"]) if args.use_class_weights else None

    # Model construction uses architecture settings, while load_from controls weight loading
    fusion_config = dict(model_cfg)
    fusion_config.pop("load_from", None)

    lit_model = MMModel(
        model=None, learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)), pos_weight=pos_weight,
        scheduler_patience=5, scheduler_monitor="val_mcc", fusion_method=fusion_method,
        fusion_config=fusion_config, feature_dims=feature_dims,
        modality_dropout_config=dict(train_cfg.get("modality_dropout", {})),
    )

    print_section("Model initialisation")
    if pretrained_checkpoint:
        print(f"Initialising model from pretrained checkpoint: {pretrained_checkpoint}")

        load_strict = bool(model_cfg.get("load_strict", True))
        missing_keys, unexpected_keys = lit_model.load_pretrained_weights(
            str(pretrained_checkpoint), strict=load_strict,
        )

        print("Pretrained model weights loaded successfully.")

        if missing_keys:
            print(f"Missing pretrained keys: {missing_keys}")

        if unexpected_keys:
            print(f"Unexpected pretrained keys: {unexpected_keys}")
    else:
        print("Initialising model from random weights")

    lit_model.example_input_array = (first_batch,)
    early_stopping_patience = int(train_cfg.get("early_stopping", 0))
    checkpoint = ModelCheckpoint(
        dirpath=run_dir, filename="checkpoint_best", monitor="val_mcc", mode="max",
    )
    callbacks = [checkpoint, EpochSummaryPrinter(include_m3care_cl=(fusion_method == "m3care"))]
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_mcc", mode="max", patience=early_stopping_patience),
        )
    run_name = resolve_run_name(
        args.run_name, outcome=outcome, fusion_method=fusion_method, run_id=run_id,
    )
    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs", version="")
    wandb_logger: WandbLogger | None = None
    logger: CSVLogger | WandbLogger | list[CSVLogger | WandbLogger]
    if args.wandb:
        import wandb

        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            try:
                wandb.login(key=wandb_api_key, relogin=True)
            except (ValueError, wandb.errors.Error) as exc:
                # Remove a rejected authentication key so W&B can use stored credentials
                # Other login errors leave the key available for W&B initialisation to retry
                print(f"W&B login from WANDB_API_KEY failed: {exc}")
                if isinstance(exc, wandb.errors.AuthenticationError):
                    os.environ.pop("WANDB_API_KEY", None)
                    print(
                        "Ignoring WANDB_API_KEY and falling back to stored W&B "
                        "credentials (~/.netrc)."
                    )
        wandb_logger = WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name, id=run_id,
            save_dir=str(DEFAULT_WANDB_DIR), log_model=True,
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )

        if fusion_method == "m3care":
            wandb_run = wandb_logger.experiment

            wandb_run.define_metric("epoch")

            m3care_epoch_metrics = (
                "train_observed_contrastive_loss", "train_missing_contrastive_loss",
                "train_weighted_observed_contrastive_loss",
                "train_weighted_missing_contrastive_loss", "train_weighted_contrastive_loss",
                "train_observed_contrastive_sample_count", "train_missing_contrastive_sample_count",
                "train_observed_contrastive_sample_fraction",
                "train_missing_contrastive_sample_fraction",
                "train_learned_target_reconstruction_loss", "train_learned_target_contrastive_loss",
                "train_weighted_learned_target_reconstruction_loss",
                "train_weighted_learned_target_contrastive_loss",
                "train_weighted_learned_target_loss",
                "train_learned_target_reconstruction_sample_count",
                "train_learned_target_reconstruction_sample_fraction",
                "train_learned_target_contrastive_sample_count",
                "train_learned_target_contrastive_sample_fraction",
                "train_observed_contrastive_enabled", "train_missing_contrastive_enabled",
                "train_learned_target_supervision_enabled",

                "val_observed_contrastive_loss", "val_missing_contrastive_loss",
                "val_weighted_observed_contrastive_loss",
                "val_weighted_missing_contrastive_loss", "val_weighted_contrastive_loss",
                "val_observed_contrastive_sample_count", "val_missing_contrastive_sample_count",
                "val_observed_contrastive_sample_fraction",
                "val_missing_contrastive_sample_fraction", "val_learned_target_reconstruction_loss",
                "val_learned_target_contrastive_loss",
                "val_weighted_learned_target_reconstruction_loss",
                "val_weighted_learned_target_contrastive_loss", "val_weighted_learned_target_loss",
                "val_learned_target_reconstruction_sample_count",
                "val_learned_target_reconstruction_sample_fraction",
                "val_learned_target_contrastive_sample_count",
                "val_learned_target_contrastive_sample_fraction",
                "val_observed_contrastive_enabled", "val_missing_contrastive_enabled",
                "val_learned_target_supervision_enabled",
            )

            for metric_name in m3care_epoch_metrics:
                wandb_run.define_metric(metric_name, step_metric="epoch")
        wandb_logger.experiment.config.update(
            {
                **cfg,
                "run_id": run_id,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "general_task": str(run_layers[0]),
                "patient_stay": str(run_layers[1]),
                "record": str(run_layers[2]),
                "cohort": str(run_layers[3]),
                "preprocessing": str(run_layers[4]),
                "dataset_sizes": dataset_sizes,
                "dataset_batches": batch_sizes,
            }
        )
        logger = [csv_logger, wandb_logger]
    else:
        logger = csv_logger

    print_section("Run Setup")
    print(f"Training {fusion_method} model with accelerator={accelerator} for {epochs} epoch(s).")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Deterioration outcome: {outcome}")
    print(f"Data bundle: {bundle_name}")
    print(f"Feature dimensions: {feature_dims}")
    if pos_weight is None:
        print("Positive class weight: disabled")
    else:
        print(f"Positive class weight: {float(pos_weight.detach().cpu().item()):.3f}")
    if early_stopping_patience > 0:
        print(f"Early stopping on validation MCC with patience={early_stopping_patience}.")
    if args.wandb:
        print(f"W&B init timeout: {args.wandb_init_timeout}s")

    # Report the enabled M3Care losses and their weights
    if fusion_method == "m3care":
        contrastive_modalities = fusion_config.get("contrastive_modalities")
        observed_enabled = bool(fusion_config.get("enable_observed_contrastive", False))
        missing_enabled = bool(fusion_config.get("enable_missing_contrastive", False))
        if contrastive_modalities is not None or observed_enabled or missing_enabled:
            print(
                f"Contrastive modalities: {contrastive_modalities or 'default (all imputed modalities)'}",
            )
            print(f"Observed contrastive enabled: {observed_enabled}")
            print(f"Missing contrastive enabled: {missing_enabled}")
            print(
                f"Observed contrastive weight: {float(fusion_config.get('observed_contrastive_weight', 1.0))}",
            )
            print(
                f"Missing contrastive weight: {float(fusion_config.get('missing_contrastive_weight', 1.0))}",
            )
            print(
                f"Minimum contrastive samples: {int(fusion_config.get('min_contrastive_samples', 2))}",
            )

        learned_target_enabled = bool(fusion_config.get("learned_cross_modal_target", False))
        learned_target_supervision = bool(
            fusion_config.get("enable_learned_target_supervision", False),
        )

        if learned_target_enabled or learned_target_supervision:
            print(f"Learned cross-modal target: {learned_target_enabled}")
            print(f"Learned target supervision: {learned_target_supervision}")
            print(
                f"Learned target reconstruction type: {fusion_config.get('learned_target_loss_type', 'cosine')}",
            )
            print(
                f"Learned target reconstruction weight: {float(fusion_config.get('learned_target_loss_weight', 1.0))}",
            )
            print(
                f"Learned target contrastive weight: {float(fusion_config.get('learned_target_contrastive_weight', 1.0))}",
            )
            print(
                f"Learned target contrastive temperature: {fusion_config.get('learned_target_contrastive_temperature', 'same as contrastive_temperature')}",
            )

    trainer = L.Trainer(
        accelerator=accelerator, precision="32-true", max_epochs=epochs, logger=logger,
        callbacks=callbacks, log_every_n_steps=0, enable_progress_bar=False,
    )
    print_section("Training")
    trainer.fit(
        model=lit_model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"],
    )
    # Keep final-epoch metrics as a fallback until the best checkpoint is scored on each split
    fit_grouped_metrics = build_grouped_metric_summary(trainer.callback_metrics)
    print_section("Scoring Train/Val (best checkpoint)")
    best_ckpt_path = checkpoint.best_model_path or None
    train_grouped_metrics: dict[str, dict[str, Any]] = {}
    val_grouped_metrics: dict[str, dict[str, Any]] = {}
    if best_ckpt_path and Path(best_ckpt_path).exists():
        # Evaluate the full training split without dropout or adding rows to the training history
        eval_trainer = L.Trainer(
            accelerator=accelerator, precision="32-true", logger=False, enable_progress_bar=False,
        )
        # Lightning uses val_* keys for both passes, so relabel the training result before saving
        print("Scoring TRAIN split (relabelled to train_* below) ->")
        train_results = eval_trainer.validate(
            model=lit_model, dataloaders=build_eval_dataloader(dataloaders["train"]),
            ckpt_path=best_ckpt_path,
        )
        print("Scoring VAL split ->")
        val_results = eval_trainer.validate(
            model=lit_model, dataloaders=dataloaders["val"], ckpt_path=best_ckpt_path,
        )
        train_grouped_metrics = build_grouped_metric_summary(
            relabel_val_metrics_to_train(train_results),
        )
        val_grouped_metrics = build_grouped_metric_summary(val_results[0] if val_results else {})
    else:
        print(
            "WARNING: no best checkpoint found; metrics.csv falls back to "
            "last-epoch train/val metrics from training."
        )
    print_section("Testing")
    test_results = trainer.test(model=lit_model, dataloaders=dataloaders["test"], ckpt_path="best")
    test_grouped_metrics = build_grouped_metric_summary(test_results[0] if test_results else {})
    # Best-checkpoint scores replace final-epoch values in this order
    grouped_metrics = merge_grouped_metric_summaries(
        fit_grouped_metrics, train_grouped_metrics, val_grouped_metrics, test_grouped_metrics,
    )
    print_section("Predicting")
    prediction_batches = trainer.predict(
        model=lit_model, dataloaders=dataloaders["test"], ckpt_path="best",
    )
    print_section("Writing Outputs")
    write_predictions_csv(prediction_batches, predictions_path)
    modality_availability_batches = (
        collect_modality_availability_batches(dataloaders["test"])
        if uses_expert_modality_availability(fusion_method)
        else None
    )
    moe_expert_paths: dict[str, Path] = {}
    if fusion_method in EXPERT_METHODS:
        moe_expert_paths = write_moe_expert_outputs(
            prediction_batches, moe_expert_gate_weights_path, moe_expert_summary_path,
            moe_modality_expert_summary_path, modality_availability_batches,
            moe_expert_names(lit_model.model),
            shared_pool_modalities=shared_expert_pool_modalities(lit_model.model),
            router_contribution_summary_path=moe_router_contribution_summary_path,
        )

    m3care_attention_params_paths: dict[str, Path] = {}
    if fusion_method in M3CARE_METHODS:
        m3care_params = None
        if best_ckpt_path and Path(best_ckpt_path).exists():
            best_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            lit_model.load_state_dict(best_state["state_dict"], strict=True)
            m3care_params = extract_m3care_imputation_params(lit_model)

        m3care_modalities_to_impute = model_cfg.get("modalities_to_impute", None)
        if m3care_modalities_to_impute is None:
            imputation_module = getattr(lit_model.model, "imputation", None)
            if imputation_module is not None:
                m3care_modalities_to_impute = list(imputation_module.modality_names)
            else:
                m3care_modalities_to_impute = list(
                    getattr(lit_model.model, "modality_names", ("tabular", "ecg", "cxr")),
                )

        if fusion_method == "mtand_m3care":
            m3care_fused_modalities = list(getattr(lit_model.model, "fusion_modality_names", None))
        else:
            m3care_fused_modalities = list(
                model_cfg.get("modality_names", ("tabular", "ecg", "cxr")),
            )

        m3care_attention_params_paths["m3care_imputation_attentions_path"] = (
            write_m3care_imputation_attentions_csv(
                prediction_batches, m3care_imputation_attentions_path,
                modality_names=list(m3care_modalities_to_impute),
            )
        )
        m3care_attention_params_paths["m3care_fusion_attentions_path"] = (
            write_m3care_fusion_attentions_csv(
                prediction_batches, m3care_fusion_attentions_path,
                modality_names=m3care_fused_modalities,
            )
        )

        if m3care_params is not None:
            m3care_imputation_params_path.parent.mkdir(parents=True, exist_ok=True)
            with m3care_imputation_params_path.open("w", encoding="utf-8") as f:
                json.dump(m3care_params, f, indent=2)
            m3care_attention_params_paths["m3care_imputation_params_path"] = (
                m3care_imputation_params_path
            )
            print(f"M3Care imputation params stored at: {m3care_imputation_params_path}")
        else:
            m3care_attention_params_paths["m3care_imputation_params_path"] = None

    modality_ablation_paths: dict[str, Path] = {}
    evaluation_cfg = cfg.get("evaluation", {})
    modality_ablation_cfg = (
        evaluation_cfg.get("modality_ablation", {}) if isinstance(evaluation_cfg, Mapping) else {}
    )
    metric_row = write_metrics_csv(
        metrics_path,
        {
            "run_id": run_id,
            "model": fusion_method,
            "outcome": outcome,
            "bundle": bundle_name,
            "checkpoint_path": checkpoint.best_model_path,
        },
        grouped_metrics,
    )
    if wandb_logger is not None:
        update_wandb_summary(wandb_logger, grouped_metrics)
    write_history_csv(
        Path(csv_logger.log_dir) / "metrics.csv", history_path,
        include_m3care_cl=(fusion_method == "m3care"),
    )

    if isinstance(modality_ablation_cfg, Mapping) and bool(
        modality_ablation_cfg.get("enabled", False),
    ):
        print_section("Modality Ablation")
        best_ckpt = checkpoint.best_model_path or str(checkpoint_path)
        if best_ckpt and Path(best_ckpt).exists():
            best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            lit_model.load_state_dict(best_state["state_dict"], strict=True)
            print(f"Loaded best checkpoint for modality ablation: {best_ckpt}")
        else:
            print(
                "WARNING: no best checkpoint found; modality ablation falls back "
                "to in-memory weights and may not match test_predictions.csv."
            )
        # Restore the evaluation device after Lightning moves the model to CPU during teardown
        # Small CPU/GPU differences can select different experts, changing ablation results
        lit_model.to(_usable_device())
        modality_ablation_scenarios = resolve_modality_ablation_scenarios(
            lit_model.model, cfg, dataloader=dataloaders["test"],
        )
        modality_ablation_paths = run_modality_ablation(
            lit_model,
            dataloaders["test"],
            run_dir / "modality_ablation",
            modality_ablation_scenarios,
            fusion_method=fusion_method,
            expert_names=(
                moe_expert_names(lit_model.model) if fusion_method in EXPERT_METHODS else None
            ),
            shared_pool_modalities=(
                shared_expert_pool_modalities(lit_model.model)
                if fusion_method in EXPERT_METHODS
                else None
            ),
            val_dataloader=dataloaders.get("val"),
        )

    write_manifest(
        {
            "run_id": run_id,
            "run_dir": run_dir,
            "config_path": config_path,
            "checkpoint_path": checkpoint.best_model_path or checkpoint_path,
            "predictions_path": predictions_path,
            **moe_expert_paths,
            **m3care_attention_params_paths,
            "modality_ablation_paths": modality_ablation_paths,
            "metrics_path": metrics_path,
            "history_path": history_path,
            "model": fusion_method,
            "model_config": dict(model_cfg),
            "outcome": outcome,
            "bundle": bundle_name,
            "seed": seed,
            "accelerator": accelerator,
            "feature_dims": feature_dims,
            "use_class_weights": args.use_class_weights,
            "pos_weight": None if pos_weight is None else float(pos_weight.detach().cpu().item()),
            "summary_metrics": metric_row,
            "metrics": grouped_metrics,
            "dataset_sizes": dataset_sizes,
            "dataset_batches": batch_sizes,
        },
        manifest_path,
    )

    print(f"Run outputs stored at: {run_dir}")


if __name__ == "__main__":
    main()
