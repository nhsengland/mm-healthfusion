"""Evaluate saved checkpoints with selected modalities removed.

Write scenario metrics, predictions and expert summaries to modality_ablation.

Usage:
    python -m src.evaluation.modality_ablation --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.modality_ablation --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.collate import (
    batch_modality_availability as shared_batch_modality_availability,
)
from src.data.datasets import HOSPITALISATION_MODALITIES
from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    _model_device,
    _move_batch_to_device,
    _scenario_torchmetrics,
    _usable_device,
    discover_checkpoints,
    load_config,
)
from src.evaluation.output_expert import (
    EXPERT_METHODS,
    collapse_shared_expert_pool,
    moe_expert_names,
    shared_expert_pool_modalities,
)
from src.models.base import canonical_modalities
from src.training.lightning import MMModel

# Default deterioration modality order, including optional time series
MODALITIES = ("tabular", "ecg", "cxr", "ts")
NO_ABLATION_SCENARIO = "no_ablation"


@dataclass(frozen=True)
class ModalityScenario:
    """A named set of modalities to keep during evaluation."""

    name: str
    modalities: tuple[str, ...]

    @property
    def label(self) -> str:
        return "+".join(self.modalities)


# Order subsets by size, then modality order
# Scenarios requiring unavailable modalities have no eligible rows
SCENARIOS = (
    # singletons
    ModalityScenario("tabular_only", ("tabular",)),
    ModalityScenario("ecg_only", ("ecg",)),
    ModalityScenario("cxr_only", ("cxr",)),
    ModalityScenario("ts_only", ("ts",)),
    # pairs
    ModalityScenario("tabular_ecg", ("tabular", "ecg")),
    ModalityScenario("tabular_cxr", ("tabular", "cxr")),
    ModalityScenario("tabular_ts", ("tabular", "ts")),
    ModalityScenario("ecg_cxr", ("ecg", "cxr")),
    ModalityScenario("ecg_ts", ("ecg", "ts")),
    ModalityScenario("cxr_ts", ("cxr", "ts")),
    # triples
    ModalityScenario("tabular_ecg_cxr", ("tabular", "ecg", "cxr")),
    ModalityScenario("tabular_ecg_ts", ("tabular", "ecg", "ts")),
    ModalityScenario("tabular_cxr_ts", ("tabular", "cxr", "ts")),
    ModalityScenario("ecg_cxr_ts", ("ecg", "cxr", "ts")),
    # full set
    ModalityScenario(NO_ABLATION_SCENARIO, MODALITIES),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run modality ablation for a single checkpoint (--checkpoint) or for "
            "every run found under a directory tree (--path)."
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
            "Root directory to search recursively for run checkpoints. Every "
            "checkpoint_best.ckpt file found under it is evaluated in place, "
            "writing outputs to each run's own directory."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "Path to the YAML config used to build the test dataloader. "
            "Defaults to config.yaml in each checkpoint's run directory (its "
            "parent folder), which training writes there. With --path this is a "
            "fallback used only for runs missing their own config.yaml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where test_modality_ablation_metrics.csv and "
            "test_modality_ablation_predictions_wide.csv are written. "
            "Defaults to the checkpoint's run directory (its parent folder). "
            "Ignored in --path mode, where each run writes to its own directory."
        ),
    )
    parser.add_argument(
        "--fusion-method",
        choices=("concat", "m3care", "fusemoe", "smp", "mtand_m3care"),
        default=None,
        help=(
            "Fusion method for model-specific ablation outputs. Defaults to "
            "model.name or model.fusion_method from the config."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --path mode, skip runs that already have a "
            "test_modality_ablation_metrics.csv (useful to resume a batch)."
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
    return parser.parse_args(argv)


def scenarios_for_active_modalities(
    active_modalities: Iterable[str],
) -> tuple[ModalityScenario, ...]:
    """Return all non-empty subsets of the active modalities.

    Name the full set no_ablation and use the standard modality order.
    """
    active = canonical_modalities(tuple(str(modality) for modality in active_modalities))
    if not active:
        raise ValueError("active_modalities must include at least one known modality.")
    selected = [
        ModalityScenario(_subset_scenario_name(modalities), modalities)
        for size in range(1, len(active))
        for modalities in combinations(active, size)
    ]
    selected.append(ModalityScenario(NO_ABLATION_SCENARIO, active))
    return tuple(selected)


def hospitalisation_default_scenarios(
    active_modalities: Iterable[str],
) -> tuple[ModalityScenario, ...]:
    """Return each modality alone, each set with one removed and the full set.

    This keeps the number of evaluations manageable for seven hospitalisation inputs.
    """
    active = canonical_modalities(tuple(str(modality) for modality in active_modalities))
    if not active:
        raise ValueError("active_modalities must include at least one known modality.")

    scenarios = [
        ModalityScenario(_subset_scenario_name((modality,)), (modality,))
        for modality in active
    ]
    scenarios.extend(
        ModalityScenario(
            f"without_{dropped}",
            tuple(modality for modality in active if modality != dropped),
        )
        for dropped in active
    )
    scenarios.append(ModalityScenario(NO_ABLATION_SCENARIO, active))
    return tuple(scenarios)


def _subset_scenario_name(modalities: Sequence[str]) -> str:
    """Return the stable name used for a kept-modality subset."""
    if len(modalities) == 1:
        return f"{modalities[0]}_only"
    return "_".join(modalities)


def run_modality_ablation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: str | Path,
    scenarios: Iterable[ModalityScenario] | None = None,
    *,
    fusion_method: str = "concat",
    expert_names: list[str] | tuple[str, ...] | None = None,
    shared_pool_modalities: int | None = None,
    val_dataloader: DataLoader | None = None,
) -> dict[str, Path]:
    """Evaluate modality subsets and write their outputs to output_dir.

    Without explicit scenarios, use the hospitalisation defaults for models with
    hospitalisation modalities and the standard subset scenarios for other models.

    When val_dataloader is supplied, select the threshold with the highest MCC for
    each scenario on validation data. Apply it to test data and report it as
    best_mcc_threshold, with its test MCC in best_mcc. Leave both fields blank when
    no validation data is supplied.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_method = fusion_method.lower()
    if scenarios is None:
        scenarios = _default_scenarios_for_model(model, dataloader)
    scenarios = tuple(scenarios)
    wide_rows, scenario_outputs = collect_modality_ablation_outputs(
        model,
        dataloader,
        scenarios,
    )

    val_thresholds: dict[str, float] | None = None
    if val_dataloader is not None:
        _, val_scenario_outputs = collect_modality_ablation_outputs(
            model,
            val_dataloader,
            scenarios,
        )
        val_thresholds = calibrate_scenario_thresholds(val_scenario_outputs)

    metrics_path = output_dir / "test_modality_ablation_metrics.csv"
    predictions_path = output_dir / "test_modality_ablation_predictions_wide.csv"
    modality_names = _scenario_modality_names(scenarios)
    write_scenario_metrics_csv(
        scenario_outputs,
        metrics_path,
        val_thresholds,
        scenarios=scenarios,
    )
    write_wide_predictions_csv(
        wide_rows,
        scenarios,
        predictions_path,
        modality_names=modality_names,
    )
    paths = {
        "test_modality_ablation_metrics_path": metrics_path,
        "test_modality_ablation_predictions_wide_path": predictions_path,
    }
    if fusion_method in EXPERT_METHODS:
        expert_summary_path = output_dir / "test_modality_ablation_expert_summary.csv"
        write_expert_summary_csv(
            scenario_outputs,
            expert_summary_path,
            expert_names,
            shared_pool_modalities=shared_pool_modalities,
            scenarios=scenarios,
        )
        paths["test_modality_ablation_expert_summary_path"] = expert_summary_path

        router_contribution_path = (
            output_dir / "test_modality_ablation_router_contribution.csv"
        )
        router_rows = write_router_contribution_summary_csv(
            scenario_outputs,
            router_contribution_path,
            modality_names=modality_names,
            scenarios=scenarios,
        )
        if router_rows:
            paths["test_modality_ablation_router_contribution_path"] = (
                router_contribution_path
            )
    return paths


def collect_modality_ablation_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    scenarios: Iterable[ModalityScenario] = SCENARIOS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[torch.Tensor] | int]]]:
    """Collect predictions only for rows with all modalities required by the scenario."""
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("At least one modality scenario is required.")

    device = _model_device(model)
    previous_training_state = model.training
    model.eval()

    wide_rows: list[dict[str, Any]] = []
    scenario_outputs = {
        scenario.name: _empty_scenario_output() for scenario in scenarios
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_batch_to_device(batch, device)
            availability = batch_modality_availability(batch)
            batch_rows = _base_wide_rows(batch, availability, scenarios)

            for scenario in scenarios:
                eligible = scenario_eligibility(scenario, availability)
                _record_scenario_exclusions(
                    scenario_outputs[scenario.name],
                    eligible,
                )
                if not bool(eligible.any().item()):
                    continue

                masked_batch = mask_batch_for_scenario(batch, scenario)
                eligible_batch = _filter_batch_rows(masked_batch, eligible)
                prediction = model.predict_step(eligible_batch, batch_idx)
                _collect_scenario_output(
                    scenario_outputs[scenario.name],
                    prediction,
                )
                _add_scenario_predictions(batch_rows, scenario, prediction, eligible)

            wide_rows.extend(batch_rows)

    if previous_training_state:
        model.train()

    return wide_rows, scenario_outputs


def batch_modality_availability(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return which modalities are present for each sample before masking."""
    return shared_batch_modality_availability(batch)


def scenario_eligibility(
    scenario: ModalityScenario,
    availability: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Select rows with all modalities required by the scenario.

    If a required modality is missing from availability, no rows are eligible.
    """
    first = next(iter(availability.values()))
    if scenario.name == NO_ABLATION_SCENARIO:
        return torch.ones_like(first, dtype=torch.bool)

    eligible = torch.ones_like(first, dtype=torch.bool)
    for modality in scenario.modalities:
        modality_available = availability.get(modality)
        if modality_available is None:
            return torch.zeros_like(first, dtype=torch.bool)
        eligible = eligible & modality_available.to(dtype=torch.bool)
    return eligible


def mask_batch_for_scenario(
    batch: Mapping[str, torch.Tensor],
    scenario: ModalityScenario,
) -> dict[str, torch.Tensor]:
    """Clone a batch and force non-scenario modalities to be unavailable."""
    masked = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    keep = set(scenario.modalities)
    drop_mask = torch.ones_like(masked["label"], dtype=torch.bool)
    batch_modalities = tuple(str(modality) for modality in masked.get("modality_names", ()))
    modalities = batch_modalities or MODALITIES
    for modality in modalities:
        if modality not in keep:
            MMModel._drop_modality_for_mask(masked, modality, drop_mask)
    return masked


def calibrate_scenario_thresholds(
    scenario_outputs: Mapping[str, dict[str, list[torch.Tensor] | int | str]],
) -> dict[str, float]:
    """Choose the threshold with the highest MCC for each validation scenario.

    Apply the returned thresholds to a separate test split. Use 0.5 when a scenario
    has no positive examples or no negative examples.
    """
    thresholds: dict[str, float] = {}
    for scenario_name, output in scenario_outputs.items():
        labels = _cat_optional(output["label"])
        probabilities = _cat_optional(output["probability"])
        if labels is None or probabilities is None or labels.numel() == 0:
            thresholds[scenario_name] = 0.5
            continue
        targets = labels.to(dtype=torch.int64).reshape(-1)
        probabilities = probabilities.to(dtype=torch.float32).reshape(-1)
        positives = int(targets.sum().item())
        if positives == 0 or positives == targets.numel():
            thresholds[scenario_name] = 0.5
            continue
        _, best_threshold = _best_mcc_threshold(targets, probabilities)
        thresholds[scenario_name] = best_threshold
    return thresholds


def write_scenario_metrics_csv(
    scenario_outputs: Mapping[str, dict[str, list[torch.Tensor] | int | str]],
    output_path: str | Path,
    val_thresholds: Mapping[str, float] | None = None,
    *,
    scenarios: Iterable[ModalityScenario] | None = None,
) -> list[dict[str, Any]]:
    """Write one metric row per ablation scenario."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_lookup = _scenario_lookup(scenarios)
    rows = [
        _scenario_metrics_row(
            scenario_name,
            output,
            None if val_thresholds is None else val_thresholds.get(scenario_name),
            scenario_lookup=scenario_lookup,
        )
        for scenario_name, output in scenario_outputs.items()
    ]
    fieldnames = [
        "scenario", "modalities", "eligible_n", "excluded_missing_modality_n", "positive_n",
        "negative_n", "predicted_positive_n", "predicted_negative_n",
        "predicted_positive_fraction", "true_positive", "true_negative", "false_positive",
        "false_negative", "auroc", "ba", "mcc", "best_mcc", "best_mcc_threshold", "recall",
        "specificity", "probability_min", "probability_max", "probability_mean", "probability_std",
        "probability_p05", "probability_p25", "probability_median", "probability_p75",
        "probability_p95",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_wide_predictions_csv(
    rows: list[dict[str, Any]],
    scenarios: Iterable[ModalityScenario],
    output_path: str | Path,
    *,
    modality_names: Sequence[str] | None = None,
) -> None:
    """Write one row per stay, with blank cells for ineligible scenario outputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the configured modality order, or the first row order for older callers
    if modality_names is None:
        has_columns = (
            [key for key in rows[0] if key.startswith("has_")]
            if rows
            else [f"has_{modality}" for modality in MODALITIES]
        )
    else:
        has_columns = [f"has_{modality}" for modality in modality_names]
    fieldnames = ["ed_stay_id", "label", *has_columns]
    for scenario in scenarios:
        fieldnames.extend(
            [
                f"{scenario.name}_probability",
                f"{scenario.name}_prediction",
            ]
        )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_expert_summary_csv(
    scenario_outputs: Mapping[str, dict[str, list[torch.Tensor] | int | str]],
    output_path: str | Path,
    expert_names: list[str] | tuple[str, ...] | None = None,
    shared_pool_modalities: int | None = None,
    *,
    scenarios: Iterable[ModalityScenario] | None = None,
) -> list[dict[str, Any]]:
    """Write one metric row per scenario and expert.

    For shared pools, average the per-modality gate weights for each expert before
    writing the summary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_lookup = _scenario_lookup(scenarios)
    num_experts = _infer_num_experts(scenario_outputs, shared_pool_modalities)
    rows: list[dict[str, Any]] = []
    for scenario_name, output in scenario_outputs.items():
        gate_weights = _cat_optional(output["gate_weights"])
        if gate_weights is None and num_experts is None:
            continue
        if gate_weights is not None and gate_weights.ndim != 2:
            raise ValueError("gate_weights must have shape (n_samples, n_experts).")
        if gate_weights is not None and shared_pool_modalities is not None:
            gate_weights = _collapse_shared_expert_pool(
                gate_weights,
                shared_pool_modalities,
            )

        scenario_num_experts = (
            num_experts
            if gate_weights is None
            else int(gate_weights.shape[1])
        )
        resolved_expert_names = _resolve_expert_names(
            scenario_num_experts,
            expert_names,
        )
        if gate_weights is None:
            top_expert_counts = torch.zeros(scenario_num_experts, dtype=torch.int64)
            nonzero_gate_counts = torch.zeros(
                scenario_num_experts,
                dtype=torch.int64,
            )
            mean_gate_weights = None
            eligible_n = 0
        else:
            top_expert_counts = torch.bincount(
                gate_weights.argmax(dim=1),
                minlength=scenario_num_experts,
            )
            nonzero_gate_counts = gate_weights.gt(0).sum(dim=0)
            mean_gate_weights = gate_weights.mean(dim=0)
            eligible_n = int(gate_weights.shape[0])

        for expert_index in range(scenario_num_experts):
            rows.append(
                {
                    "scenario": scenario_name,
                    "modalities": _scenario_label(scenario_name, scenario_lookup),
                    "fusemoe_expert": expert_index,
                    "fusemoe_expert_name": resolved_expert_names[expert_index],
                    "eligible_n": eligible_n,
                    "mean_gate_weight": (
                        ""
                        if mean_gate_weights is None
                        else float(mean_gate_weights[expert_index].item())
                    ),
                    "nonzero_gate_count": int(
                        nonzero_gate_counts[expert_index].item()
                    ),
                    "nonzero_gate_fraction": (
                        ""
                        if eligible_n == 0
                        else (
                            float(nonzero_gate_counts[expert_index].item())
                            / float(eligible_n)
                        )
                    ),
                    "top_expert_count": int(top_expert_counts[expert_index].item()),
                    "top_expert_fraction": (
                        ""
                        if eligible_n == 0
                        else (
                            float(top_expert_counts[expert_index].item())
                            / float(eligible_n)
                        )
                    ),
                }
            )

    fieldnames = [
        "scenario", "modalities", "fusemoe_expert", "fusemoe_expert_name", "eligible_n",
        "mean_gate_weight", "nonzero_gate_count", "nonzero_gate_fraction", "top_expert_count",
        "top_expert_fraction",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_router_contribution_summary_csv(
    scenario_outputs: Mapping[str, dict[str, list[torch.Tensor] | int | str]],
    output_path: str | Path,
    *,
    modality_names: Sequence[str] | None = None,
    scenarios: Iterable[ModalityScenario] | None = None,
) -> list[dict[str, Any]]:
    """Write each router's output magnitude and share for each scenario.

    For permod and disjoint routing, use router_output_norms to measure each
    modality's contribution to the combined output. Include only modalities present
    in the scenario, since masked inputs can still produce a non-zero router output.
    Return an empty list and write nothing when no scenario has router norms.
    """
    output_path = Path(output_path)
    modality_names = tuple(modality_names or MODALITIES)
    scenario_lookup = _scenario_lookup(scenarios)
    rows: list[dict[str, Any]] = []
    saw_norms = False
    for scenario_name, output in scenario_outputs.items():
        norms = _cat_optional(output.get("router_output_norms", []))
        if norms is None:
            continue
        if norms.ndim != 2 or not 0 < norms.shape[1] <= len(modality_names):
            raise ValueError(
                "router_output_norms must have shape (n_samples, num_modalities)."
            )
        saw_norms = True
        norms = norms.to(dtype=torch.float64)
        scenario = scenario_lookup[scenario_name]
        # Read only modalities with router outputs to avoid indexing beyond the model width
        present_indices = [
            modality_names.index(modality)
            for modality in scenario.modalities
            if modality in modality_names
            and modality_names.index(modality) < norms.shape[1]
        ]
        if not present_indices:
            continue
        present_norms = norms[:, present_indices]
        denom = present_norms.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(norms.dtype).tiny
        )
        contribution_fraction = present_norms / denom
        eligible_n = int(norms.shape[0])
        for local_index, router_index in enumerate(present_indices):
            rows.append(
                {
                    "scenario": scenario_name,
                    "modalities": _scenario_label(scenario_name, scenario_lookup),
                    "router": router_index,
                    "router_name": modality_names[router_index],
                    "eligible_n": eligible_n,
                    "mean_output_norm": float(
                        norms[:, router_index].mean().item()
                    ),
                    "mean_contribution_fraction": float(
                        contribution_fraction[:, local_index].mean().item()
                    ),
                }
            )

    if not saw_norms:
        return []

    fieldnames = [
        "scenario", "modalities", "router", "router_name", "eligible_n", "mean_output_norm",
        "mean_contribution_fraction",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _base_wide_rows(
    batch: Mapping[str, torch.Tensor],
    availability: Mapping[str, torch.Tensor],
    scenarios: Iterable[ModalityScenario],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ed_stay_ids = batch["ed_stay_id"].detach().cpu()
    labels = batch["label"].detach().cpu()
    # Record availability only for modalities included in the ablation scenarios.
    # The dataset may contain additional modalities that the model does not use.
    active_modalities = _scenario_modality_names(tuple(scenarios))

    has_columns = {
        modality: availability[modality].detach().cpu()
        for modality in active_modalities
        if modality in availability
    }
    for index in range(labels.numel()):
        row = {
            "ed_stay_id": int(ed_stay_ids[index].item()),
            "label": int(labels[index].item()),
            **{
                f"has_{modality}": int(column[index].item())
                for modality, column in has_columns.items()
            },
        }
        for scenario in scenarios:
            row[f"{scenario.name}_probability"] = ""
            row[f"{scenario.name}_prediction"] = ""
        rows.append(row)
    return rows


def _add_scenario_predictions(
    rows: list[dict[str, Any]],
    scenario: ModalityScenario,
    prediction: Mapping[str, torch.Tensor],
    eligible: torch.Tensor,
) -> None:
    probabilities = prediction["probability"].detach().cpu().reshape(-1)
    predictions = prediction["prediction"].detach().cpu().reshape(-1)
    eligible = eligible.detach().cpu()
    eligible_indices = eligible.nonzero(as_tuple=False).flatten()
    if probabilities.numel() != eligible_indices.numel():
        raise ValueError("Prediction count must match eligible scenario rows.")
    for prediction_index, row_index in enumerate(eligible_indices.tolist()):
        row = rows[row_index]
        row[f"{scenario.name}_probability"] = float(
            probabilities[prediction_index].item()
        )
        row[f"{scenario.name}_prediction"] = int(
            predictions[prediction_index].item()
        )


def _record_scenario_exclusions(
    output: dict[str, list[torch.Tensor] | int],
    eligible: torch.Tensor,
) -> None:
    eligible = eligible.detach().cpu()
    output["excluded_n"] = int(output["excluded_n"]) + int(
        (~eligible).sum().item()
    )


def _collect_scenario_output(
    output: dict[str, list[torch.Tensor] | int],
    prediction: Mapping[str, torch.Tensor],
) -> None:
    # Keep single-sample outputs one-dimensional so batches can be concatenated
    for key in ("label", "probability", "prediction"):
        output[key].append(torch.atleast_1d(prediction[key].detach().cpu()))
    if "gate_weights" in prediction:
        output["gate_weights"].append(
            torch.atleast_1d(prediction["gate_weights"].detach().cpu())
        )
    if "router_output_norms" in prediction:
        output["router_output_norms"].append(
            torch.atleast_1d(prediction["router_output_norms"].detach().cpu())
        )


def _filter_batch_rows(
    batch: Mapping[str, torch.Tensor],
    row_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = int(row_mask.numel())
    return {
        key: (
            value[row_mask]
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
            )
            else value
        )
        for key, value in batch.items()
    }


def _scenario_metrics_row(
    scenario_name: str,
    output: Mapping[str, list[torch.Tensor] | int | str],
    val_threshold: float | None = None,
    *,
    scenario_lookup: Mapping[str, ModalityScenario] | None = None,
) -> dict[str, Any]:
    labels = _cat_optional(output["label"])
    probabilities = _cat_optional(output["probability"])
    predictions = _cat_optional(output["prediction"])
    eligible_n = 0 if labels is None else int(labels.numel())
    row = {
        "scenario": scenario_name,
        "modalities": _scenario_label(scenario_name, scenario_lookup),
        "eligible_n": eligible_n,
        "excluded_missing_modality_n": int(output["excluded_n"]),
        "positive_n": "",
        "negative_n": "",
        "auroc": "",
        "ba":  "",
        "mcc":  "",
        "best_mcc":  "",
        "best_mcc_threshold":  "",
        "recall":  "",
        "specificity":  "",
        "predicted_positive_n": "",
        "predicted_negative_n": "",
        "predicted_positive_fraction": "",
        "true_positive": "",
        "true_negative": "",
        "false_positive": "",
        "false_negative": "",
        "probability_min": "",
        "probability_max": "",
        "probability_mean": "",
        "probability_std": "",
        "probability_p05": "",
        "probability_p25": "",
        "probability_median": "",
        "probability_p75": "",
        "probability_p95": "",
    }
    if (
        labels is None
        or probabilities is None
        or predictions is None
        or eligible_n == 0
    ):
        return row

    labels = labels.to(dtype=torch.float32).reshape(-1)
    probabilities = probabilities.to(dtype=torch.float32).reshape(-1)
    predictions = predictions.to(dtype=torch.int64).reshape(-1)
    targets = labels.to(dtype=torch.int64)
    positives = int(targets.sum().item())
    negatives = int(targets.numel() - positives)
    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())
    predicted_positives = int(predictions.sum().item())
    predicted_negatives = int(predictions.numel() - predicted_positives)
    # Use the training metric collection and leave AUROC blank for single-class scenarios
    if positives and negatives:
        scenario_metrics = _scenario_torchmetrics(probabilities, targets)
        auroc: float | str = scenario_metrics["auroc"]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            scenario_metrics = _scenario_torchmetrics(probabilities, targets)
        auroc = ""
    # Apply the validation threshold to test predictions
    # Leave these fields blank when no validation threshold is available
    if val_threshold is None:
        best_mcc: float | str = ""
        best_mcc_threshold: float | str = ""
    else:
        best_mcc = _mcc_at_threshold(targets, probabilities, float(val_threshold))
        best_mcc_threshold = float(val_threshold)
    row.update(
        {
            "auroc": auroc,
            "ba": scenario_metrics["ba"],
            "mcc": scenario_metrics["mcc"],
            "best_mcc": best_mcc,
            "best_mcc_threshold": best_mcc_threshold,
            "recall": scenario_metrics["recall"],
            "specificity": scenario_metrics["specificity"],
            "positive_n": positives,
            "negative_n": negatives,
            "predicted_positive_n": predicted_positives,
            "predicted_negative_n": predicted_negatives,
            "predicted_positive_fraction": _safe_divide(
                predicted_positives,
                predictions.numel(),
            ),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            **_probability_summary(probabilities),
        }
    )
    return row


def _probability_summary(probabilities: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(
        probabilities,
        torch.tensor(
            [0.05, 0.25, 0.5, 0.75, 0.95],
            dtype=probabilities.dtype,
            device=probabilities.device,
        ),
    )
    return {
        "probability_min": float(probabilities.min().item()),
        "probability_max": float(probabilities.max().item()),
        "probability_mean": float(probabilities.mean().item()),
        "probability_std": float(probabilities.std(unbiased=False).item()),
        "probability_p05": float(quantiles[0].item()),
        "probability_p25": float(quantiles[1].item()),
        "probability_median": float(quantiles[2].item()),
        "probability_p75": float(quantiles[3].item()),
        "probability_p95": float(quantiles[4].item()),
    }


def _best_mcc_threshold(targets: torch.Tensor, probabilities: torch.Tensor) -> tuple[float, float]:
    best_mcc = float("-inf")
    best_threshold = 0.5
    sorted_indices = torch.argsort(probabilities, descending=True)
    sorted_probabilities = probabilities[sorted_indices]
    sorted_targets = targets[sorted_indices].to(dtype=torch.int64)
    thresholds, counts = torch.unique_consecutive(
        sorted_probabilities,
        return_counts=True,
    )

    total_positives = int(sorted_targets.sum().item())
    total_negatives = int(sorted_targets.numel() - total_positives)
    true_positive = 0
    false_positive = 0
    offset = 0

    for threshold, count in zip(thresholds, counts):
        group_size = int(count.item())
        group_targets = sorted_targets[offset : offset + group_size]
        group_positives = int(group_targets.sum().item())
        group_negatives = group_size - group_positives

        true_positive += group_positives
        false_positive += group_negatives
        true_negative = total_negatives - false_positive
        false_negative = total_positives - true_positive
        mcc = _mcc(true_positive, true_negative, false_positive, false_negative)
        threshold_value = float(threshold.item())
        if (
            mcc > best_mcc
            or (mcc == best_mcc and threshold_value < best_threshold)
        ):
            best_mcc = mcc
            best_threshold = threshold_value
        offset += group_size
    return best_mcc, best_threshold


def _mcc_at_threshold(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> float:
    """Calculate MCC with predictions positive when probability is at least threshold."""
    targets = targets.to(dtype=torch.int64).reshape(-1)
    predictions = (probabilities.reshape(-1) >= threshold).to(dtype=torch.int64)
    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())
    return _mcc(true_positive, true_negative, false_positive, false_negative)


def _mcc(true_positive: int, true_negative: int, false_positive: int, false_negative: int) -> float:
    """Calculate MCC from counts using the same metric implementation as training.

    Rebuild prediction and target tensors for the metric API. This preserves the
    training calculation's precision and handling of single-class predictions.
    """
    total = true_positive + true_negative + false_positive + false_negative
    if total == 0:
        return 0.0

    tp = torch.tensor(true_positive, dtype=torch.float32)
    tn = torch.tensor(true_negative, dtype=torch.float32)
    fp = torch.tensor(false_positive, dtype=torch.float32)
    fn = torch.tensor(false_negative, dtype=torch.float32)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator.item() == 0.0:
        return 0.0
    return float((numerator / denominator).item())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _cat_optional(value: Any) -> torch.Tensor | None:
    if not value:
        return None
    return torch.cat(value, dim=0)


def _infer_num_experts(
    scenario_outputs: Mapping[str, dict[str, list[torch.Tensor] | int | str]],
    shared_pool_modalities: int | None = None,
) -> int | None:
    for output in scenario_outputs.values():
        gate_weights = _cat_optional(output["gate_weights"])
        if gate_weights is not None:
            if gate_weights.ndim != 2:
                raise ValueError("gate_weights must have shape (n_samples, n_experts).")
            num_columns = int(gate_weights.shape[1])
            if shared_pool_modalities is not None and shared_pool_modalities > 1:
                if num_columns % shared_pool_modalities != 0:
                    raise ValueError(
                        "gate weight columns must be divisible by "
                        "shared_pool_modalities to collapse a shared expert pool."
                    )
                return num_columns // shared_pool_modalities
            return num_columns
    return None


def _collapse_shared_expert_pool(gate_weights: torch.Tensor, num_modalities: int) -> torch.Tensor:
    """Average each expert's gate weights across modality routers.

    Gate columns are grouped by modality. Leave weights unchanged for one modality.
    """
    return collapse_shared_expert_pool(gate_weights, num_modalities)


def _resolve_expert_names(
    num_experts: int,
    expert_names: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if expert_names is None:
        return tuple(str(index) for index in range(num_experts))
    if len(expert_names) != num_experts:
        raise ValueError("expert_names must match the gate weight expert count.")
    return tuple(str(name) for name in expert_names)


def _empty_scenario_output() -> dict[str, list[torch.Tensor] | int]:
    return {
        "label": [],
        "probability": [],
        "prediction": [],
        "gate_weights": [],
        "router_output_norms": [],
        "excluded_n": 0,
    }


def _scenario_lookup(
    scenarios: Iterable[ModalityScenario] | None,
) -> Mapping[str, ModalityScenario]:
    if scenarios is None:
        return SCENARIO_BY_NAME
    return {scenario.name: scenario for scenario in scenarios}


def _scenario_modality_names(scenarios: Sequence[ModalityScenario]) -> tuple[str, ...]:
    """Return the active modality order represented by a scenario collection."""
    for scenario in scenarios:
        if scenario.name == NO_ABLATION_SCENARIO:
            return scenario.modalities
    return canonical_modalities(
        tuple(
            dict.fromkeys(modality for scenario in scenarios for modality in scenario.modalities)
        )
    )


def _scenario_label(
    scenario_name: str,
    scenario_lookup: Mapping[str, ModalityScenario] | None = None,
) -> str:
    lookup = SCENARIO_BY_NAME if scenario_lookup is None else scenario_lookup
    return lookup[scenario_name].label


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    fusion_method: str | None = None,
) -> dict[str, Path]:
    """Run modality ablation for one checkpoint and return the output paths.

    Read config.yaml from the run directory by default. Write to modality_ablation
    below output_dir, or below the run directory when output_dir is omitted.
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
    dataloaders = get_dataloaders(cfg, splits=("val", "test"))
    model_cfg = cfg.get("model", {})
    if fusion_method is None and isinstance(model_cfg, Mapping):
        fusion_method = str(
            model_cfg.get("fusion_method", model_cfg.get("name", "concat"))
        ).lower()
    if fusion_method is None:
        fusion_method = "concat"

    inner_model = getattr(model, "model", model)
    expert_names = None
    shared_pool_modalities = None
    if fusion_method in EXPERT_METHODS:
        expert_names = moe_expert_names(inner_model)
        # Average gate weights across modality routers to match the training expert summary
        shared_pool_modalities = shared_expert_pool_modalities(inner_model)

    # Use the model modality order and task defaults
    scenarios = resolve_modality_ablation_scenarios(
        inner_model,
        cfg,
        dataloader=dataloaders["test"],
    )

    output_dir = run_dir if output_dir is None else Path(output_dir)
    return run_modality_ablation(
        model,
        dataloaders["test"],
        output_dir / "modality_ablation",
        scenarios,
        fusion_method=fusion_method,
        expert_names=expert_names,
        shared_pool_modalities=shared_pool_modalities,
        val_dataloader=dataloaders.get("val"),
    )


def resolve_modality_ablation_scenarios(
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    *,
    dataloader: DataLoader | None = None,
) -> tuple[ModalityScenario, ...]:
    """Choose default scenarios from the task and active modalities.

    Use all subsets for deterioration and individual or leave-one-out modalities for
    hospitalisation. Read data.task first, falling back to the outcome for older configs.
    """
    inner_model = getattr(model, "model", model)
    active = _resolve_active_modalities(inner_model, cfg, dataloader)
    if _is_hospitalisation_config(cfg):
        return hospitalisation_default_scenarios(active)
    return scenarios_for_active_modalities(active)


def _default_scenarios_for_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> tuple[ModalityScenario, ...]:
    """Resolve defaults for direct API callers without a saved config."""
    inner_model = getattr(model, "model", model)
    candidates = (
        getattr(inner_model, "active_modalities", None),
        getattr(inner_model, "modality_names", None),
        getattr(getattr(dataloader, "dataset", None), "modality_names", None),
    )
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, str):
            candidate = (candidate,)
        active = canonical_modalities(tuple(str(modality) for modality in candidate))
        if any(modality in HOSPITALISATION_MODALITIES for modality in active):
            return hospitalisation_default_scenarios(active)
        return scenarios_for_active_modalities(active)
    return SCENARIOS


def _resolve_active_modalities(
    inner_model: torch.nn.Module,
    cfg: Mapping[str, Any],
    dataloader: DataLoader | None,
) -> tuple[str, ...]:
    """Resolve the modality order used by the model and its data loader."""
    candidates: list[Iterable[str] | None] = [
        getattr(inner_model, "active_modalities", None),
        getattr(inner_model, "modality_names", None),
    ]
    model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(model_cfg, Mapping):
        candidates.append(model_cfg.get("modality_names"))
    if isinstance(data_cfg, Mapping):
        candidates.append(data_cfg.get("modalities"))
    dataset = getattr(dataloader, "dataset", None)
    candidates.append(getattr(dataset, "modality_names", None))

    for candidate in candidates:
        if candidate:
            if isinstance(candidate, str):
                candidate = (candidate,)
            return canonical_modalities(tuple(str(modality) for modality in candidate))

    ts_cfg = (
        model_cfg.get("ts_modality")
        if isinstance(model_cfg, Mapping)
        else None
    )
    if ts_cfg is None and isinstance(cfg, Mapping):
        ts_cfg = cfg.get("ts_modality")
    ts_enabled = bool(ts_cfg.get("enabled", False)) if isinstance(ts_cfg, Mapping) else False
    return MODALITIES if ts_enabled else MODALITIES[:3]


def _is_hospitalisation_config(cfg: Mapping[str, Any]) -> bool:
    """Return whether a saved config belongs to the hospitalisation task."""
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(data_cfg, Mapping):
        task = data_cfg.get("task", data_cfg.get("dataset"))
        if task is not None:
            return str(task).lower() == "hospitalisation"

    outcome = (
        data_cfg.get("outcome")
        if isinstance(data_cfg, Mapping)
        else None
    ) or cfg.get("outcome", "")
    return str(outcome).lower().startswith("hospitalisation_")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.checkpoint is not None:
        paths = evaluate_checkpoint(
            args.checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
            fusion_method=args.fusion_method,
        )
        run_dir = Path(args.checkpoint).parent
        output_dir = run_dir if args.output_dir is None else Path(args.output_dir)
        print(f"Modality ablation outputs stored at: {output_dir / 'modality_ablation'}")
        for path in paths.values():
            print(path)
        return

    checkpoints = discover_checkpoints(args.path)
    if not checkpoints:
        print(
            f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {args.path}.",
        )
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {args.path}.")
    failures: list[tuple[Path, str]] = []
    evaluated = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        existing = run_dir / "modality_ablation" / "test_modality_ablation_metrics.csv"
        if args.skip_existing and existing.exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
        try:
            evaluate_checkpoint(
                checkpoint,
                config_path=args.config,
                fusion_method=args.fusion_method,
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
