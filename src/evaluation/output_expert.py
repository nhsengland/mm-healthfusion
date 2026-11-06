"""Write expert gate weights and usage summaries for saved checkpoints.

Evaluate the test split and write results to the expert directory.

Usage:
    python -m src.evaluation.output_expert --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.output_expert --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from src.data.collate import batch_modality_availability
from src.data.splits import get_dataloaders
from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    _usable_device,
    discover_checkpoints,
    load_config,
)
from src.training.lightning import MMModel


# Fusion methods whose predictions include expert gate weights
EXPERT_METHODS = {"fusemoe", "smp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write MoE expert diagnostics for a single checkpoint (--checkpoint) "
            "or for every run found under a directory tree (--path)."
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
            "Directory whose 'expert' subfolder receives the diagnostics. "
            "Defaults to the checkpoint's run directory (its parent folder). "
            "Ignored in --path mode, where each run writes to its own directory."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --path mode, skip runs that already have an "
            "expert/test_expert_summary.csv (useful to resume a batch)."
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


def configured_fusion_method(cfg: Mapping[str, object]) -> str:
    """Return the fusion method declared in ``config.yaml``."""
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return "concat"
    return str(model_cfg.get("fusion_method", model_cfg.get("name", "concat"))).lower()


def resolve_expert_output_dir(run_dir: str | Path, fusion_method: str) -> Path:
    """Return the expert diagnostics folder for MoE fusion methods."""
    if fusion_method.lower() in EXPERT_METHODS:
        return Path(run_dir) / "expert"
    return Path(run_dir)


def uses_expert_modality_availability(fusion_method: str) -> bool:
    return fusion_method.lower() in EXPERT_METHODS


def routing_moe(model: torch.nn.Module) -> torch.nn.Module | None:
    """Find the mixture-of-experts module used for routing.

    Check model.fusion first, then nested encoder modules. Fall back to model.fusion
    when no routing module is found.
    """
    fusion = getattr(model, "fusion", None)
    if fusion is not None and hasattr(fusion, "expert_subsets"):
        return fusion
    for module in model.modules():
        if hasattr(module, "expert_subsets") and hasattr(module, "router_type"):
            return module
    return fusion


def moe_expert_names(model: torch.nn.Module) -> tuple[str, ...] | None:
    fusion = routing_moe(model)
    if fusion is None:
        return None
    num_experts = int(getattr(fusion, "num_experts"))
    if getattr(fusion, "router_type", None) == "permod":
        # Name the shared experts after averaging their per-modality gate weights
        return tuple(f"shared_expert_{index}" for index in range(num_experts))
    if getattr(fusion, "expert_structure", "shared") == "subset":
        return tuple(subset.name for subset in getattr(fusion, "expert_subsets"))
    return tuple("shared" for _ in range(num_experts))


def shared_expert_pool_modalities(model: torch.nn.Module) -> int | None:
    """Return the number of modality routers sharing a permod expert pool.

    Return None for other router types, which need no averaging across modalities.
    """
    fusion = routing_moe(model)
    if fusion is None:
        return None
    if getattr(fusion, "router_type", None) != "permod":
        return None
    return int(getattr(fusion, "num_modalities"))


def collapse_shared_expert_pool(gate_weights: torch.Tensor, num_modalities: int) -> torch.Tensor:
    """Average each expert's gate weights across modality routers.

    Columns are grouped by modality, with num_experts columns per group. Averaging
    keeps each row's weights summing to one. Leave weights unchanged for one modality.
    """
    if num_modalities <= 1:
        return gate_weights
    n_samples, n_columns = gate_weights.shape
    if n_columns % num_modalities != 0:
        raise ValueError(
            "gate weight columns must be divisible by num_modalities to collapse "
            f"a shared expert pool. Got {n_columns} columns for {num_modalities} "
            "modalities."
        )
    num_experts = n_columns // num_modalities
    return gate_weights.reshape(n_samples, num_modalities, num_experts).mean(dim=1)


def collect_modality_availability_batches(
    dataloader: Iterable[Mapping[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Return natural modality availability for each dataloader batch."""
    return [batch_modality_availability(batch, detach_cpu=True) for batch in dataloader]


def write_moe_expert_outputs(
    prediction_batches: list[dict[str, torch.Tensor]],
    gate_weights_path: str | Path,
    expert_summary_path: str | Path,
    modality_expert_summary_path: str | Path | None = None,
    modality_availability_batches: list[dict[str, torch.Tensor]] | None = None,
    expert_names: list[str] | tuple[str, ...] | None = None,
    shared_pool_modalities: int | None = None,
    router_contribution_summary_path: str | Path | None = None,
) -> dict[str, Path]:
    """Write expert summaries when prediction batches contain gate weights.

    For shared pools, average each expert's weights across modality routers. When
    router_contribution_summary_path is supplied and batches include
    router_output_norms, also write each router's output magnitude and share.
    Count all experts tied for the largest positive weight as top experts. Export their
    pipe-separated IDs in top_experts and fill top_expert only for a unique winner.
    """
    gate_batches = [
        batch["gate_weights"].detach().cpu()
        for batch in prediction_batches
        if "gate_weights" in batch
    ]
    if not gate_batches:
        return {}

    gate_weights = torch.cat(gate_batches, dim=0)
    if gate_weights.ndim != 2:
        raise ValueError("gate_weights must have shape (n_samples, n_experts).")
    if shared_pool_modalities is not None:
        gate_weights = collapse_shared_expert_pool(gate_weights, shared_pool_modalities)

    gate_weights_path = Path(gate_weights_path)
    expert_summary_path = Path(expert_summary_path)
    gate_weights_path.parent.mkdir(parents=True, exist_ok=True)
    expert_summary_path.parent.mkdir(parents=True, exist_ok=True)
    identifier_column = next(
        (
            column
            for column in ("hadm_id", "ed_stay_id")
            if all(column in batch for batch in prediction_batches)
        ),
        None,
    )
    if identifier_column is None:
        raise ValueError(
            "Prediction batches must consistently contain hadm_id or ed_stay_id."
        )
    paths = {
        "test_expert_gate_weights_path": gate_weights_path,
        "test_expert_summary_path": expert_summary_path,
    }
    modality_availability = (
        _aligned_modality_availability(
            prediction_batches,
            modality_availability_batches,
        )
        if modality_availability_batches is not None
        else None
    )

    num_experts = int(gate_weights.shape[1])
    expert_names = _resolve_expert_names(num_experts, expert_names)
    top_expert_mask = _top_expert_mask(gate_weights)
    expert_columns = [f"expert_{expert_index}_gate_weight" for expert_index in range(num_experts)]
    modality_columns = {
        modality: f"has_{modality}"
        for modality in (modality_availability or {})
    }
    fieldnames = [
        identifier_column,
        "label",
        "probability",
        "prediction",
        *modality_columns.values(),
        "top_expert",
        "top_experts",
        *expert_columns,
    ]

    with gate_weights_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        row_offset = 0
        for batch in prediction_batches:
            if "gate_weights" not in batch:
                continue
            batch_size = int(batch["gate_weights"].shape[0])
            row_start = row_offset
            batch_gate_weights = gate_weights[row_start : row_start + batch_size]
            row_offset += batch_size
            batch_top_expert_mask = top_expert_mask[row_start : row_start + batch_size]

            for index in range(batch_size):
                availability_index = row_start + index
                top_experts = batch_top_expert_mask[index].nonzero(as_tuple=True)[0].tolist()
                row = {
                    identifier_column: int(batch[identifier_column][index].item()),
                    "label": int(batch["label"][index].item()),
                    "probability": float(batch["probability"][index].item()),
                    "prediction": int(batch["prediction"][index].item()),
                    "top_expert": top_experts[0] if len(top_experts) == 1 else "",
                    "top_experts": "|".join(str(expert) for expert in top_experts),
                }
                row.update(
                    {
                        column: (
                            ""
                            if modality_availability is None
                            else int(
                                modality_availability[modality][
                                    availability_index
                                ].item()
                            )
                        )
                        for modality, column in modality_columns.items()
                    }
                )
                row.update(
                    {
                        column: float(batch_gate_weights[index, expert_index].item())
                        for expert_index, column in enumerate(expert_columns)
                    }
                )
                writer.writerow(row)

    top_expert_counts = top_expert_mask.sum(dim=0)
    nonzero_gate_counts = gate_weights.gt(0).sum(dim=0)
    mean_gate_weights = gate_weights.mean(dim=0)
    mean_aux_loss = _mean_prediction_aux_loss(prediction_batches)
    summary_fieldnames = [
        "expert", "expert_name", "mean_gate_weight", "nonzero_gate_count", "nonzero_gate_fraction",
        "top_expert_count", "top_expert_fraction", "mean_aux_loss",
    ]
    with expert_summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fieldnames)
        writer.writeheader()
        for expert_index in range(num_experts):
            writer.writerow(
                {
                    "expert": expert_index,
                    "expert_name": expert_names[expert_index],
                    "mean_gate_weight": float(mean_gate_weights[expert_index].item()),
                    "nonzero_gate_count": int(
                        nonzero_gate_counts[expert_index].item()
                    ),
                    "nonzero_gate_fraction": (
                        float(nonzero_gate_counts[expert_index].item())
                        / float(gate_weights.shape[0])
                    ),
                    "top_expert_count": int(top_expert_counts[expert_index].item()),
                    "top_expert_fraction": (
                        float(top_expert_counts[expert_index].item())
                        / float(gate_weights.shape[0])
                    ),
                    "mean_aux_loss": "" if mean_aux_loss is None else mean_aux_loss,
                }
            )

    if (
        modality_expert_summary_path is not None
        and modality_availability is not None
    ):
        modality_expert_summary_path = Path(modality_expert_summary_path)
        _write_modality_expert_summary_csv(
            gate_weights,
            modality_availability,
            modality_expert_summary_path,
            expert_names,
        )
        paths["test_modality_expert_summary_path"] = (
            modality_expert_summary_path
        )

    if (
        router_contribution_summary_path is not None
        and modality_availability is not None
    ):
        router_contribution_summary_path = Path(router_contribution_summary_path)
        written = _write_router_contribution_summary_csv(
            prediction_batches,
            modality_availability,
            router_contribution_summary_path,
        )
        if written:
            paths["test_router_contribution_summary_path"] = (
                router_contribution_summary_path
            )

    return paths


def _top_expert_mask(gate_weights: torch.Tensor) -> torch.Tensor:
    """Include every expert tied for the largest positive weight in each row."""
    return gate_weights.eq(gate_weights.max(dim=1, keepdim=True).values) & gate_weights.gt(0)


def _resolve_expert_names(
    num_experts: int,
    expert_names: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if expert_names is None:
        return tuple("shared" for _ in range(num_experts))
    if len(expert_names) != num_experts:
        raise ValueError("expert_names must match the gate weight expert count.")
    return tuple(str(name) for name in expert_names)


def _aligned_modality_availability(
    prediction_batches: list[dict[str, torch.Tensor]],
    modality_availability_batches: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if len(prediction_batches) != len(modality_availability_batches):
        raise ValueError(
            "modality_availability_batches must align with prediction_batches."
        )

    # Keep the configured modality order from the batch
    present_modalities = list(modality_availability_batches[0]) if modality_availability_batches else []
    aligned: dict[str, list[torch.Tensor]] = {
        modality: [] for modality in present_modalities
    }
    for prediction_batch, availability_batch in zip(
        prediction_batches,
        modality_availability_batches,
        strict=True,
    ):
        if "gate_weights" not in prediction_batch:
            continue
        batch_size = int(prediction_batch["gate_weights"].shape[0])
        for modality in aligned:
            availability = (
                availability_batch[modality]
                .detach()
                .cpu()
                .reshape(-1)
                .to(dtype=torch.bool)
            )
            if int(availability.numel()) != batch_size:
                raise ValueError(
                    "modality availability batch size must match gate weights."
                )
            aligned[modality].append(availability)

    if not aligned or not next(iter(aligned.values())):
        raise ValueError("No modality availability batches aligned with gate weights.")

    return {
        modality: torch.cat(values, dim=0)
        for modality, values in aligned.items()
    }


def _write_router_contribution_summary_csv(
    prediction_batches: list[dict[str, torch.Tensor]],
    modality_availability: Mapping[str, torch.Tensor],
    output_path: str | Path,
) -> bool:
    """Write each router's output magnitude and share of the combined output.

    Use router_output_norms for permod and disjoint routing. Include only present
    modalities in each row and its total, since masked inputs can still produce a
    non-zero router output. Return False without writing when norms are absent.
    """
    norm_batches = [
        batch["router_output_norms"].detach().cpu()
        for batch in prediction_batches
        if "router_output_norms" in batch
    ]
    if not norm_batches:
        return False

    norms = torch.cat(norm_batches, dim=0).to(dtype=torch.float64)
    if norms.ndim != 2 or norms.shape[1] <= 0:
        raise ValueError(
            "router_output_norms must have shape (n_samples, num_modalities)."
        )

    modality_order = tuple(modality_availability)[: norms.shape[1]]
    missing = [modality for modality in modality_order if modality not in modality_availability]
    if missing:
        raise ValueError(
            f"modality availability is missing routers: {', '.join(missing)}."
        )

    availability = torch.stack(
        [modality_availability[modality].to(dtype=torch.bool) for modality in modality_order],
        dim=1,
    )
    if availability.shape[0] != norms.shape[0]:
        raise ValueError("modality availability must match router norm rows.")

    # Calculate each present router's share of the output magnitude
    present = availability.to(dtype=norms.dtype)
    present_norms = norms * present
    denom = present_norms.sum(dim=1, keepdim=True).clamp_min(torch.finfo(norms.dtype).tiny)
    contribution_fraction = present_norms / denom

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "router", "router_name", "mean_output_norm", "mean_contribution_fraction", "eligible_n",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for router_index, modality in enumerate(modality_order):
            eligible = availability[:, router_index]
            eligible_n = int(eligible.sum().item())
            if eligible_n == 0:
                writer.writerow(
                    {
                        "router": router_index,
                        "router_name": modality,
                        "mean_output_norm": "",
                        "mean_contribution_fraction": "",
                        "eligible_n": 0,
                    }
                )
                continue
            writer.writerow(
                {
                    "router": router_index,
                    "router_name": modality,
                    "mean_output_norm": float(
                        norms[eligible, router_index].mean().item()
                    ),
                    "mean_contribution_fraction": float(
                        contribution_fraction[eligible, router_index].mean().item()
                    ),
                    "eligible_n": eligible_n,
                }
            )
    return True


def _write_modality_expert_summary_csv(
    gate_weights: torch.Tensor,
    modality_availability: Mapping[str, torch.Tensor],
    output_path: str | Path,
    expert_names: list[str] | tuple[str, ...] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gate_weights.ndim != 2:
        raise ValueError("gate_weights must have shape (n_samples, n_experts).")

    modalities = list(modality_availability)
    availability = torch.stack(
        [modality_availability[modality].to(dtype=torch.bool) for modality in modalities],
        dim=1,
    )
    if availability.shape[0] != gate_weights.shape[0]:
        raise ValueError("modality availability must match gate weight rows.")

    fieldnames = [
        "modalities", *(f"has_{modality}" for modality in modalities), "expert", "expert_name",
        "eligible_n", "mean_gate_weight", "nonzero_gate_count", "nonzero_gate_fraction",
        "top_expert_count", "top_expert_fraction",
    ]
    num_experts = int(gate_weights.shape[1])
    expert_names = _resolve_expert_names(num_experts, expert_names)
    seen_combinations = [
        tuple(int(value) for value in row)
        for row in torch.unique(
            availability.to(dtype=torch.int64),
            dim=0,
        ).tolist()
    ]
    seen_combinations.sort()
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for combination in seen_combinations:
            group_mask = (
                availability
                == torch.tensor(combination, dtype=torch.bool).unsqueeze(0)
            ).all(dim=1)
            group_gate_weights = gate_weights[group_mask]
            eligible_n = int(group_gate_weights.shape[0])
            top_expert_counts = _top_expert_mask(group_gate_weights).sum(dim=0)
            nonzero_gate_counts = group_gate_weights.gt(0).sum(dim=0)
            mean_gate_weights = group_gate_weights.mean(dim=0)
            for expert_index in range(num_experts):
                writer.writerow(
                    {
                        "modalities": _modality_combination_label(
                            combination, modalities
                        ),
                        **{
                            f"has_{modality}": combination[index]
                            for index, modality in enumerate(modalities)
                        },
                        "expert": expert_index,
                        "expert_name": expert_names[expert_index],
                        "eligible_n": eligible_n,
                        "mean_gate_weight": float(
                            mean_gate_weights[expert_index].item()
                        ),
                        "nonzero_gate_count": int(
                            nonzero_gate_counts[expert_index].item()
                        ),
                        "nonzero_gate_fraction": (
                            float(nonzero_gate_counts[expert_index].item())
                            / float(eligible_n)
                        ),
                        "top_expert_count": int(
                            top_expert_counts[expert_index].item()
                        ),
                        "top_expert_fraction": (
                            float(top_expert_counts[expert_index].item())
                            / float(eligible_n)
                        ),
                    }
                )


def _modality_combination_label(
    combination: tuple[int, ...],
    modalities: tuple[str, ...] | list[str],
) -> str:
    labels = [name for name, present in zip(modalities, combination, strict=True) if present]
    return "+".join(labels) if labels else "none"


def _mean_prediction_aux_loss(prediction_batches: list[dict[str, torch.Tensor]]) -> float | None:
    aux_losses = [
        batch["aux_loss"].detach().cpu().reshape(-1).to(dtype=torch.float32)
        for batch in prediction_batches
        if "aux_loss" in batch
    ]
    if not aux_losses:
        return None
    return float(torch.cat(aux_losses).mean().item())


def write_checkpoint_expert_outputs(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    output_dir: str | Path,
    *,
    fusion_method: str,
    expert_names: list[str] | tuple[str, ...] | None = None,
    shared_pool_modalities: int | None = None,
) -> dict[str, Path]:
    """Predict the test split and write expert summaries below output_dir/expert.

    Write gate weights, expert usage, modality summaries and available router
    contributions. Return an empty mapping for models without expert gates.
    """
    if fusion_method.lower() not in EXPERT_METHODS:
        return {}

    inner_model = getattr(model, "model", model)
    if expert_names is None:
        expert_names = moe_expert_names(inner_model)
    if shared_pool_modalities is None:
        shared_pool_modalities = shared_expert_pool_modalities(inner_model)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    previous_training_state = model.training
    model.eval()
    prediction_batches: list[dict[str, torch.Tensor]] = []
    modality_availability_batches: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            modality_availability_batches.append(
                batch_modality_availability(batch, detach_cpu=True)
            )
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            prediction = model.predict_step(batch, batch_idx)
            prediction_batches.append(
                {
                    key: value.detach().cpu()
                    for key, value in prediction.items()
                    if isinstance(value, torch.Tensor)
                }
            )
    if previous_training_state:
        model.train()
    # Write standalone summaries under expert, as training does
    expert_output_dir = Path(output_dir) / "expert"
    return write_moe_expert_outputs(
        prediction_batches,
        expert_output_dir / "test_expert_gate_weights.csv",
        expert_output_dir / "test_expert_summary.csv",
        expert_output_dir / "test_modality_expert_summary.csv",
        modality_availability_batches,
        expert_names,
        shared_pool_modalities=shared_pool_modalities,
        router_contribution_summary_path=(
            expert_output_dir / "test_router_contribution_summary.csv"
        ),
    )


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write expert summaries for one checkpoint and return their paths.

    Read config.yaml from the run directory by default. Write to expert below
    output_dir, or below the run directory when output_dir is omitted.
    """
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    config_path = (
        run_dir / "config.yaml" if config_path is None else Path(config_path)
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_config(config_path)
    fusion_method = configured_fusion_method(cfg)
    if fusion_method not in EXPERT_METHODS:
        raise ValueError(
            f"Fusion method '{fusion_method}' has no MoE expert outputs. Expected "
            f"one of {sorted(EXPERT_METHODS)}."
        )

    # Check that CUDA can start before loading the checkpoint
    map_location = _usable_device()
    model = MMModel.load_from_checkpoint(str(checkpoint), map_location=map_location)

    dataloaders = get_dataloaders(cfg, splits=("test",))
    output_dir = run_dir if output_dir is None else Path(output_dir)
    return write_checkpoint_expert_outputs(
        model,
        dataloaders["test"],
        output_dir,
        fusion_method=fusion_method,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.checkpoint is not None:
        paths = evaluate_checkpoint(
            args.checkpoint,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        run_dir = Path(args.checkpoint).parent
        output_dir = run_dir if args.output_dir is None else Path(args.output_dir)
        print(f"Expert diagnostics stored at: {output_dir / 'expert'}")
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
    skipped = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir = checkpoint.parent
        existing = run_dir / "expert" / "test_expert_summary.csv"
        if args.skip_existing and existing.exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue

        try:
            run_config = run_dir / "config.yaml"
            if not run_config.exists() and args.config is not None:
                run_config = Path(args.config)
            configured_method = configured_fusion_method(load_config(run_config))
            if configured_method not in EXPERT_METHODS:
                skipped += 1
                print(
                    f"[{index}/{len(checkpoints)}] skip (non-MoE: "
                    f"{configured_method}): {run_dir}"
                )
                continue

            print(f"[{index}/{len(checkpoints)}] evaluating: {run_dir}")
            evaluate_checkpoint(checkpoint, config_path=run_config)
            evaluated += 1
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not args.continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(
        f"Done. Evaluated {evaluated}/{len(checkpoints)} run(s), "
        f"skipped {skipped} non-MoE run(s)."
    )
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


if __name__ == "__main__":
    main()
