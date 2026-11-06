"""Create chest X-ray heatmaps for a saved model.

Read the chosen patient split and save heatmaps and a manifest under the run
directory. Hospitalisation runs explain each configured chest X-ray input.
Patients without an available image receive no heatmap.

Run from the project root:

    python -m src.explanation.explain_cxr --path /path/to/run
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.splits import resolve_split_parquet
from src.evaluation.eval_utils import _usable_device, load_config, resolve_run
from src.explanation.freeze_m3care import use_frozen_graph
from src.explanation.gradcam import (
    CXR_NORMALISATION,
    DEFAULT_CXR_ROOT,
    CXRGradCAM,
    cxr_available,
    cxr_modalities,
    gradcam_config,
    train_identifiers,
)
from src.explanation.shap_gradcam import (
    live_cxr_logit_fn,
    modality_shapley,
    modality_shapley_frozen,
    patient_cxr_context,
    shap_weighted_cam,
)
from src.explanation.shap_utils import (
    DEFAULT_SHAP_SEED,
    SplitBatches,
    estimate_mm_summary,
    git_revision,
)

# Report input contributions on the probability scale
MODALITY_CONTRIBUTION_SCALE = "probability"


def parse_args(argv=None):
    """Read command-line options for a run and its heatmaps."""
    parser = argparse.ArgumentParser(description="Create chest X-ray heatmaps from a saved run.")
    # Choose the run and patients
    parser.add_argument("--path", required=True,
                        help="Run folder containing checkpoint_best.ckpt and config.yaml.")
    parser.add_argument("--stays", type=int, nargs="+", default=None,
                        help="Stay IDs to explain. By default, use the first ten patients "
                             "in the split's saved order.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="Patient split to explain (default: test).")

    # Locate image files and saved features
    parser.add_argument("--cxr-root", default=DEFAULT_CXR_ROOT,
                        help=f"Chest X-ray image folder (default: {DEFAULT_CXR_ROOT}).")
    parser.add_argument("--raw-pkl", default=None,
                        help="Raw saved image features. By default, find them from the run "
                             "configuration.")
    parser.add_argument("--keys-csv", default=None,
                        help="CSV linking patients to images. By default, find it from the "
                             "run configuration.")

    # Choose where to save images
    parser.add_argument("--out-dir", default=None,
                        help="Folder for heatmaps and manifest (default: <path>/gradcam).")
    parser.add_argument("--overlay-alpha", type=float, default=0.4,
                        help="Heatmap opacity over the X-ray (default: 0.4).")

    # Set options for SHAP-weighted heatmaps
    parser.add_argument("--shap-weighted", action="store_true",
                        help="Also write heatmaps weighted by SHAP values.")
    parser.add_argument("--explainer", default="gradient", choices=("kernel", "gradient"),
                        help="Method for weighting image features (default: gradient).")
    parser.add_argument("--kernel-nsamples", default="auto",
                        help="Number of samples for kernel SHAP, or auto (default: auto).")
    parser.add_argument("--shap-background", type=int, default=50,
                        help="Number of training patients used as the SHAP reference "
                             "(default: 50). Use 0 only to reproduce older heatmaps.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SHAP_SEED,
                        help=f"Random seed for SHAP sampling (default: {DEFAULT_SHAP_SEED}).")

    return parser.parse_args(argv)


def _run_details(args, device: str) -> dict:
    """Record settings needed to interpret the images and manifest.

    Include SHAP settings only when weighted heatmaps were requested.
    """
    details = {
        **git_revision(),
        "modality_contribution_scale": MODALITY_CONTRIBUTION_SCALE,
        "split": args.split,
        "normalise": CXR_NORMALISATION,
        "overlay_alpha": args.overlay_alpha,
        "device": device,
        "stays": list(args.stays),
    }
    if args.shap_weighted:
        details["shap"] = {
            "shap_weighted": True,
            "explainer": args.explainer,
            "kernel_nsamples": args.kernel_nsamples,
            "background_size": args.shap_background,
            # Fix the background draw and SHAP sampling across repeated runs
            "seed": args.seed,
            # Mark runs that use an all-zero image reference
            "zero_baseline": args.shap_background == 0,
        }
    return details


def main(argv=None) -> None:
    """Create heatmaps and a manifest for the chosen patients."""
    args = parse_args(argv)
    checkpoint_path, config_path = resolve_run(args.path)
    device = _usable_device()

    run_config = load_config(config_path)
    task = str((run_config.get("data") or {}).get("task", "deterioration")).lower()
    parquet = resolve_split_parquet(run_config, args.split)
    try:
        batches = SplitBatches(config_path, args.split)
    except Exception as exc:  # warn before using one-patient batches
        if task == "hospitalisation":
            raise SystemExit(
                f"Hospitalisation Grad-CAM needs the {args.split} split to preserve all "
                f"seven modality inputs: {type(exc).__name__}: {exc}"
            ) from exc
        warnings.warn(
            f"Could not load the {args.split} split ({type(exc).__name__}: {exc}); falling "
            "back to parquet-derived single-stay batches.",
            RuntimeWarning, stacklevel=2,
        )
        batches = None
    cxr_inputs = cxr_modalities(batches)
    if not cxr_inputs:
        raise SystemExit("This run has no configured CXR input to explain.")
    if args.raw_pkl and len(cxr_inputs) > 1:
        raise SystemExit(
            "--raw-pkl can only override a run with one configured CXR input. "
            "For multiple CXR inputs, use the files derived from the run config."
        )

    if args.stays is None:
        if batches is not None:
            args.stays = batches.first_identifiers(10)
        elif parquet is not None:
            # Only deterioration falls back to the parquet, keyed by ed_stay_id
            try:
                args.stays = [
                    int(x)
                    for x in pd.read_parquet(parquet, columns=["ed_stay_id"])["ed_stay_id"].head(10)
                ]
            except Exception as exc:
                raise SystemExit(
                    f"Could not choose patients from {parquet} "
                    f"({type(exc).__name__}: {exc}). Pass --stays."
                ) from exc
        else:
            raise SystemExit(
                f"Could not load the {args.split} split to choose patients. Pass --stays."
            )

    train_ids = train_identifiers(config_path, task)

    root_out_dir = Path(args.out_dir) if args.out_dir else Path(args.path) / "gradcam"
    root_out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for cxr_modality in cxr_inputs:
        cfg = gradcam_config(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            cxr_root=args.cxr_root,
            cxr_modality=cxr_modality,
            batches=batches,
            train_identifiers=train_ids,
            device=device,
            raw_pkl=args.raw_pkl,
            keys_csv=args.keys_csv,
        )
        identifier_column = cfg.identifier_column
        g = CXRGradCAM(cfg)
        modalities = tuple(getattr(g.model, "modality_names", ("tabular", "ecg", "cxr")))
        if parquet:
            g.attach_context_parquet(parquet)
        use_frozen = use_frozen_graph(g.mm)
        modality_out_dir = (
            root_out_dir / cxr_modality
            if len(cxr_inputs) > 1 else root_out_dir
        )
        modality_out_dir.mkdir(parents=True, exist_ok=True)

        for identifier in args.stays:
            entry = {
                identifier_column: identifier,
                "cxr_modality": cxr_modality,
                "fusion_method": g.fusion_method,
                "outputs": [],
            }
            modality_contribution = None
            try:
                idx = batches.index_of(identifier) if batches is not None else None
                if task == "hospitalisation" and idx is None:
                    raise ValueError(
                        f"{identifier_column} {identifier} is not in the {args.split} split."
                    )
                if idx is not None and not cxr_available(batches, idx, cxr_modality):
                    raise ValueError(
                        f"{cxr_modality} is unavailable for {identifier_column} {identifier}."
                    )
                paths = g.resolve_stay_images(identifier)
                if not paths:
                    raise ValueError(
                        f"{cxr_modality} has no resolvable CXR images for "
                        f"{identifier_column} {identifier}."
                    )
                stay_emb = g.stay_embedding(paths)
                context_row = g.context_from_parquet(identifier)
                scorer = target_idx = None
                if idx is not None:
                    patient = patient_cxr_context(g, batches, idx, stay_emb, use_frozen)
                    logit_fn, mm_batch = patient.logit_fn, patient.batch
                    scorer, target_idx = patient.scorer, patient.target_idx
                    if patient.co_batch is not None:
                        entry["graph_batch_size"] = batches.batch_size
                        entry["co_batch"] = list(patient.co_batch)
                else:
                    # Without the split, score from the saved parquet context
                    mm_batch = g._base_batch_from_row(stay_emb[None, :], context_row)
                    logit_fn = live_cxr_logit_fn(g, mm_batch, 0, cxr_modality)

                # Add input contributions to every overlay title
                if scorer is not None:
                    ms = modality_shapley_frozen(
                        scorer, target_idx, modalities=modalities,
                        target_modality=cxr_modality,
                    )
                else:
                    ms = modality_shapley(
                        g, mm_batch, modalities=modalities,
                        target_modality=cxr_modality,
                    )
                cams = g.heatmap_for_stay(
                    identifier, context_row=context_row, logit_fn=logit_fn,
                )
                phi = ms.values
                contribution = estimate_mm_summary(
                    {
                        modality: np.array([value], dtype=np.float64)
                        for modality, value in phi.items()
                    },
                    modalities=list(phi),
                )["global_contribution"]
                modality_contribution = {
                    modality: float(contribution[modality])
                    for modality in phi
                }
                entry["modality_contribution"] = modality_contribution
                entry["cxr_fraction"] = float(ms.cxr_fraction)
            except (KeyError, ValueError) as exc:
                entry["error"] = str(exc)
                manifest.append(entry)
                print(f"{identifier_column} {identifier}: {exc}")
                continue

            for image_index, cam in enumerate(cams):
                stem = f"{identifier_column.removesuffix('_id')}{identifier}"
                png = modality_out_dir / (
                    f"{stem}_img{image_index}_{cam.dicom_id}.png"
                )
                g.overlay(
                    cam,
                    str(png),
                    alpha=args.overlay_alpha,
                    modality_contribution=modality_contribution,
                )
                record = {
                    "image_index": image_index,
                    "dicom_id": cam.dicom_id,
                    "prob": cam.prob,
                    "logit": cam.logit,
                    "overlay": str(png),
                }
                entry["outputs"].append(record)

                if args.shap_weighted:
                    raws = [g.raw_gap_from_image(path) for path in paths]
                    others = (
                        torch.stack(
                            [
                                embedding for other_index, embedding in enumerate(raws)
                                if other_index != image_index
                            ],
                            dim=0,
                        )
                        if len(paths) > 1 else None
                    )
                    nsamples = args.kernel_nsamples
                    if nsamples != "auto":
                        nsamples = int(nsamples)
                    weighted = shap_weighted_cam(
                        g,
                        paths[image_index],
                        context_row=g.context_from_parquet(identifier),
                        other_raw_embeddings=others,
                        explainer=args.explainer,
                        kernel_nsamples=nsamples,
                        background_size=args.shap_background,
                        shap_seed=args.seed,
                        background_seed=args.seed,
                        logit_fn=logit_fn,
                    )
                    weighted_png = png.with_name(
                        f"{png.stem}_shapweighted.png"
                    )
                    g.overlay(
                        weighted,
                        str(weighted_png),
                        alpha=args.overlay_alpha,
                        modality_contribution=modality_contribution,
                    )
                    record["shap_weighted_overlay"] = str(weighted_png)
            manifest.append(entry)

    run_details = _run_details(args, device)
    run_details["task"] = task
    run_details["cxr_modalities"] = cxr_inputs
    manifest_path = root_out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"run_details": run_details, "entries": manifest}, indent=2)
    )
    print(
        f"Wrote {sum(len(entry['outputs']) for entry in manifest)} overlays + "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
