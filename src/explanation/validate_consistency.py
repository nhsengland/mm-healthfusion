"""Check that chest X-ray heatmaps rebuild the features the model was trained on.

Grad-CAM recomputes image features from the pixels, so a heatmap is only
faithful if that path reproduces the saved pipeline. For each chest X-ray input
of a run, three checks run over up to n patients whose images are on disk:

* A, raw features: image -> DenseNet.features -> relu -> average pooling,
  averaged over the patient's images, equals the patient's row in the raw
  feature pickle.
* B, scaling: min-max scaling of A, with statistics fitted as explain_cxr fits
  them, equals the features the model reads from the split.
* C, prediction: the rebuilt image -> fusion -> probability forward equals the
  saved prediction for the patient.

Check C scores each patient as explain_cxr does. Models that score patients
separately use the patient's own row from the split with all its inputs.
M3Care with imputation links patients through a (B, B) similarity graph, so its
saved probability only reproduces against the patient's real batch. The check
then rebuilds that batch and scores through the fixed-graph score function.

Run from the project root:

    python -m src.explanation.validate_consistency --path /path/to/run

Only --path is required. --cxr-root, --raw-pkl and --keys-csv are optional
overrides: the image folder defaults to the MIMIC-CXR-JPG root, and the raw
feature pickle and keys CSV are found from the run configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.eval_utils import (
    _usable_device,
    load_config,
    predictions_path,
    prediction_identifier_column,
    read_predictions,
    resolve_run,
)
from src.explanation.freeze_m3care import use_frozen_graph
from src.explanation.gradcam import (
    DEFAULT_CXR_ROOT,
    CXRGradCAM,
    apply_cxr_norm,
    cxr_available,
    cxr_modalities,
    gradcam_config,
    train_identifiers,
)
from src.explanation.shap_gradcam import patient_cxr_context
from src.explanation.shap_utils import SplitBatches


def parse_args(argv=None):
    """Read command-line options for a run and its checks."""
    parser = argparse.ArgumentParser(
        description="Check that chest X-ray heatmaps rebuild a saved run's features."
    )
    # Choose the run and patients
    parser.add_argument("--path", required=True,
                        help="Run folder containing checkpoint_best.ckpt and config.yaml.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"),
                        help="Patient split to check (default: test).")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of patients with images on disk to check for each "
                             "chest X-ray input, in saved order (default: 10).")

    # Locate image files and saved features
    parser.add_argument("--cxr-root", default=DEFAULT_CXR_ROOT,
                        help=f"Chest X-ray image folder (default: {DEFAULT_CXR_ROOT}).")
    parser.add_argument("--raw-pkl", default=None,
                        help="Raw saved image features. By default, find them from the run "
                             "configuration.")
    parser.add_argument("--keys-csv", default=None,
                        help="CSV linking patients to images. By default, find it from the "
                             "run configuration.")

    # Set how the checks run
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Largest absolute difference that passes a check (default: 1e-3).")

    return parser.parse_args(argv)


@dataclass
class CheckResult:
    """Summarise one check over the patients it compared."""

    name: str
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    n: int
    detail: str = ""


def _summarise(
    name: str,
    diffs: list,
    tol: float,
    per_feature: bool = True,
    detail: str = "",
) -> CheckResult:
    """Collapse one check's per-patient differences into a pass or fail."""
    if not diffs:
        return CheckResult(name, False, float("nan"), float("nan"), 0,
                           ", ".join(x for x in ("no samples", detail) if x))
    stacked = (
        np.concatenate([d.reshape(-1) for d in diffs]) if per_feature
        else np.asarray(diffs)
    )
    mx, mn = float(stacked.max()), float(stacked.mean())
    return CheckResult(name, mx < tol, mx, mn, len(diffs), detail)


def run_checks(
    g: CXRGradCAM,
    batches: SplitBatches,
    predictions: pd.DataFrame | None,
    n: int = 10,
    tol: float = 1e-3,
) -> list[CheckResult]:
    """Run checks A, B and C for g's chest X-ray input over up to n patients.

    Patients come from batches in split order and must have the input available
    and every image on disk. predictions holds the split's saved predictions, or
    None to skip check C.
    """
    cxr_modality = g.cxr_modality
    feats_key = f"{cxr_modality}_feats"
    source_column = batches.source_identifier_column
    source_ids = batches.dataset.frame[source_column].to_numpy()

    raw = pd.read_pickle(g.raw_cxr_pkl)
    # Take each patient's first key row, as resolve_stay_images does
    key_ids = g.keys[source_column]
    key_rows = {
        int(value): row
        for row, value in reversed(list(enumerate(key_ids.to_numpy())))
        if pd.notna(value)
    }

    # Join on the predictions' own key: hospitalisation runs before the hadm_id
    # change saved ed_stay_id, which the hospitalisation split also carries
    stored_probs: dict[int, float] = {}
    id_column = batches.identifier_column
    if predictions is not None:
        id_column = prediction_identifier_column(predictions)
        stored_probs = dict(
            zip(predictions[id_column].astype("int64"), predictions["probability"].astype(float))
        )

    use_frozen = use_frozen_graph(g.mm)
    a_diffs, b_diffs, c_diffs = [], [], []
    multi_timestep = 0

    checked = 0
    for idx, source_id in enumerate(source_ids):
        if checked >= n:
            break
        source_id = int(source_id)
        try:
            paths = g.resolve_stay_images(source_id)
        except KeyError:
            continue
        if not paths or not all(Path(p).exists() for p in paths):
            continue
        if not cxr_available(batches, idx, cxr_modality):
            continue

        sample = batches.dataset[idx]
        stored_scaled = np.atleast_2d(sample[feats_key].numpy())
        if stored_scaled.shape[0] != 1:
            # Grad-CAM averages all images into one row, so it cannot rebuild
            # a patient whose input keeps several timesteps
            multi_timestep += 1
            continue
        raw_row = np.atleast_2d(np.asarray(raw[key_rows[source_id]], dtype=np.float32))

        # The saved patient feature is scale(mean_i(raw_i)): images are averaged
        # before scaling, so rebuild the raw mean first
        with torch.no_grad():
            raws = torch.stack([g.raw_gap_from_image(p) for p in paths], dim=0)
            mean_raw = raws.mean(dim=0)   # (1024,)

        # A: mean raw features vs the raw pickle row
        a_diffs.append(np.abs(mean_raw.cpu().numpy() - raw_row[0]))

        # B: scaled mean vs the features the model reads, through apply_cxr_norm
        # so the check exercises the transform the heatmaps apply
        stay_emb = mean_raw
        if g.min_values is not None:
            stay_emb = apply_cxr_norm(mean_raw, g.min_values, g.max_values)
            b_diffs.append(np.abs(stay_emb.cpu().numpy() - stored_scaled[0]))

        # C: prediction from the rebuilt forward vs the saved prediction
        if predictions is not None and id_column not in sample:
            raise ValueError(
                f"Predictions are keyed by {id_column}, which the {cxr_modality} split "
                "does not carry."
            )
        identifier = int(sample[id_column]) if predictions is not None else None
        if identifier in stored_probs:
            patient = patient_cxr_context(g, batches, idx, stay_emb, use_frozen)
            cams = g.heatmap_for_stay(source_id, logit_fn=patient.logit_fn)
            c_diffs.append(abs(cams[0].prob - stored_probs[identifier]))
        checked += 1

    def _summ(name, diffs, per_feature=True, detail=""):
        return _summarise(name, diffs, tol, per_feature=per_feature, detail=detail)

    skipped = f"{multi_timestep} multi-timestep skipped" if multi_timestep else ""
    c_detail = ", ".join(x for x in ("frozen graph" if use_frozen else "", skipped) if x)
    if predictions is None:
        c_detail = "no predictions"

    return [
        _summ("A: raw features == pkl", a_diffs, detail=skipped),
        _summ(f"B: min-max(raw) == split {feats_key}", b_diffs, detail=skipped),
        _summ("C: prediction fidelity vs stored", c_diffs, per_feature=False,
              detail=c_detail),
    ]


def main(argv=None) -> None:
    """Run the checks for every chest X-ray input of a run."""
    args = parse_args(argv)
    checkpoint_path, config_path = resolve_run(args.path)
    device = _usable_device()
    task = str((load_config(config_path).get("data") or {}).get("task", "deterioration")).lower()

    batches = SplitBatches(config_path, args.split)
    cxr_inputs = cxr_modalities(batches)
    if not cxr_inputs:
        raise SystemExit("This run has no configured CXR input to check.")
    if args.raw_pkl and len(cxr_inputs) > 1:
        raise SystemExit(
            "--raw-pkl can only override a run with one configured CXR input. "
            "For multiple CXR inputs, use the files derived from the run config."
        )
    train_ids = train_identifiers(config_path, task)

    # Check C compares against the run's saved predictions and is skipped without them
    predictions = None
    pred_path = predictions_path(args.path, args.split)
    if pred_path.exists():
        predictions = read_predictions(pred_path)

    all_pass = True
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
        g = CXRGradCAM(cfg)

        if use_frozen_graph(g.mm):
            print(f"check C: frozen m3care graph (batch_size={batches.batch_size})")
        results = run_checks(g, batches, predictions, n=args.n, tol=args.tolerance)
        print(f"\nSelf-consistency checks for {cxr_modality} "
              f"(split={args.split}, n={args.n}, tolerance={args.tolerance}):")
        for r in results:
            status = "PASS" if r.passed else ("SKIP" if r.n == 0 else "FAIL")
            all_pass = all_pass and (r.passed or r.n == 0)
            print(f"  [{status}] {r.name}: max|d|={r.max_abs_diff:.3e} "
                  f"mean|d|={r.mean_abs_diff:.3e} (n={r.n}) {r.detail}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")


if __name__ == "__main__":
    main()
