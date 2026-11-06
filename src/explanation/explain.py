"""Explain how model inputs affect predictions for a saved run.

Write input contribution results, feature explanations and plots for
deterioration or hospitalisation models.

Run from the project root:

    python -m src.explanation.explain --checkpoint /path/to/checkpoint_best.ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.collate import get_collate_fn
from src.data.splits import find_split_summary, get_datasets, resolve_split_dir
from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    _usable_device,
    discover_checkpoints,
    load_config,
)
from src.explanation import shap_utils
from src.explanation.freeze_m3care import use_frozen_graph
from src.explanation.shap_utils import DEFAULT_SHAP_SEED
from src.training.lightning import MMModel

EXPLAIN_SUBDIR = "explanation"

# Use the last output in each mode to recognise a completed run
TABULAR_COMPLETION_MARKER = "shap_global_tabular_importance.csv"
MM_COMPLETION_MARKER = "mm_shap_modality_contribution.json"
SUMMARY_PLOT_FILENAMES = {
    "heatmap": "shap_global_tabular_summary_heatmap.png",
    "beeswarm": "shap_global_tabular_summary_beeswarm.png",
}


def _global_completion_marker(summary_plot: str) -> str:
    """Return the last summary image written for the chosen plot style."""
    # When both plots are requested, the beeswarm is written last
    return SUMMARY_PLOT_FILENAMES["beeswarm" if summary_plot == "both" else summary_plot]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read options for one checkpoint or a folder of runs."""
    parser = argparse.ArgumentParser(
        description="Explain one saved model or every run in a folder."
    )
    # Choose runs and patients
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default=None, help="Path to one saved model.")
    source.add_argument("--path", default=None, help="Folder to search for saved models.")
    parser.add_argument("--stays", type=int, nargs="+", default=None,
                        help="Stay IDs for patient plots. Defaults to the first ten "
                             "test patients in saved order.")
    parser.add_argument("--n-explain", type=int, default=-1,
                        help="Number of test patients to explain by feature SHAP, in saved "
                             "order (-1 means all).")
    parser.add_argument("--n-mmshap", type=int, default=-1,
                        help="Number of test patients used for modality contributions, in saved "
                             "order (-1 means all).")

    # Choose explanations and samples
    parser.add_argument("--exp-mode", choices=("mmshap", "global", "local", "all"), default="all",
                        help="What to write: mmshap for modality contributions, global for "
                             "contributions and feature summary, local for feature summary "
                             "and patient plots, or all for everything (default).")
    parser.add_argument("--explainer", default="gradient", choices=("kernel", "gradient"),
                        help="Method for feature SHAP values: gradient (faster) or kernel "
                             "(slower). This does not affect modality contributions.")
    parser.add_argument("--n-background", type=int, default=100,
                        help="Number of training patients used as the SHAP reference.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SHAP_SEED,
                        help="Random seed for the training sample and SHAP calculations "
                             f"(default: {DEFAULT_SHAP_SEED}).")

    # Options for outputs and plots
    parser.add_argument("--output-dir", default=None,
                        help="Folder for results (default: beside the saved model).")
    parser.add_argument("--summary-plot", choices=("heatmap", "beeswarm", "both"),
                        default="heatmap",
                        help="Feature summary plot: heatmap (default), beeswarm or both.")
    parser.add_argument("--max-display", dest="max_display",
                        type=int, default=21,
                        help="Maximum number of features in summary and patient plots.")
    parser.add_argument("--fusion-method", default=None,
                        help="Override the recorded name for how the model combines inputs.")

    # Control folder runs (when --path is used instead of --checkpoint)
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="When searching a folder, skip runs with the requested result "
                             "already complete.")
    parser.add_argument("--continue-on-error", action="store_true", default=False,
                        help="Keep processing other runs if one fails.")
    
    # Options for how to combine features
    parser.add_argument("--no-merge-onehot", dest="merge_onehot", action="store_false",
                        default=True,
                        help="Keep binary category columns separate in results and plots.")
    parser.add_argument("--merge-acuity", dest="merge_multiclass", action="store_true",
                        default=False,
                        help="Combine the acuity categories into one feature.")

    return parser.parse_args(argv)


def _make_batch(dataset, indices, collate_fn):
    """Prepare the selected patient rows as a model batch."""
    return collate_fn([dataset[int(i)] for i in indices])


def _model_modalities(model: MMModel, dataset) -> tuple[str, ...]:
    """Find the model's input names, falling back to the dataset names."""
    modalities = getattr(model.model, "modality_names", None)
    if modalities:
        return tuple(str(modality) for modality in modalities)
    dataset_modalities = getattr(dataset, "modality_names", None)
    if dataset_modalities:
        return tuple(str(modality) for modality in dataset_modalities)
    return shap_utils.MODALITIES


def _cxr_missing_by_stay(
    dataset, ed_ids: Sequence[int], modalities: Sequence[str],
) -> np.ndarray | None:
    """Mark deterioration patients with no chest X-ray input.

    Return None when the model has no cxr input or the split lacks its flags.
    """
    frame = dataset.frame.set_index("ed_stay_id")
    if "cxr" not in modalities or "cxr_missing" not in frame:
        return None
    return frame["cxr_missing"].reindex(ed_ids).to_numpy(dtype=int)


def _run_details(**settings: Any) -> dict:
    """Record the saved revision and settings behind these explanations."""
    return {**shap_utils.git_revision(), **settings}


def _to_jsonable(obj: Any) -> Any:
    """Convert result values so they can be written as JSON."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def explain_checkpoint(
    checkpoint: str | Path,
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    explainer: str = "gradient",
    exp_mode: str = "all",
    n_explain: int = -1,
    n_background: int = 100,
    n_mmshap: int = -1,
    stays: Sequence[int] | None = None,
    max_display: int = 21,
    summary_plot: str = "heatmap",
    merge_onehot: bool = True,
    merge_multiclass: bool = False,
    seed: int = DEFAULT_SHAP_SEED,
    fusion_method: str | None = None,
    graph_chunk_size: int = 2048,
) -> dict[str, Path]:
    """Explain one saved model and return the paths of its output files."""
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    config_path = run_dir / "config.yaml" if config_path is None else Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = dict(load_config(config_path))
    cfg.setdefault("data", {})
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["split_dir"] = cfg["data"].get("split_dir")
    cfg["data"]["num_workers"] = 0  # Read patient data in this process
    split_dir = resolve_split_dir(cfg)

    model_cfg = cfg.get("model", {})
    if fusion_method is None:
        fusion_method = str(model_cfg.get("fusion_method", model_cfg.get("name", "concat"))).lower()
    outcome = str(cfg.get("data", {}).get("outcome", cfg.get("outcome", "deterioration")))

    map_location = _usable_device()
    model = MMModel.load_from_checkpoint(str(checkpoint), map_location=map_location)
    model.eval()

    # Keep the other patients fixed when M3Care fills in missing inputs
    # Other models can score each patient separately
    use_frozen = use_frozen_graph(model)
    batch_size = int(cfg.get("data", {}).get("batch_size", 32))

    datasets = get_datasets(
        cfg, splits=("test",) if exp_mode == "mmshap" else ("train", "test")
    )
    test_ds = datasets["test"]
    identifier_column = test_ds.identifier_column
    if stays is None:
        stays = tuple(
            int(x)
            for x in test_ds.frame[test_ds.source_identifier_column]
            .head(10)
            .to_numpy()
        )
    collate_fn = get_collate_fn(cfg.get("model", {}).get("collate_style") or model_cfg.get("name"))
    tabular_is_sequential = bool(test_ds.tabular_is_sequential)
    active_modalities = _model_modalities(model, test_ds)

    out_dir = (run_dir if output_dir is None else Path(output_dir)) / EXPLAIN_SUBDIR

    rng = np.random.default_rng(seed)
    outputs: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    # Measure how each input type contributes across patients
    if exp_mode in ("mmshap", "global", "all"):
        # Use patients in saved order, or all patients when no limit is set
        if use_frozen:
            # Keep each patient's original neighbours while measuring modality contributions
            mm, ed_ids = shap_utils.compute_mm_shap_frozen_graph(
                model, test_ds, collate_fn, batch_size=batch_size, n_mmshap=n_mmshap,
                modalities=active_modalities, tabular_is_sequential=tabular_is_sequential,
            )
            n_mm = len(ed_ids)
        else:
            n_mm_sel = len(test_ds) if (n_mmshap is None or n_mmshap < 0) else min(n_mmshap, len(test_ds))
            mm_idx = np.arange(n_mm_sel)
            n_mm = len(mm_idx)
            mm_batch = _make_batch(test_ds, mm_idx, collate_fn)
            mm = shap_utils.compute_mm_shap(model, mm_batch, modalities=active_modalities)
            ed_ids = mm_batch[identifier_column].detach().cpu().numpy().reshape(-1)

        mm_pct = {m: mm["global_contribution"][m] * 100.0 for m in mm["modalities"]}
        mm_summary_path = out_dir / "mm_shap_modality_contribution.json"
        mm_payload = {
            "fusion_method": fusion_method,
            "outcome": outcome,
            "n_samples": int(n_mm),
            "global_contribution_pct": mm_pct,
            "mean_abs_shapley": mm["mean_abs_shapley"],
            "efficiency_residual": mm["efficiency_residual"],
            "v_all_mean_probability": mm["v_all_mean"],
            "v_none_mean_probability": mm["v_none_mean"],
            "note": (
                "MM-SHAP (Parcalabescu & Frank, ACL 2023): normalized absolute Shapley mass per "
                f"modality, exact {len(active_modalities)}-modality coalition game, "
                "performance-agnostic (value = predicted probability). Complementary to "
                "modality_ablation.py / modality_permutation.py."
                + (" M3Care scored against the cached real-batch graph, so cross-patient "
                   "imputation stays active." if use_frozen else "")
            ),
        }
        mm_summary_path.write_text(json.dumps(_to_jsonable(mm_payload), indent=2))
        outputs["mm_shap_json"] = mm_summary_path

        # Save run details before feature explanations begin
        manifest_path = out_dir / "shap_manifest.json"
        manifest_path.write_text(json.dumps(_to_jsonable(_run_details(
            fusion_method=fusion_method,
            outcome=outcome,
            explainer=explainer,
            graph_chunk_size=(graph_chunk_size if use_frozen else None),
            modalities=list(active_modalities),
            batch_size=batch_size,
            n_explain=n_explain,
            n_background=n_background,
            n_mmshap=n_mmshap,
            stays=([int(x) for x in stays] if stays else None),
            seed=seed,
            merge_onehot=merge_onehot,
            merge_multiclass=merge_multiclass,
            split_dir=str(split_dir) if split_dir else None,
        )), indent=2))
        outputs["shap_manifest_json"] = manifest_path

        mm_csv = out_dir / "mm_shap_per_sample.csv"
        per_sample = mm["per_sample_contribution"]  # One row per patient and input type
        with mm_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [identifier_column]
                + [f"{m}_contribution" for m in mm["modalities"]]
            )
            for i in range(per_sample.shape[0]):
                writer.writerow([int(ed_ids[i])] + [f"{per_sample[i, j]:.6f}" for j in range(per_sample.shape[1])])
        outputs["mm_shap_csv"] = mm_csv

        mm_bar = out_dir / "mm_shap_modality_contribution.png"
        try:
            if outcome.startswith("hospitalisation_"):
                shap_utils.plot_global_modality_contribution(
                    mm, mm_bar, fusion_type=fusion_method,
                    palette=["#6D2E8E", "#0E7C84", "#BE7C1C", "#3B4C61"],
                )
            else:
                # Group deterioration results by chest X-ray availability
                cxr_missing = _cxr_missing_by_stay(test_ds, ed_ids, active_modalities)
                if cxr_missing is None:
                    warnings.warn(
                        "No CXR modality flags found, so mm_shap_modality_contribution.png was not written.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    raise ValueError("CXR availability flags are unavailable")
                shap_utils.plot_modality_contribution(
                    mm, mm_bar, cxr_missing, fusion_type=fusion_method,
                    palette=["#6D2E8E", "#0E7C84", "#BE7C1C", "#3B4C61"],
                )
        except ValueError as error:
            warnings.warn(
                f"Could not plot MM-SHAP modality-contribution bars: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            outputs["mm_shap_plot"] = mm_bar

    if exp_mode == "mmshap":
        return outputs

    # Explain individual patient features
    train_ds = datasets["train"]
    summary_json = find_split_summary(split_dir) if split_dir else None
    if summary_json is None:
        raise FileNotFoundError(f"No *_split_summary.json under split_dir={split_dir}")
    feature_names = shap_utils.get_feature_names(summary_json, expected_width=test_ds.tabular_width)
    if "shap_manifest_json" not in outputs:
        manifest_path = out_dir / "shap_manifest.json"
        manifest_path.write_text(json.dumps(_to_jsonable(_run_details(
            fusion_method=fusion_method,
            outcome=outcome,
            explainer=explainer,
            graph_chunk_size=None,
            modalities=list(active_modalities),
            batch_size=batch_size,
            n_explain=n_explain,
            n_background=n_background,
            n_mmshap=None,
            stays=([int(x) for x in stays] if stays else None),
            seed=seed,
            merge_onehot=merge_onehot,
            merge_multiclass=merge_multiclass,
            split_dir=str(split_dir) if split_dir else None,
        )), indent=2))
        outputs["shap_manifest_json"] = manifest_path
    # Explain test patients in saved order and sample the training reference separately
    n_bg = min(n_background, len(train_ds))
    bg_idx = rng.choice(len(train_ds), size=n_bg, replace=False)
    bg_batch = _make_batch(train_ds, bg_idx, collate_fn)

    if use_frozen:
        # Keep each patient's original neighbours while explaining features
        explanation, exp_ed_ids = shap_utils.get_shap_values_frozen_graph(
            model, test_ds, collate_fn, bg_batch, feature_names,
            batch_size=batch_size, tabular_is_sequential=tabular_is_sequential,
            engine=explainer, kernel_nsamples="auto", gradient_nsamples="auto",
            chunk_size=graph_chunk_size, n_explain=n_explain, shap_seed=seed,
        )
    else:
        n_exp_sel = len(test_ds) if (n_explain is None or n_explain < 0) else min(n_explain, len(test_ds))
        exp_idx = np.arange(n_exp_sel)
        exp_batch = _make_batch(test_ds, exp_idx, collate_fn)
        exp_ed_ids = exp_batch[identifier_column].detach().cpu().numpy().reshape(-1)

        explanation = shap_utils.get_shap_values(
            model, exp_batch, bg_batch, feature_names=feature_names,
            tabular_is_sequential=tabular_is_sequential,
            engine=explainer,
            kernel_nsamples="auto",
            gradient_nsamples="auto",
            shap_seed=seed,
        )

    # Combine category columns when requested so one variable has one contribution
    if merge_onehot or merge_multiclass:
        explanation = shap_utils.merge_onehot_shap(
            explanation, merge_binary=merge_onehot, merge_multiclass=merge_multiclass,
        )
    plot_feature_names = list(explanation.feature_names)

    # Use the same feature names and ranking in the CSV and plots
    csv_feature_names = [shap_utils._display_feature_name(n) for n in plot_feature_names]
    mean_abs = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    feat_csv = out_dir / TABULAR_COMPLETION_MARKER
    with feat_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "mean_abs_shap", "mean_shap"])
        mean_signed = explanation.values.mean(axis=0)
        for j in order:
            writer.writerow([csv_feature_names[j], f"{mean_abs[j]:.6e}", f"{mean_signed[j]:.6e}"])
    outputs["feature_importance_csv"] = feat_csv

    outcome_disp = outcome.replace("deterioration_", "").replace("_", " ").title()

    if exp_mode in ("global", "local", "all"):
        # Write the beeswarm last when both summary plots are requested
        styles = ("heatmap", "beeswarm") if summary_plot == "both" else (summary_plot,)
        for style in styles:
            summary_png = out_dir / SUMMARY_PLOT_FILENAMES[style]
            shap_utils.get_shap_summary_plot(
                explanation, outcome=outcome_disp, fusion_type=fusion_method,
                modality="tabular", max_features=max_display,
                save_path=summary_png, heatmap=(style == "heatmap"),
            )
            outputs[f"feature_summary_{style}"] = summary_png

    # Plot requested patients that were explained and warn about the rest
    exp_id_to_row = {int(s): i for i, s in enumerate(np.asarray(exp_ed_ids).reshape(-1))}
    requested = [int(s) for s in (stays or ())]
    unknown = [s for s in requested if s not in exp_id_to_row]
    plot_rows = [exp_id_to_row[s] for s in requested if s in exp_id_to_row]
    if unknown and exp_mode in ("local", "all"):
        kept = (
            f"Continuing with the {len(plot_rows)} that are: "
            f"{' '.join(str(s) for s in requested if s in exp_id_to_row)}."
            if plot_rows else
            "None of the requested stays are available, so no decision plots will be written."
        )
        warnings.warn(
            f"--stays {unknown} not in the explained set. "
            f"{len(exp_id_to_row)} of {len(test_ds)} test stays were explained "
            f"(n_explain={n_explain}), taken in dataset order; raise --n-explain "
            f"(-1 explains the whole split) or name stays from within it. {kept}",
            RuntimeWarning,
            stacklevel=2,
        )

    if exp_mode in ("local", "all") and plot_rows:
        n_plots = len(plot_rows)
        local_ids = np.asarray([exp_ed_ids[r] for r in plot_rows]).reshape(-1)
        local_mods = local_contrib = local_prob = None
        if exp_mode == "all":
            # Reuse the modality contributions already calculated for these patients
            local_mods = mm["modalities"]
            if use_frozen:
                id_to_row = {int(sid): i for i, sid in enumerate(np.asarray(ed_ids).reshape(-1))}
                local_contrib = np.stack([
                    mm["per_sample_contribution"][id_to_row[int(sid)]]
                    if int(sid) in id_to_row else mm["per_sample_contribution"][k]
                    for k, sid in enumerate(local_ids)
                ], axis=0)
                local_prob = np.asarray([
                    mm["v_all"][id_to_row[int(sid)]] if int(sid) in id_to_row else np.nan
                    for sid in local_ids
                ], dtype=float)
            else:
                local_batch = _make_batch(test_ds, exp_idx[plot_rows], collate_fn)
                local_ids = (
                    local_batch[identifier_column].detach().cpu().numpy().reshape(-1)
                )
                local_mm = shap_utils.compute_mm_shap(
                    model, local_batch, modalities=active_modalities, show_progress=False,
                )
                local_contrib = local_mm["per_sample_contribution"]
                local_prob = np.asarray(local_mm["v_all"], dtype=float)
        # Show original measurements when the source file is available
        originals = shap_utils.load_original_tabular_values(
            summary_json, plot_feature_names, [int(s) for s in local_ids],
            identifier_column=identifier_column,
        )
        # Mark filled-in values instead of showing them as observed measurements
        stay_row_index = {
            int(s): i
            for i, s in enumerate(
                test_ds.frame[test_ds.source_identifier_column].to_numpy()
            )
        }
        for k in range(n_plots):
            stay_id = int(local_ids[k])
            local_png = out_dir / f"shap_global_tabular_decision_stay{stay_id}.png"
            mm_scores_k = None
            prob_k = None
            if exp_mode == "all":
                mm_scores_k = {
                    m: round(float(local_contrib[k, j]) * 100.0, 2)
                    for j, m in enumerate(local_mods)
                }
                prob_k = float(local_prob[k]) if np.isfinite(local_prob[k]) else None
            imputed_k: dict[str, bool] = {}
            label_k: float | None = None
            row = stay_row_index.get(stay_id)
            if row is not None:
                sample = test_ds[row]
                mask = np.asarray(sample["tab_missing"]).reshape(-1)
                if mask.size == len(feature_names):  # Apply this mask only when widths match
                    imputed_k = {n: bool(mask[j]) for j, n in enumerate(feature_names)}
                label_k = float(np.asarray(sample["label"]).reshape(-1)[0])
            shap_utils.get_shap_local_decision_plot(
                explanation[plot_rows[k]], save_tab_path=local_png,
                mm_scores=mm_scores_k,
                stay_id=stay_id, label=label_k, probability=prob_k,
                original_values=originals.get(stay_id),
                imputed=imputed_k or None,
            )
            outputs[f"feature_local_stay{stay_id}"] = local_png

    return outputs


def main(argv: list[str] | None = None) -> None:
    """Explain one saved run or every run found in a folder."""
    args = parse_args(argv)

    def _run_one(checkpoint: Path, output_dir: Path | None) -> dict[str, Path]:
        """Explain one checkpoint with the selected options."""
        return explain_checkpoint(
            checkpoint,
            output_dir=output_dir,
            explainer=args.explainer,
            exp_mode=args.exp_mode,
            n_explain=args.n_explain,
            n_background=args.n_background,
            n_mmshap=args.n_mmshap,
            stays=args.stays,
            max_display=args.max_display,
            summary_plot=args.summary_plot,
            merge_onehot=args.merge_onehot,
            merge_multiclass=args.merge_multiclass,
            seed=args.seed,
            fusion_method=args.fusion_method,
        )

    if args.checkpoint is not None:
        paths = _run_one(Path(args.checkpoint),
                         Path(args.output_dir) if args.output_dir else None)
        out_dir = (Path(args.checkpoint).parent if args.output_dir is None
                   else Path(args.output_dir)) / EXPLAIN_SUBDIR
        print(f"Explanation outputs stored at: {out_dir}")
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
        completion_marker = {
            "mmshap": MM_COMPLETION_MARKER,
            "global": _global_completion_marker(args.summary_plot),
        }.get(args.exp_mode, TABULAR_COMPLETION_MARKER)
        existing = run_dir / EXPLAIN_SUBDIR / completion_marker
        if args.skip_existing and existing.exists():
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
            continue
        print(f"[{index}/{len(checkpoints)}] explaining: {run_dir}")
        try:
            _run_one(checkpoint, None)
            evaluated += 1
        except Exception as error:  # noqa: BLE001 - Report each failed run
            if not args.continue_on_error:
                raise
            failures.append((run_dir, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(f"Done. Explained {evaluated}/{len(checkpoints)} run(s).")
    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir, message in failures:
            print(f"  {run_dir}: {message}")


if __name__ == "__main__":
    main()
