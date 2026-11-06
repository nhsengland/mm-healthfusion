"""Compare saved predictions from two runs using paired statistical tests.

Generate predictions first with python -m src.evaluation.generate_prediction.

Usage:
    python -m src.evaluation.compare_prediction --run-a <baseline> --run-b <candidate>
    python -m src.evaluation.compare_prediction --run-a <baseline> --path <runs-root>
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Save plots without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from src.evaluation.eval_utils import (
    DEFAULT_CHECKPOINT_NAME,
    discover_checkpoints,
    load_config,
    prediction_identifier_column,
    predictions_path,
    read_predictions,
)

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "model_comparison"

VALID_SPLITS = ("val", "test")
# Use the exact McNemar test when there are few discordant pairs
MCNEMAR_EXACT_MAX_DISCORDANT = 25
# Use a fixed seed for repeatable MCC permutation tests
N_PERMUTATIONS = 10_000
N_PERMUTATION_SEED = 0
# Control the false discovery rate across each batch of MCC tests
FDR_ALPHA = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two models' predictions with paired significance tests.",
    )
    parser.add_argument(
        "--run-a",
        required=True,
        help="Run directory of model A / the baseline (contains "
        "{split}_predictions.csv). Always a single fixed run, used in both "
        "single and batch mode.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-b",
        help="Run directory of model B (contains {split}_predictions.csv). "
        "Single-comparison mode, compared against --run-a.",
    )
    source.add_argument(
        "--path",
        default=None,
        help=(
            "Root directory to search recursively for candidate model-B runs. "
            "Every run holding checkpoint_best.ckpt is compared against "
            "--run-a (the baseline), writing outputs to each candidate's own "
            "directory. Batch mode, mutually exclusive with --run-b."
        ),
    )
    parser.add_argument(
        "--split",
        dest="splits",
        choices=VALID_SPLITS,
        nargs="+",
        default=list(VALID_SPLITS),
        help="Which split(s) to compare. Defaults to both val and test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to probabilities. Defaults to 0.5.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=(
            f"Resamples for the paired MCC permutation test (default: "
            f"{N_PERMUTATIONS}). Lower (e.g. 1000) for quick exploration. The "
            f"smallest reportable p-value is 1/(N+1)."
        ),
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=FDR_ALPHA,
        help=(
            f"Target false-discovery rate for the Benjamini-Hochberg correction "
            f"applied across a --path batch's MCC-permutation p-values (default: "
            f"{FDR_ALPHA}). Ignored in single --run-b mode (one test, nothing to "
            f"correct)."
        ),
    )
    parser.add_argument(
        "--name-a",
        default=None,
        help="Display name for model A. Defaults to the run-a directory name.",
    )
    parser.add_argument(
        "--name-b",
        default=None,
        help="Display name for model B. Defaults to the run-b directory name.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for comparison outputs (default: "
            f"{DEFAULT_OUTPUT_DIR}). Each comparison writes to "
            "<output-dir>/<task>/<outcome>/<name-a>_vs_<name-b>/."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In --path mode, skip candidates that already have "
            "{split}_prediction_comparison.txt for every requested split."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "In --path mode, keep going when a comparison fails and report the "
            "failures at the end instead of aborting the whole batch."
        ),
    )
    return parser.parse_args(argv)


def align_predictions(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    """Match prediction rows by stay ID and check that their labels agree."""
    identifier_column = prediction_identifier_column(frame_a)
    if prediction_identifier_column(frame_b) != identifier_column:
        raise ValueError("Prediction files use different identifier columns.")
    merged = frame_a.merge(
        frame_b,
        on=identifier_column,
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            f"No shared {identifier_column} between the two prediction files."
        )
    if not (merged["label_a"] == merged["label_b"]).all():
        raise ValueError(
            "Ground-truth labels differ between the two files for some stays. "
            "are they really the same split?"
        )
    return merged


def compare_predictions(
    merged: pd.DataFrame,
    *,
    threshold: float = 0.5,
    name_a: str = "model_a",
    name_b: str = "model_b",
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, Any]:
    """Calculate confusion matrices and statistical tests for aligned predictions.

    Treat Mann-Whitney results as reference only because paired predictions do
    not meet the test's independence assumption.
    """
    labels = merged["label_a"].to_numpy()
    probability_a = merged["probability_a"].to_numpy()
    probability_b = merged["probability_b"].to_numpy()
    prediction_a = (probability_a >= threshold).astype(int)
    prediction_b = (probability_b >= threshold).astype(int)

    correct_a = prediction_a == labels
    correct_b = prediction_b == labels
    # McNemar compares pairs where only one model is correct
    both_correct = int((correct_a & correct_b).sum())
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    neither = int((~correct_a & ~correct_b).sum())
    discordant = a_only + b_only
    mcnemar_result = mcnemar(
        [[both_correct, a_only], [b_only, neither]],
        exact=discordant < MCNEMAR_EXACT_MAX_DISCORDANT,
    )

    return {
        "n_stays": int(len(merged)),
        "threshold": float(threshold),
        "model_a": name_a,
        "model_b": name_b,
        "confusion_matrix_a": _confusion_dict(labels, prediction_a),
        "confusion_matrix_b": _confusion_dict(labels, prediction_b),
        "mcc_a": float(matthews_corrcoef(labels, prediction_a)),
        "mcc_b": float(matthews_corrcoef(labels, prediction_b)),
        "mcnemar": {
            "both_correct": both_correct,
            f"{name_a}_correct_only": a_only,
            f"{name_b}_correct_only": b_only,
            "neither_correct": neither,
            "n_discordant": discordant,
            "exact": bool(discordant < MCNEMAR_EXACT_MAX_DISCORDANT),
            "statistic": float(mcnemar_result.statistic),
            "p_value": float(mcnemar_result.pvalue),
        },
        "mcc_permutation": _mcc_permutation_test(
            labels,
            prediction_a,
            prediction_b,
            n_permutations=n_permutations,
        ),
        "wilcoxon_signed_rank": _paired_probability_test(
            probability_a,
            probability_b,
        ),
        "mann_whitney_u": _mann_whitney(probability_a, probability_b),
    }


def _confusion_dict(labels: Any, predictions: Any) -> dict[str, int]:
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def _paired_probability_test(probability_a: Any, probability_b: Any) -> dict[str, Any]:
    # Wilcoxon raises when every paired difference is zero (identical outputs)
    if (probability_a == probability_b).all():
        return {"statistic": "", "p_value": "", "note": "identical probabilities"}
    result = wilcoxon(probability_a, probability_b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _mann_whitney(probability_a: Any, probability_b: Any) -> dict[str, Any]:
    result = mannwhitneyu(probability_a, probability_b, alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "note": "assumes independent samples, but predictions here are paired",
    }


def _mcc_permutation_test(
    labels: Any,
    prediction_a: Any,
    prediction_b: Any,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = N_PERMUTATION_SEED,
) -> dict[str, Any]:
    """Test the MCC difference between model B and model A by swapping paired predictions.

    A positive difference means model B scores higher. Swap each stay's two
    predictions with probability 0.5 and count differences at least as large in
    absolute value. Add one to the count and sample total to avoid a zero p-value.
    Inputs must be aligned with one row per stay. Repeated rows from the same
    patient would require swaps grouped by patient.
    """
    labels = np.asarray(labels)
    prediction_a = np.asarray(prediction_a)
    prediction_b = np.asarray(prediction_b)
    observed = float(
        matthews_corrcoef(labels, prediction_b)
        - matthews_corrcoef(labels, prediction_a)
    )
    # Identical decisions give a zero MCC difference with nothing to permute
    if (prediction_a == prediction_b).all():
        return {
            "delta_mcc": observed,
            "statistic": observed,
            "p_value": "",
            "n_permutations": int(n_permutations),
            "note": "identical predictions",
        }

    rng = np.random.default_rng(seed)
    abs_observed = abs(observed)
    n_stays = len(labels)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        swap = rng.random(n_stays) < 0.5
        permuted_a = np.where(swap, prediction_b, prediction_a)
        permuted_b = np.where(swap, prediction_a, prediction_b)
        null_gap = matthews_corrcoef(labels, permuted_b) - matthews_corrcoef(
            labels, permuted_a
        )
        if abs(null_gap) >= abs_observed:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_permutations + 1)
    return {
        "delta_mcc": observed,
        "statistic": observed,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
    }


def benjamini_hochberg(
    pvalues: list[float],
    *,
    alpha: float = FDR_ALPHA,
) -> tuple[list[float], list[bool]]:
    """Adjust p-values for multiple comparisons and flag rejected hypotheses.

    Use the Benjamini-Hochberg method to control the expected proportion of false
    positives among rejected hypotheses at alpha. Preserve input order and return
    two empty lists for empty input.
    """
    if not pvalues:
        return [], []
    reject, p_adjusted, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    return [float(p) for p in p_adjusted], [bool(flag) for flag in reject]


def _write_bh_summary(
    dest_dir: Path,
    split: str,
    name_a: str,
    entries: list[tuple[str, dict[str, Any]]],
) -> tuple[Path, int] | None:
    """Write adjusted MCC test results, sorted by the original p-value.

    Entries contain a candidate name and test results with p_value_bh and reject_h0.
    Exclude tests with no p-value. Return the CSV path and number of rejected
    hypotheses, or None when there are no test results.
    """
    testable = [(name_b, perm) for name_b, perm in entries if isinstance(perm["p_value"], float)]
    if not testable:
        return None
    testable.sort(key=lambda entry: entry[1]["p_value"])
    frame = pd.DataFrame(
        {
            "model_a": [name_a] * len(testable),
            "model_b": [name_b for name_b, _ in testable],
            "delta_mcc": [perm["delta_mcc"] for _, perm in testable],
            "p_value": [perm["p_value"] for _, perm in testable],
            "p_value_bh": [perm["p_value_bh"] for _, perm in testable],
            "reject_h0": [perm["reject_h0"] for _, perm in testable],
        }
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dest_dir / f"mcc_permutation_bh_{split}.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path, int(sum(perm["reject_h0"] for _, perm in testable))


def compare_run_predictions(
    run_dir_a: str | Path,
    run_dir_b: str | Path,
    splits: tuple[str, ...] = VALID_SPLITS,
    *,
    threshold: float = 0.5,
    name_a: str = "model_a",
    name_b: str = "model_b",
    n_permutations: int = N_PERMUTATIONS,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare two run directories across the requested splits."""
    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        merged = align_predictions(
            read_predictions(predictions_path(run_dir_a, split)),
            read_predictions(predictions_path(run_dir_b, split)),
        )
        result = compare_predictions(
            merged,
            threshold=threshold,
            name_a=name_a,
            name_b=name_b,
            n_permutations=n_permutations,
        )
        results[split] = result
        if output_dir is not None:
            output_path = Path(output_dir) / f"{split}_prediction_comparison.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(_format_result(split, result), encoding="utf-8")
            result["output_path"] = str(output_path)
            png_path = output_path.with_name(f"{split}_confusion_matrices.png")
            _save_confusion_png(split, result, png_path)
            result["confusion_png_path"] = str(png_path)
    return results


def _save_confusion_png(split: str, result: dict[str, Any], path: Path) -> None:
    """Render both models' confusion matrices side by side to a PNG."""
    panels = (
        (result["model_a"], result["confusion_matrix_a"], result["mcc_a"]),
        (result["model_b"], result["confusion_matrix_b"], result["mcc_b"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, (name, matrix, mcc) in zip(axes, panels):
        grid = [
            [matrix["true_negative"], matrix["false_positive"]],
            [matrix["false_negative"], matrix["true_positive"]],
        ]
        axis.imshow(grid, cmap="Blues")
        vmax = max(max(row) for row in grid) or 1
        for i, row in enumerate(grid):
            for j, count in enumerate(row):
                axis.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > vmax / 2 else "black",
                )
        axis.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        axis.set_yticks([0, 1], labels=["true 0", "true 1"])
        axis.set_title(f"{name}\nMCC={mcc:.4f}")
    figure.suptitle(
        f"{split} confusion matrices "
        f"(n={result['n_stays']}, threshold={result['threshold']})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=600)
    plt.close(figure)


def _format_result(split: str, result: dict[str, Any]) -> str:
    lines = [
        f"=== {split} (n={result['n_stays']}, threshold={result['threshold']}) ==="
    ]
    for name_key, matrix_key, mcc_key in (
        ("model_a", "confusion_matrix_a", "mcc_a"),
        ("model_b", "confusion_matrix_b", "mcc_b"),
    ):
        matrix = result[matrix_key]
        lines.append(
            f"{result[name_key]}: mcc={result[mcc_key]:.4f} "
            f"TN={matrix['true_negative']} FP={matrix['false_positive']} "
            f"FN={matrix['false_negative']} TP={matrix['true_positive']}"
        )
    permutation = result["mcc_permutation"]
    if permutation["p_value"] == "":
        lines.append(
            f"MCC permutation: {permutation.get('note', 'not computed')} "
            f"(dMCC={permutation['delta_mcc']:+.4f})"
        )
    else:
        line = (
            f"MCC permutation (B={permutation['n_permutations']}): "
            f"dMCC={permutation['delta_mcc']:+.4f} p={permutation['p_value']:.4g}"
        )
        # Include adjusted p-values after all batch comparisons are complete
        if "p_value_bh" in permutation:
            verdict = "reject H0" if permutation["reject_h0"] else "retain H0"
            line += (
                f" | BH p={permutation['p_value_bh']:.4g} "
                f"(FDR={permutation.get('fdr_alpha', FDR_ALPHA):g}: {verdict})"
            )
        lines.append(line)
    mcnemar_result = result["mcnemar"]
    lines.append(
        f"McNemar: n_discordant={mcnemar_result['n_discordant']} "
        f"(exact={mcnemar_result['exact']}) "
        f"stat={mcnemar_result['statistic']:.4f} p={mcnemar_result['p_value']:.4e}"
    )
    lines.append(_format_test("Wilcoxon signed-rank", result["wilcoxon_signed_rank"]))
    lines.append(_format_test("Mann-Whitney U", result["mann_whitney_u"]))
    return "\n".join(lines) + "\n"


def _format_test(label: str, test: dict[str, Any]) -> str:
    if test["p_value"] == "":
        return f"{label}: {test.get('note', 'not computed')}"
    return f"{label}: stat={test['statistic']:.4f} p={test['p_value']:.4g}"


def _has_predictions_for(run_dir: str | Path, splits: tuple[str, ...]) -> bool:
    """True when the run has ``{split}_predictions.csv`` for every requested split."""
    run_dir = Path(run_dir)
    return all((run_dir / f"{split}_predictions.csv").exists() for split in splits)


def _run_scope(run_dir: str | Path) -> Path:
    """Return the task and outcome path used to group comparison outputs."""
    config_path = Path(run_dir) / "config.yaml"
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        data_cfg = {}

    outcome = str(data_cfg.get("outcome", cfg.get("outcome", ""))).strip().lower()
    if not outcome:
        raise ValueError(f"No outcome is configured in {config_path}.")

    task = str(data_cfg.get("task", data_cfg.get("dataset", ""))).strip().lower()
    if not task:
        task = outcome.split("_", maxsplit=1)[0]
    if task not in {"hospitalisation", "deterioration"}:
        raise ValueError(
            f"Unsupported task {task!r} in {config_path}. Expected hospitalisation "
            "or deterioration."
        )
    outcome = outcome.removeprefix(f"{task}_")
    if not outcome:
        raise ValueError(f"No outcome remains after the task prefix in {config_path}.")
    return Path(task) / outcome


def _comparison_output_dir(
    output_root: str | Path,
    run_dir_a: str | Path,
    run_dir_b: str | Path,
    name_a: str,
    name_b: str,
) -> Path:
    """Build an output path that distinguishes runs with the same folder name."""
    scope_a = _run_scope(run_dir_a)
    scope_b = _run_scope(run_dir_b)
    if scope_a != scope_b:
        raise ValueError(
            "Runs have different task/outcome configurations: "
            f"{scope_a!r} != {scope_b!r}."
        )
    return Path(output_root) / scope_a / f"{name_a}_vs_{name_b}"


def run_batch(
    root: str | Path,
    run_dir_a: str | Path,
    splits: tuple[str, ...],
    *,
    threshold: float = 0.5,
    name_a: str | None = None,
    n_permutations: int = N_PERMUTATIONS,
    fdr_alpha: float = FDR_ALPHA,
    output_root: str | Path | None = None,
    skip_existing: bool = False,
    continue_on_error: bool = False,
) -> None:
    """Compare a baseline with each checkpoint found under root.

    Write to each candidate's directory, or a task and outcome directory below
    output_root. Skip the baseline itself and runs without the requested predictions.
    """
    baseline = Path(run_dir_a)
    name_a = name_a or baseline.name
    checkpoints = discover_checkpoints(root)
    if not checkpoints:
        print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {root}.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {root}.")
    failures: list[tuple[Path, str]] = []
    # Keep results so p-values can be adjusted together after the batch finishes
    batch_results: list[tuple[str, dict[str, dict[str, Any]]]] = []
    compared = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        run_dir_b = checkpoint.parent
        if run_dir_b.resolve() == baseline.resolve():
            print(f"[{index}/{len(checkpoints)}] skip (baseline): {run_dir_b}")
            continue
        out_dir = (
            run_dir_b
            if output_root is None
            else _comparison_output_dir(
                output_root,
                baseline,
                run_dir_b,
                name_a,
                run_dir_b.name,
            )
        )
        if skip_existing and all(
            (out_dir / f"{split}_prediction_comparison.txt").exists()
            for split in splits
        ):
            print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir_b}")
            continue
        if not _has_predictions_for(run_dir_b, splits):
            print(f"[{index}/{len(checkpoints)}] skip (no predictions): {run_dir_b.name}")
            continue
        print(f"[{index}/{len(checkpoints)}] comparing: {name_a} vs {run_dir_b.name}")
        try:
            results = compare_run_predictions(
                baseline,
                run_dir_b,
                splits,
                threshold=threshold,
                name_a=name_a,
                name_b=run_dir_b.name,
                n_permutations=n_permutations,
                output_dir=out_dir,
            )
            compared += 1
            batch_results.append((run_dir_b.name, results))
            for split in splits:
                output_path = results[split].get("output_path")
                if output_path is not None:
                    print(f"    Saved at: {output_path}")
        except Exception as error:  # noqa: BLE001 - reported per-run below
            if not continue_on_error:
                raise
            failures.append((run_dir_b, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")

    print(f"Done. Compared {name_a} against {compared}/{len(checkpoints)} run(s).")

    # Adjust p-values together for each split and update the reports and summary
    summary_dir = Path(output_root) if output_root is not None else Path(root)
    for split in splits:
        entries = [(name_b, results[split]["mcc_permutation"]) for name_b, results in batch_results]
        testable = [perm for _, perm in entries if isinstance(perm["p_value"], float)]
        if not testable:
            continue
        p_adjusted, reject = benjamini_hochberg(
            [perm["p_value"] for perm in testable], alpha=fdr_alpha
        )
        for perm, p_bh, rejected in zip(testable, p_adjusted, reject):
            perm["p_value_bh"] = float(p_bh)
            perm["reject_h0"] = bool(rejected)
            perm["fdr_alpha"] = float(fdr_alpha)
        # Re-render each candidate's .txt now that its BH-corrected p is known
        for _, results in batch_results:
            result = results[split]
            output_path = result.get("output_path")
            if output_path is not None:
                Path(output_path).write_text(
                    _format_result(split, result), encoding="utf-8"
                )
        summary = _write_bh_summary(summary_dir, split, name_a, entries)
        if summary is None:
            continue
        csv_path, n_rejected = summary
        print(
            f"BH MCC summary [{split}]: {n_rejected}/{len(testable)} significant at "
            f"FDR={fdr_alpha:g} -> {csv_path}"
        )

    if failures:
        print(f"{len(failures)} run(s) failed:")
        for run_dir_b, message in failures:
            print(f"  {run_dir_b}: {message}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    splits = tuple(dict.fromkeys(args.splits))

    if args.path is not None:
        run_batch(
            args.path,
            args.run_a,
            splits,
            threshold=args.threshold,
            name_a=args.name_a,
            n_permutations=args.n_permutations,
            fdr_alpha=args.fdr_alpha,
            output_root=args.output_dir,
            skip_existing=args.skip_existing,
            continue_on_error=args.continue_on_error,
        )
        return

    name_a = args.name_a or Path(args.run_a).name
    name_b = args.name_b or Path(args.run_b).name
    output_dir = (
        None
        if args.output_dir is None
        else _comparison_output_dir(
            args.output_dir,
            args.run_a,
            args.run_b,
            name_a,
            name_b,
        )
    )
    results = compare_run_predictions(
        args.run_a,
        args.run_b,
        splits,
        threshold=args.threshold,
        name_a=name_a,
        name_b=name_b,
        n_permutations=args.n_permutations,
        output_dir=output_dir,
    )
    for split in splits:
        print()
        print(_format_result(split, results[split]))
        output_path = results[split].get("output_path")
        if output_path is not None:
            print(f"Saved at: {output_path}")
        png_path = results[split].get("confusion_png_path")
        if png_path is not None:
            print(f"Confusion matrices: {png_path}")


if __name__ == "__main__":
    main()
