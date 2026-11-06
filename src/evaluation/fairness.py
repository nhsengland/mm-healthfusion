"""Compare group fairness using saved predictions and bootstrap confidence intervals.

Generate predictions first with python -m src.evaluation.generate_prediction.

Usage:
    python -m src.evaluation.fairness --checkpoint <run>/checkpoint_best.ckpt
    python -m src.evaluation.fairness --path <runs-root>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_ratio,
    equal_opportunity_ratio,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)

from src.data import splits
from src.evaluation.eval_utils import (
    bias_corrected_ci,
    DEFAULT_CHECKPOINT_NAME,
    discover_checkpoints,
    load_config,
    prediction_identifier_column,
    read_predictions,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

PLOT_STYLE = {"font.size": 16, "font.weight": "normal"}

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
STATIC_PATHS = {
    "deterioration": OUTPUTS_DIR / "mds_ed" / "mds_ed_base.csv",
    "hospitalisation": {
        history: OUTPUTS_DIR / "hospitalisation" / "3.1" / "extract" / history / "static.csv"
        for history in ("requiredHistory", "optionalHistory")
    },
}
DEFAULT_OUTPUT_SUBDIR = "fairness"

PALETTE = ["#6D2E8E", "#0E7C84", "#BE7C1C", "#3B4C61"]
PALETTE_CMAP = matplotlib.colors.ListedColormap(PALETTE)

# Columns and labels in plot order
DEMOGRAPHIC_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("sex", "Sex"),
    ("age_group", "Age"),
    ("race_group", "Ethnicity"),
)
HOSPITALISATION_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    *DEMOGRAPHIC_ATTRIBUTES,
    ("insurance", "Insurance"),
)
CXR_ATTRIBUTE = ("cxr_availability", "CXR Availability")
ATTRIBUTES: tuple[tuple[str, str], ...] = (
    *DEMOGRAPHIC_ATTRIBUTES,
    CXR_ATTRIBUTE,
)

# Include the lower age limit, exclude the upper limit
AGE_BINS = [0, 30, 50, 65, 80, np.inf]
AGE_LABELS = ["<30", "30-49", "50-64", "65-79", "80+"]

# Sort ages from youngest to oldest, other groups alphabetically
CATEGORY_ORDERS = {"age_group": AGE_LABELS}

# Match admissions for hospitalisation and ED stays for deterioration
DETERIORATION_STATIC_ID_COL = "general_ed_stay_id"
HOSPITALISATION_STATIC_ID_COL = "last_hadm_id"

# Use the same metric order throughout
RATIO_METRICS = {
    "DPR": ("Demographic Parity", demographic_parity_ratio),
    "EQO": ("Equalised Odds", equalized_odds_ratio),
    "EOP": ("Equal Opportunity", equal_opportunity_ratio),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="Path to a trained checkpoint. Its parent directory identifies the run.",
    )
    source.add_argument(
        "--path",
        help="Root to search recursively for run checkpoints.",
    )
    parser.add_argument(
        "--static",
        help="Demographics CSV override. Defaults to the task-specific source from config.yaml.",
    )
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--n-boot", "-b", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --path mode, skip runs that already have a fairness/ output.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In --path mode, report per-run failures instead of aborting.",
    )
    return parser.parse_args(argv)


def attributes_for_task(task: str) -> tuple[tuple[str, str], ...]:
    """Choose the attributes to compare for this task."""
    return HOSPITALISATION_ATTRIBUTES if task == "hospitalisation" else ATTRIBUTES


def _task_from_config(cfg: dict[str, Any]) -> str:
    """Read the task, falling back to the outcome for older runs."""
    data_cfg = cfg.get("data", {})
    task_value = data_cfg.get("task", data_cfg.get("dataset"))
    if task_value is None:
        outcome = str(data_cfg.get("outcome", cfg.get("outcome", ""))).lower()
        task_value = "hospitalisation" if outcome.startswith("hospitalisation_") else "deterioration"
    return str(task_value).lower()

    
def _collapse_race(value: object) -> str:
    """Map a race description to a broad ethnicity group."""
    if not isinstance(value, str):
        return "Other"
    text = value.upper()
    if text.startswith("WHITE"):
        return "White"
    if text.startswith("BLACK"):
        return "Black"
    if text.startswith("HISPANIC"):
        return "Hispanic"
    if text.startswith("ASIAN"):
        return "Asian"
    return "Other"


def _map_sex(value: object) -> str:
    """Map 1 to Male, 0 to Female and other values to Unknown."""
    if value == 1:
        return "Male"
    if value == 0:
        return "Female"
    return "Unknown"


def load_static_attributes(static_path: str | Path) -> pd.DataFrame:
    """Read the static CSV and prepare demographic groups."""
    static_path = Path(static_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static attributes file not found: {static_path}")

    columns = set(pd.read_csv(static_path, nrows=0).columns)
    if HOSPITALISATION_STATIC_ID_COL in columns:
        id_column = HOSPITALISATION_STATIC_ID_COL
        ethnicity_column = "demographics_ethnicity"
    elif DETERIORATION_STATIC_ID_COL in columns:
        id_column = DETERIORATION_STATIC_ID_COL
        ethnicity_column = "general_race"
    else:
        raise ValueError(
            f"{static_path} has no supported stay identifier. Expected "
            f"{HOSPITALISATION_STATIC_ID_COL!r} or {DETERIORATION_STATIC_ID_COL!r}."
        )
    required = {id_column, "demographics_gender", "demographics_age", ethnicity_column}
    insurance_column = "demographics_insurance"
    if id_column == HOSPITALISATION_STATIC_ID_COL:
        required.add(insurance_column)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{static_path} is missing demographic columns: {missing}")

    frame = pd.read_csv(static_path, usecols=sorted(required))
    frame = frame.dropna(subset=[id_column])
    identifier_column = (
        "hadm_id"
        if id_column == HOSPITALISATION_STATIC_ID_COL
        else "ed_stay_id"
    )
    frame[identifier_column] = frame[id_column].astype("int64")
    frame["sex"] = frame["demographics_gender"].map(_map_sex)
    frame["race_group"] = frame[ethnicity_column].map(_collapse_race)
    frame["age_group"] = pd.cut(
        frame["demographics_age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype("object")
    output_columns = [identifier_column, "sex", "age_group", "race_group"]
    if id_column == HOSPITALISATION_STATIC_ID_COL:
        frame["insurance"] = frame[insurance_column]
        output_columns.append("insurance")
    # Deterioration static has one row per ECG record, so keep one row per stay
    frame = frame[output_columns].drop_duplicates()
    conflicting = int(frame[identifier_column].duplicated().sum())
    if conflicting:
        raise ValueError(
            f"{conflicting} {identifier_column} values in {static_path} have "
            "conflicting demographic attributes."
        )
    return frame


def _is_present(missing: object) -> bool:
    """Check whether any missingness flag marks an available value."""
    values = np.asarray(missing)
    return values.size > 0 and bool(np.any(values.astype("int64") == 0))


def load_cxr_availability(split_parquet: str | Path) -> pd.DataFrame:
    """Find which ED stays have a chest X-ray."""
    split_parquet = Path(split_parquet)
    import pyarrow.parquet as pq

    parquet_columns = set(pq.ParquetFile(split_parquet).schema.names)
    missing_columns = sorted(
        column
        for column in parquet_columns
        if column == "cxr_missing"
        or (column.startswith("cxr_") and column.endswith("_missing"))
    )
    if not missing_columns:
        raise ValueError(f"{split_parquet} has no CXR missingness column.")
    frame = pd.read_parquet(split_parquet, columns=["ed_stay_id", *missing_columns])
    frame["ed_stay_id"] = frame["ed_stay_id"].astype("int64")
    present = np.zeros(len(frame), dtype=bool)
    for column in missing_columns:
        present |= frame[column].map(_is_present).to_numpy(dtype=bool)
    frame["cxr_availability"] = np.where(present, "CXR present", "CXR absent")
    return frame[["ed_stay_id", "cxr_availability"]]


def build_sensitive_frame(
    predictions: pd.DataFrame,
    static: pd.DataFrame,
    cxr: pd.DataFrame | None = None,
    attributes: tuple[tuple[str, str], ...] = DEMOGRAPHIC_ATTRIBUTES,
) -> pd.DataFrame:
    """Match predictions to group attributes and reject missing values."""
    prediction_id = prediction_identifier_column(predictions)
    static_id = prediction_identifier_column(static)
    if prediction_id != static_id:
        raise ValueError(
            "Prediction and static identifiers differ: "
            f"{prediction_id!r} versus {static_id!r}."
        )
    merged = predictions.merge(static, on=prediction_id, how="left")
    if cxr is not None:
        merged = merged.merge(cxr, on="ed_stay_id", how="left")
        if CXR_ATTRIBUTE not in attributes:
            attributes += (CXR_ATTRIBUTE,)
    if len(merged) != len(predictions):
        raise ValueError(
            f"Joining attributes changed {len(predictions)} predictions into "
            f"{len(merged)} rows. Sensitive-attribute sources repeat an identifier."
        )
    for column, _ in attributes:
        missing = int(merged[column].isna().sum())
        if missing:
            raise ValueError(
                f"{missing} of {len(merged)} stays have no {column!r} after the "
                "join. Sensitive-attribute sources do not cover the predictions."
            )
    return merged


def bootstrapped_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Return fairness ratios and confidence bounds, keeping group sizes fixed when resampling."""
    rng = np.random.default_rng(seed)
    group_indices = [np.where(sensitive == g)[0] for g in np.unique(sensitive)]

    samples: dict[str, list[float]] = {name: [] for name in RATIO_METRICS}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(g, size=len(g), replace=True) for g in group_indices]
        )
        yt, yp, sf = y_true[idx], y_pred[idx], sensitive[idx]
        for name, (_, fn) in RATIO_METRICS.items():
            value = fn(yt, yp, sensitive_features=sf)
            # Keep the legacy minimum of 0.001 for odds and opportunity ratios
            samples[name].append(max(value, 0.001) if name != "DPR" else value)

    results: dict[str, dict[str, float]] = {}
    for name, (_, fn) in RATIO_METRICS.items():
        arr = np.asarray(samples[name], dtype=np.float64)
        observed = float(fn(y_true, y_pred, sensitive_features=sensitive))
        observed = max(observed, 0.001) if name != "DPR" else observed
        low, high = bias_corrected_ci(arr, observed)
        results[name] = {"value": observed, "ci_low": low, "ci_high": high}
    return results


def group_error_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> pd.DataFrame:
    """Return error rates, selection rates and counts for each group."""
    frame = MetricFrame(
        metrics={
            "FPR": false_positive_rate,
            "FNR": false_negative_rate,
            "Selection Rate": selection_rate,
            "Count": count,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    return frame.by_group


@plt.rc_context(PLOT_STYLE)
def plot_group_metrics(by_group: pd.DataFrame, attribute: str, save_path: Path) -> None:
    """Plot group rates as percentages alongside group counts."""
    axes_grid = by_group.plot.bar(
        subplots=True,
        layout=(2, 2),
        colormap=PALETTE_CMAP,
        legend=False,
        figsize=(12, 8),
    )
    axes = axes_grid.flatten()
    for index, ax in enumerate(axes):
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        ax.set_xlabel("")
        is_rate = index < len(axes) - 1  # The last chart shows counts
        container = ax.containers[0]
        max_height = max(container.datavalues, default=0.0)
        # Keep rates on a 0–100% scale, extending it for labels when needed
        if is_rate:
            ax.set_ylim(0.0, max(1.0, max_height * 1.18))
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
            )
        else:
            ax.set_ylim(0.0, max_height * 1.18 or 1.0)
        # Show percentages for rates and whole numbers for counts
        labels = [
            f"{v * 100:.0f}%" if is_rate else f"{v:.0f}" for v in container.datavalues
        ]
        ax.bar_label(container, labels=labels, padding=2, fontsize=12)
    fig = axes[0].figure
    # Leave room for the title
    fig.suptitle(f"Error estimates by {attribute}", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=600, transparent=True)
    plt.close(fig)


@plt.rc_context(PLOT_STYLE)
def plot_across_models(
    results: dict[str, dict[str, Any]],
    save_path: Path,
    attributes: tuple[tuple[str, str], ...] = ATTRIBUTES,
) -> None:
    """Compare fairness ratios across models for each attribute."""
    models = list(results)
    metrics = list(RATIO_METRICS)
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(models))]
    n_columns = min(2, len(attributes))
    n_rows = int(np.ceil(len(attributes) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(6.5 * n_columns, 4.5 * n_rows), sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(attributes) :]:
        ax.remove()
    axes = axes[: len(attributes)]
    fig.suptitle("Group-fairness ratios by model", fontsize=22)
    fig.supylabel("Ratio (min/max across groups)")

    width = 0.8 / max(len(models), 1)
    for ax, (_, label) in zip(axes, attributes, strict=True):
        x = np.arange(len(metrics))
        for j, model in enumerate(models):
            attr = results[model]["attributes"].get(label, {})
            values = [attr.get(m, {}).get("value", np.nan) for m in metrics]
            lows = [
                max(attr.get(m, {}).get("value", 0) - attr.get(m, {}).get("ci_low", 0), 0)
                for m in metrics
            ]
            highs = [
                max(attr.get(m, {}).get("ci_high", 0) - attr.get(m, {}).get("value", 0), 0)
                for m in metrics
            ]
            positions = x + j * width
            ax.bar(
                positions,
                values,
                width,
                label=model,
                color=colors[j],
                yerr=[lows, highs],
                capsize=3,
                alpha=0.85,
            )
            # Label values above their confidence intervals
            for xi, value, high in zip(positions, values, highs, strict=True):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value * 100:.0f}%",
                        (xi, value + high),
                        textcoords="offset points",
                        xytext=(0, 3),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax.set_title(label, fontsize=22)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.0, 1.18)  # Leave room for labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(models), 2),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(save_path, dpi=600, transparent=True)
    plt.close(fig)


def resolve_split_parquet(config_path: str | Path, split: str) -> Path:
    """Find the requested split parquet, as the dataset loader does."""

    parquet = splits.resolve_split_parquet(load_config(config_path), split)
    if parquet is None:
        raise ValueError(f"config {config_path} has no data.split_dir")
    return parquet


def resolve_static_path(config_path: str | Path) -> Path:
    """Use the configured static CSV or the task's default path.

    Hospitalisation uses required history only when the split path names it.
    Otherwise, it uses optional history.
    """

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    configured = data_cfg.get("static_path") or cfg.get("static_path")
    if configured:
        return Path(configured)

    task = _task_from_config(cfg)
    if task != "hospitalisation":
        return STATIC_PATHS["deterioration"]

    split_dir = data_cfg.get("split_dir") or data_cfg.get("bundle_dir") or data_cfg.get("data_dir")
    history = "requiredHistory" if "requiredHistory" in str(split_dir) else "optionalHistory"
    return STATIC_PATHS["hospitalisation"][history]


def compute_fairness_for_run(
    predictions_path: str | Path,
    config_path: str | Path,
    static_path: str | Path | None,
    output_dir: str | Path,
    *,
    split: str = "test",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate, save and return fairness results for one run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(predictions_path)

    cfg = load_config(config_path)
    task = _task_from_config(cfg)
    attribute_axes = attributes_for_task(task)
    static = load_static_attributes(static_path or resolve_static_path(config_path))
    cxr = (
        load_cxr_availability(resolve_split_parquet(config_path, split))
        if task != "hospitalisation"
        else None
    )
    frame = build_sensitive_frame(predictions, static, cxr, attribute_axes)

    y_true = frame["label"].to_numpy()
    y_pred = frame["prediction"].to_numpy()

    attributes: dict[str, Any] = {}
    for index, (column, label) in enumerate(attribute_axes, start=1):
        print(
            f" [{index}/{len(attribute_axes)}] Processing {label}..."
        )
        sensitive = frame[column].astype("object")
        by_group = group_error_metrics(y_true, y_pred, sensitive)
        order = CATEGORY_ORDERS.get(column)
        if order is not None:
            by_group = by_group.reindex([g for g in order if g in by_group.index])
        plot_group_metrics(by_group, label, output_dir / f"{split}_error_by_{column}.png")
        ratios = bootstrapped_fairness(
            y_true, y_pred, sensitive.to_numpy(), n_boot=n_boot, seed=seed
        )
        print(
            f"        DPR={ratios['DPR']['value']:.3f}  "
            f"EqOdds={ratios['EQO']['value']:.3f}  "
            f"EqOpp={ratios['EOP']['value']:.3f}"
        )
        attributes[label] = {
            **ratios,
            "groups": {
                str(group): {
                    "count": int(row["Count"]),
                    "selection_rate": float(row["Selection Rate"]),
                    "fpr": float(row["FPR"]),
                    "fnr": float(row["FNR"]),
                }
                for group, row in by_group.iterrows()
            },
        }

    results = {"run": output_dir.parent.name, "n": int(len(frame)), "attributes": attributes}
    with (output_dir / f"{split}_fairness_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    (output_dir / f"{split}_fairness_report.txt").write_text(
        format_report(results, split), encoding="utf-8"
    )
    print(f" Saved at: {output_dir}")
    return results


def format_report(results: dict[str, Any], split: str) -> str:
    """Format ratios, confidence intervals and group rates as a text report."""
    header = f"\nFairness report — {results['run']}  (split: {split}, n={results['n']})"
    lines = [header, "=" * len(header)]
    for label, attr in results["attributes"].items():
        lines.append("")
        lines.append(label)
        for key, (disp, _) in RATIO_METRICS.items():
            cell = attr[key]
            lines.append(
                f"  {disp:<19}: {cell['value']:.3f}  "
                f"[{cell['ci_low']:.3f}, {cell['ci_high']:.3f}]"
            )
        lines.append(
            f"  {'group':<14}{'count':>7}{'sel%':>8}{'FPR%':>8}{'FNR%':>8}{'TPR%':>8}"
        )
        for group, row in attr["groups"].items():
            lines.append(
                f"  {group:<14}{row['count']:>7}"
                f"{row['selection_rate'] * 100:>8.1f}"
                f"{row['fpr'] * 100:>8.1f}"
                f"{row['fnr'] * 100:>8.1f}"
                f"{(1 - row['fnr']) * 100:>8.1f}"
            )
    return "\n".join(lines) + "\n"


def write_summary_csv(all_results: dict[str, dict[str, Any]], path: Path) -> None:
    """Save one CSV row per run, attribute and metric."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "attribute", "metric", "value", "ci_low", "ci_high"])
        for run, results in all_results.items():
            for label, attr in results["attributes"].items():
                for metric in RATIO_METRICS:
                    cell = attr[metric]
                    writer.writerow(
                        [
                            run,
                            label,
                            metric,
                            f"{cell['value']:.4f}",
                            f"{cell['ci_low']:.4f}",
                            f"{cell['ci_high']:.4f}",
                        ]
                    )



def _run_single(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    predictions_path = run_dir / f"{args.split}_predictions.csv"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config for {run_dir} (looked for {config_path})."
        )
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions at {predictions_path}")
    output_dir = run_dir / DEFAULT_OUTPUT_SUBDIR
    return compute_fairness_for_run(
        predictions_path,
        config_path,
        args.static,
        output_dir,
        split=args.split,
        n_boot=args.n_boot,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:  
    args = parse_args(argv)

    if args.path is not None:
        checkpoints = discover_checkpoints(args.path)
        if not checkpoints:
            print(f"No '{DEFAULT_CHECKPOINT_NAME}' checkpoints found under {args.path}.")
            return
        print(f"Found {len(checkpoints)} checkpoint(s) under {args.path}.")
        all_results: dict[str, dict[str, Any]] = {}
        failures: list[tuple[Path, str]] = []
        for index, checkpoint in enumerate(checkpoints, start=1):
            run_dir = checkpoint.parent
            metrics_json = run_dir / DEFAULT_OUTPUT_SUBDIR / f"{args.split}_fairness_metrics.json"
            if args.skip_existing and metrics_json.exists():
                print(f"[{index}/{len(checkpoints)}] skip (exists): {run_dir}")
                all_results[run_dir.name] = json.loads(metrics_json.read_text())
                continue
            print(f"[{index}/{len(checkpoints)}] {run_dir}")
            try:
                all_results[run_dir.name] = _run_single(run_dir, args)
            except Exception as error:  # Continue on error when requested
                if not args.continue_on_error:
                    raise
                failures.append((run_dir, f"{type(error).__name__}: {error}"))
                print(f"    FAILED: {type(error).__name__}: {error}")

        if all_results:
            root = Path(args.path)
            summary_dir = root / DEFAULT_OUTPUT_SUBDIR
            summary_dir.mkdir(parents=True, exist_ok=True)
            write_summary_csv(all_results, summary_dir / "fairness_summary.csv")
            first_attributes = set(next(iter(all_results.values()))["attributes"])
            attribute_order = (
                HOSPITALISATION_ATTRIBUTES
                if "Insurance" in first_attributes
                else ATTRIBUTES
            )
            summary_attributes = tuple(
                attribute
                for attribute in attribute_order
                if attribute[1] in first_attributes
            )
            plot_across_models(
                all_results,
                summary_dir / "fairness_across_models.png",
                attributes=summary_attributes,
            )
            print(f"Summary: {summary_dir / 'fairness_summary.csv'}")
        if failures:
            print(f"{len(failures)} run(s) failed:")
            for run_dir, message in failures:
                print(f"  {run_dir}: {message}")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    run_dir = checkpoint.parent

    results = _run_single(run_dir, args)
    print(format_report(results, args.split))
    print(f"Saved fairness outputs under {run_dir / DEFAULT_OUTPUT_SUBDIR}")


if __name__ == "__main__":
    main()
