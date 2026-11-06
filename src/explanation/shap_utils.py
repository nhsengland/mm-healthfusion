"""SHAP helpers for tabular features and MM-SHAP contributions by input type."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

from src.data.collate import get_collate_fn
from src.data.splits import get_datasets
from src.evaluation.eval_utils import load_config
from src.explanation import freeze_m3care

MODALITIES: tuple[str, ...] = ("tabular", "ecg", "cxr")

# Seed for repeatable SHAP sampling
DEFAULT_SHAP_SEED = 42

_MMSHAP_LABELS: dict[str, str] = {"tabular": "TB-SHAP", "ecg": "ECG-SHAP", "cxr": "CXR-SHAP"}


def _modalities_from_batch(
    batch: Mapping[str, Any],
    fallback: Sequence[str] = MODALITIES,
) -> tuple[str, ...]:
    """Return input names in batch order, or use the supplied defaults."""
    modalities = batch.get("modality_names", ())
    if modalities:
        return tuple(str(modality) for modality in modalities)
    return tuple(fallback)


def _tabular_modality_name(modalities: Sequence[str]) -> str:
    """Find the tabular or static input used for feature explanations."""
    for name in modalities:
        if name in {"tabular", "static"}:
            return str(name)
    raise ValueError(
        "Could not identify the feature-interpretable modality. Expected one "
        "of 'tabular' or 'static'."
    )


def _display_feature_name(name: str) -> str:
    """Restore the original acuity category in a feature name for plotting."""
    for prefix in ("vitals_acuity=", "vitals_acuity_"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            try:
                original = int(round(float(suffix))) + 1
            except ValueError:
                return name  # eg. vitals_acuity_missing
            return f"{prefix}{original}"
    return name


def _is_onehot_indicator(name: str) -> bool:
    """Check whether a category column should be displayed as Yes or No."""
    text = str(name)
    if "=" in text:
        return True
    prefix = "vitals_acuity_"
    if not text.startswith(prefix):
        return False
    suffix = text[len(prefix):]
    if suffix == "missing":
        return True
    try:
        float(suffix)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------------------
# 1. Recover tabular feature names
# --------------------------------------------------------------------------------------
def _resolve_data_path(path: str | Path | None) -> Path | None:
    """Convert a saved path to Path without checking whether it exists."""
    return None if path is None else Path(path)


def _colocated_tabular_csv(summary_path: Path) -> Path | None:
    """Find the first tabular CSV beside the summary or in its parent folder."""
    bundle_dir = summary_path.parent
    for search_dir in (bundle_dir, bundle_dir.parent):
        matches = [
            m for m in sorted(search_dir.glob("*_tabular.csv"))
            if not m.name.startswith("._")
        ]
        if matches:
            return matches[0]
    return None


def _resolve_tabular_csv(
    summary_path: Path,
    summary: Mapping[str, Any],
    source_csv: str | Path | None = None,
) -> Path | None:
    """Use the supplied CSV, then the saved path, then a nearby tabular CSV."""
    if source_csv is not None:
        return Path(source_csv)
    recorded = _resolve_data_path(summary.get("input_tabular_path"))
    if recorded is not None and recorded.exists():
        return recorded
    return _colocated_tabular_csv(summary_path)


def load_original_tabular_values(
    summary_path: str | Path,
    feature_names: Sequence[str],
    ed_stay_ids: Sequence[int],
    source_csv: str | Path | None = None,
    identifier_column: str = "ed_stay_id",
) -> dict[int, dict[str, float]]:
    """Read original clinical measurements for plot labels, keyed by stay ID.

    Return an empty dictionary if the source data cannot be read.
    """
    summary_path = Path(summary_path)
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError):
        return {}

    csv_path = _resolve_tabular_csv(summary_path, summary, source_csv)
    if csv_path is None or not csv_path.exists():
        return {}

    identifier_values = [int(value) for value in ed_stay_ids]
    if not identifier_values:
        return {}

    # Category columns already include the category in their name
    wanted = [n for n in feature_names if "=" not in n]
    try:
        scan = pl.scan_csv(csv_path)
        available = set(scan.columns)
        source_id_candidates = (
            ("ed_hadm_id", "hadm_id")
            if identifier_column == "hadm_id"
            else ("ed_stay_id", "general_ed_stay_id")
        )
        source_id_column = next(
            (column for column in source_id_candidates if column in available), None
        )
        if source_id_column is None:
            return {}
        columns = [n for n in dict.fromkeys(wanted) if n in available]
        if not columns:
            return {}
        frame = (
            scan.filter(pl.col(source_id_column).is_in(identifier_values))
            .select([source_id_column, *columns])
            .collect()
        )
    except Exception:  # Keep plotting if labelling fails
        return {}

    out: dict[int, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        stay = int(row[source_id_column])
        out[stay] = {
            name: float(row[name])
            for name in columns
            if row[name] is not None and not (isinstance(row[name], float) and math.isnan(row[name]))
        }
    return out


def _format_original_value(value: float) -> str:
    """Show whole numbers without decimals and other values to three significant figures."""
    if not math.isfinite(value):
        return "n/a"
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    # Keep three significant figures without labels such as "1.23e+03"
    rounded = float(f"{value:.3g}")
    if rounded.is_integer() and abs(rounded) < 1e15:
        return str(int(rounded))
    return f"{rounded:f}".rstrip("0").rstrip(".")


def _expand_ohe_names(csv_path: Path, categorical_columns: Sequence[str]) -> list[str]:
    """Rebuild category column names in the order used when preparing the data."""
    if not categorical_columns:
        return []
    cat_df = pd.read_csv(csv_path, usecols=list(categorical_columns), low_memory=False)
    ohe = pd.get_dummies(
        cat_df[list(categorical_columns)],
        columns=list(categorical_columns),
        prefix=list(categorical_columns),
        prefix_sep="=",
        dummy_na=False,
        dtype="int8",
    )
    return ohe.columns.tolist()


def get_feature_names(
    summary_path: str | Path,
    expected_width: int | None = None,
    source_csv: str | Path | None = None,
) -> list[str]:
    """Read feature names in model input order and check their count if requested.

    For older summaries without saved names, rebuild them from the source CSV.
    """
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())

    stored = summary.get("tab_feats_columns")
    if stored:
        names = list(stored)
        if expected_width is not None and len(names) != expected_width:
            raise ValueError(
                f"tab_feats_columns in {summary_path.name} has {len(names)} names but "
                f"tab_feats width is {expected_width}."
            )
        return names

    continuous = list(summary.get("continous_columns", []))  # Spelling used in the saved summary
    categorical = list(summary.get("categorical_columns", []))

    csv_path = _resolve_tabular_csv(summary_path, summary, source_csv)

    if categorical and (csv_path is None or not csv_path.exists()):
        raise FileNotFoundError(
            "Cannot recover one-hot feature names: source tabular CSV not found. "
            f"Tried input_tabular_path={summary.get('input_tabular_path')!r} and a "
            f"co-located *_tabular.csv next to {summary_path.parent}. Pass source_csv=..., "
            "or re-run preprocessing to persist tab_feats_columns into the summary."
        )

    ohe_columns = _expand_ohe_names(csv_path, categorical) if categorical else []
    names = continuous + ohe_columns

    if expected_width is not None and len(names) != expected_width:
        raise ValueError(
            f"Reconstructed {len(names)} tabular feature names "
            f"({len(continuous)} continuous + {len(ohe_columns)} one-hot) but tab_feats "
            f"width is {expected_width}. Bundle and source CSV may be mismatched."
        )
    return names


# --------------------------------------------------------------------------------------
# 2. Shared batch helpers
# --------------------------------------------------------------------------------------
def _clone_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy batch tensors and keep other values unchanged."""
    return {
        key: (value.clone() if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def _slice_batch_range(
    batch: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    """Take rows from start up to stop, keeping values that are not tensors unchanged."""
    return {
        key: (value[start:stop] if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def _slice_batch(batch: Mapping[str, torch.Tensor], i: int) -> dict[str, torch.Tensor]:
    """Copy row i of each batch tensor and keep other values unchanged."""
    return {
        key: (value.clone() if isinstance(value, torch.Tensor) else value)
        for key, value in _slice_batch_range(batch, i, i + 1).items()
    }


def _tile_reference_batch(
    ref: Mapping[str, torch.Tensor], n: int, device: torch.device | None = None
) -> dict[str, torch.Tensor]:
    """Repeat a single patient n times, leaving tab_feats for the caller to supply."""
    batch: dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if not isinstance(value, torch.Tensor):
            batch[key] = value
            continue
        if key == "tab_feats":
            continue  # The caller provides the tabular data for this run
        if device is not None:
            value = value.to(device)
        batch[key] = (value.repeat((n,) + (1,) * (value.ndim - 1))
                      if value.ndim >= 1 else value.repeat(n))
    return batch


def _all_coalitions(modalities: Sequence[str]) -> list[tuple[str, ...]]:
    """List all input combinations, from none to all, preserving input order within each size."""
    modalities = tuple(modalities)
    return [subset for r in range(len(modalities) + 1)
            for subset in itertools.combinations(modalities, r)]


def git_revision() -> dict[str, Any]:
    """Return the Git revision and whether the worktree has changes, or None if unavailable."""
    repo = Path(__file__).resolve().parent

    def _git(*args: str):
        # Handle Git errors below instead of raising them here
        return subprocess.run(("git", "-C", str(repo)) + args, check=False,
                              capture_output=True, text=True, timeout=10)

    try:
        rev = _git("rev-parse", "--short", "HEAD")
        if rev.returncode != 0:
            return {"git_rev": None, "git_dirty": None}
        status = _git("status", "--porcelain")
        return {"git_rev": rev.stdout.strip(),
                "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"git_rev": None, "git_dirty": None}


# Let SHAP get predictions while changing the tabular inputs
@torch.no_grad()
def predict_probability(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor],
) -> np.ndarray:
    """Return one predicted outcome probability per batch row as a flat NumPy array."""
    device = next(model.parameters()).device
    moved = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    output = model.predict_step(moved, 0)
    return output["probability"].detach().cpu().numpy().reshape(-1)


def aggregate_ts(feats: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Average each patient's features over their recorded time steps.

    Leave data without a time dimension unchanged. If lengths are missing, average
    all time steps.
    """
    if feats.ndim == 2:
        return feats
    if feats.ndim != 3:
        raise ValueError(f"expected (B,T,F) or (B,F) tab_feats, got shape {tuple(feats.shape)}")
    batch_size, max_t, _ = feats.shape
    if lengths is None:
        return feats.mean(dim=1)
    pooled = []
    for i in range(batch_size):
        n = int(lengths[i].item()) if lengths[i].numel() == 1 else int(lengths[i])
        n = max(1, min(n, max_t))
        pooled.append(feats[i, :n].mean(dim=0))
    return torch.stack(pooled, dim=0)


def _single_row_reference(
    reference_batch: Mapping[str, torch.Tensor],
    tabular_is_sequential: bool,
    caller: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int | None]:
    """Copy a single patient and average their tabular time steps if needed."""
    ref = _clone_batch(reference_batch)
    n_ref = int(ref["label"].shape[0]) if "label" in ref else int(ref["tab_feats"].shape[0])
    if n_ref != 1:
        raise ValueError(f"reference_batch for {caller} must have exactly one row")
    tab = ref["tab_feats"]
    if tabular_is_sequential:
        return ref, aggregate_ts(tab, ref.get("tab_lengths")), int(tab.shape[1])
    return ref, tab, None


def _make_tabular_predict_fn(
    model: torch.nn.Module,
    reference_batch: Mapping[str, torch.Tensor],
    tabular_is_sequential: bool,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Build a prediction function that changes tabular data and keeps other inputs fixed."""
    ref, ref_tab, max_t = _single_row_reference(
        reference_batch, tabular_is_sequential, "_make_tabular_predict_fn")
    ref_tab_matrix = ref_tab.detach().cpu().numpy()  # (1, F)

    def f(tab_matrix: np.ndarray) -> np.ndarray:
        tab_matrix = np.atleast_2d(np.asarray(tab_matrix, dtype=np.float32))
        n = tab_matrix.shape[0]
        batch = _tile_reference_batch(ref, n)
        tab_t = torch.from_numpy(tab_matrix)
        if tabular_is_sequential:
            # Repeat the average at each time step so averaging again gives the same value
            tab_t = tab_t.unsqueeze(1).repeat(1, max_t, 1)
        batch["tab_feats"] = tab_t
        return predict_probability(model, batch)

    return f, ref_tab_matrix


def _make_tabular_grad_module(
    model: torch.nn.Module,
    reference_batch: Mapping[str, torch.Tensor],
    tabular_is_sequential: bool,
) -> tuple[torch.nn.Module, torch.Tensor]:
    """Wrap tabular predictions for Gradient SHAP while keeping other inputs fixed."""
    ref, ref_tab, max_t = _single_row_reference(
        reference_batch, tabular_is_sequential, "_make_tabular_grad_module")
    device = next(model.parameters()).device
    ref_tab = ref_tab.detach().to(device=device, dtype=torch.float32)  # (1, F)

    class _TabularProbModule(torch.nn.Module):
        def forward(self, tab_matrix: torch.Tensor) -> torch.Tensor:
            tab_matrix = tab_matrix.to(device=device, dtype=torch.float32)
            batch = _tile_reference_batch(ref, tab_matrix.shape[0], device=device)
            if tabular_is_sequential:
                # Repeating this way lets SHAP calculate how input changes affect predictions
                tab_matrix = tab_matrix.unsqueeze(1).repeat(1, max_t, 1)
            batch["tab_feats"] = tab_matrix
            return torch.sigmoid(model(batch).reshape(-1)).unsqueeze(-1)  # (n, 1)

    return _TabularProbModule(), ref_tab


# --------------------------------------------------------------------------------------
# 3. Explain tabular features with SHAP
# --------------------------------------------------------------------------------------
def _select_engine(engine: str) -> str:
    """Check that the requested SHAP method is supported."""
    if engine not in {"kernel", "gradient"}:
        raise ValueError(f"engine must be 'kernel' or 'gradient', got {engine!r}")
    return engine


def _prepare_background(
    background_batch: Mapping[str, torch.Tensor],
    tabular_is_sequential: bool,
    chosen: str,
    *,
    device: torch.device,
    kernel_background_summary: int,
    gradient_nsamples: int | str,
):
    """Prepare reference samples for the chosen SHAP method."""
    import shap

    bg = _clone_batch(background_batch)
    if tabular_is_sequential:
        matrix = aggregate_ts(bg["tab_feats"], bg.get("tab_lengths")).detach().cpu().numpy()
    else:
        matrix = bg["tab_feats"].detach().cpu().numpy()
    matrix = np.atleast_2d(matrix).astype(np.float32)
    if chosen == "kernel":
        k = min(kernel_background_summary, matrix.shape[0])
        return (shap.kmeans(matrix, k) if k > 1 else matrix), None
    grad_ns = 200 if gradient_nsamples == "auto" else int(gradient_nsamples)
    return torch.from_numpy(matrix).to(device=device, dtype=torch.float32), grad_ns


def _kernel_row(f, ref_tab, bg_summary, nsamples, seed) -> tuple[np.ndarray, float]:
    """Return feature contributions and the reference prediction using Kernel SHAP."""
    import shap

    explainer = shap.KernelExplainer(f, bg_summary)
    # KernelExplainer uses NumPy's random seed
    np.random.seed(int(seed))
    vals = explainer.shap_values(ref_tab, nsamples=nsamples, silent=True)
    return (np.asarray(vals).reshape(-1),
            float(np.asarray(explainer.expected_value).reshape(-1)[0]))


def _gradient_row(module, ref_tab, bg_tensor, nsamples, seed) -> tuple[np.ndarray, float]:
    """Return feature contributions and the reference prediction using Gradient SHAP."""
    import shap

    explainer = shap.GradientExplainer(module, bg_tensor)
    vals = explainer.shap_values(ref_tab, nsamples=nsamples, rseed=int(seed))
    if isinstance(vals, list):
        vals = vals[0]
    with torch.no_grad():
        base = float(module(bg_tensor).mean().item())
    return np.asarray(vals).reshape(-1), base


def get_shap_values(
    model: torch.nn.Module,
    explain_batch: Mapping[str, torch.Tensor],
    background_batch: Mapping[str, torch.Tensor],
    feature_names: Sequence[str],
    tabular_is_sequential: bool = False,
    engine: str = "gradient",
    kernel_nsamples: int | str = "auto",
    kernel_background_summary: int = 50,
    gradient_nsamples: int | str = "auto",
    shap_seed: int = DEFAULT_SHAP_SEED,
):
    """Explain tabular features in batch row order while keeping other inputs fixed."""
    import shap

    model.eval()
    feature_names = list(feature_names)
    n_features = len(feature_names)
    chosen = _select_engine(engine)

    background, grad_ns = _prepare_background(
        background_batch, tabular_is_sequential, chosen,
        device=next(model.parameters()).device,
        kernel_background_summary=kernel_background_summary,
        gradient_nsamples=gradient_nsamples,
    )

    # Keep each patient's other inputs fixed while explaining their tabular data
    n_explain = int(
        explain_batch["label"].shape[0] if "label" in explain_batch
        else explain_batch["tab_feats"].shape[0]
    )
    all_values = np.zeros((n_explain, n_features), dtype=np.float64)
    all_data = np.zeros((n_explain, n_features), dtype=np.float64)
    base_values = np.zeros(n_explain, dtype=np.float64)

    for i in tqdm(range(n_explain), desc="Tabular feature SHAP values", unit="patients"):
        row_batch = _slice_batch(explain_batch, i)
        if chosen == "gradient":
            module, ref_tab_t = _make_tabular_grad_module(
                model, row_batch, tabular_is_sequential)
            all_values[i], base_values[i] = _gradient_row(
                module, ref_tab_t, background, grad_ns, int(shap_seed) + i)
            ref_tab = ref_tab_t.detach().cpu().numpy()
        else:
            f, ref_tab = _make_tabular_predict_fn(
                model, row_batch, tabular_is_sequential)
            all_values[i], base_values[i] = _kernel_row(
                f, ref_tab, background, kernel_nsamples, int(shap_seed) + i)
        all_data[i] = ref_tab.reshape(-1)

    return shap.Explanation(
        values=all_values, base_values=base_values, data=all_data, feature_names=feature_names,
    )


def _onehot_group_key(name: str) -> str | None:
    """Return the feature name before =, or None if there is no category suffix."""
    text = str(name)
    return text.split("=", 1)[0] if "=" in text else None


def merge_onehot_shap(explanation, *, merge_binary: bool = True, merge_multiclass: bool = False):
    """Combine category columns by adding their SHAP contributions.

    Keep groups in the order they first appear and leave the reference prediction
    unchanged. By default, combine only features with two category columns.
    """
    import shap

    names = list(explanation.feature_names)
    values = np.asarray(explanation.values, dtype=np.float64)
    data = np.asarray(explanation.data, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
        data = data[None, :]

    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for j, nm in enumerate(names):
        key = _onehot_group_key(nm)
        gid = key if key is not None else f"__single__{j}"
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(j)

    new_names: list[str] = []
    new_vals: list[np.ndarray] = []
    new_data: list[np.ndarray] = []
    for gid in order:
        idx = groups[gid]
        is_ohe = not gid.startswith("__single__")
        merge = is_ohe and (
            (len(idx) == 2 and merge_binary) or (len(idx) > 2 and merge_multiclass)
        )
        if merge:
            new_names.append(gid)
            new_vals.append(values[:, idx].sum(axis=1))
            # Show the selected category's numeric suffix, or 0/1 for a yes/no value
            suffixes = [str(names[j]).split("=", 1)[1] for j in idx]
            active = data[:, idx].argmax(axis=1)  # Selected category in each row
            row_val = np.empty(data.shape[0], dtype=np.float64)
            for r in range(data.shape[0]):
                try:
                    row_val[r] = float(suffixes[active[r]])
                except ValueError:
                    row_val[r] = float(active[r])
            new_data.append(row_val)
        else:
            for j in idx:
                new_names.append(names[j])
                new_vals.append(values[:, j])
                new_data.append(data[:, j])

    out_values = np.stack(new_vals, axis=1)
    out_data = np.stack(new_data, axis=1)
    if single:
        out_values = out_values[0]
        out_data = out_data[0]
    return shap.Explanation(
        values=out_values, base_values=explanation.base_values,
        data=out_data, feature_names=new_names,
    )


# --------------------------------------------------------------------------------------
# 4. Measure each input type's contribution with MM-SHAP
# --------------------------------------------------------------------------------------
def _mask_batch_to_coalition(batch: Mapping[str, torch.Tensor], present: tuple[str, ...]):
    """Mark every input type outside present as missing using the evaluation helper."""
    from src.evaluation.modality_ablation import ModalityScenario, mask_batch_for_scenario

    scenario = ModalityScenario(name="+".join(present) if present else "none",
                                modalities=tuple(present))
    return mask_batch_for_scenario(batch, scenario)


def modality_coalition_values(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    modalities: Sequence[str] | None = None,
    show_progress: bool = True,
) -> dict[frozenset, np.ndarray]:
    """Get predictions for every combination of input types, including no inputs."""
    modalities = _modalities_from_batch(batch) if modalities is None else tuple(modalities)
    values: dict[frozenset, np.ndarray] = {}
    for subset in tqdm(_all_coalitions(modalities), desc="MM-SHAP modality coalitions",
                       unit="coalition",
                       disable=not show_progress):
        masked = _mask_batch_to_coalition(batch, subset)
        values[frozenset(subset)] = predict_probability(model, masked)
    return values


def shapley_from_coalitions(
    coalition_values: Mapping[frozenset, np.ndarray],
    modalities: Sequence[str] = MODALITIES,
) -> dict[str, np.ndarray]:
    """Calculate each input type's Shapley contribution from all input combinations."""
    modalities = tuple(modalities)
    M = len(modalities)
    any_val = next(iter(coalition_values.values()))
    n = any_val.shape[0]
    phi = {m: np.zeros(n, dtype=np.float64) for m in modalities}
    factorial = math.factorial
    for m in modalities:
        others = [x for x in modalities if x != m]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                weight = factorial(len(S)) * factorial(M - len(S) - 1) / factorial(M)
                v_with = coalition_values[S | {m}]
                v_without = coalition_values[S]
                phi[m] += weight * (v_with - v_without)
    return phi


def estimate_mm_summary(
    shapley_values: Mapping[str, np.ndarray],
    modalities: Sequence[str] = MODALITIES,
) -> dict[str, Any]:
    """Express absolute contributions as shares for each patient and average those shares.

    Keep the supplied input order. Patients with no contribution have all-zero shares.
    """
    modalities = tuple(modalities)
    abs_phi = np.stack([np.abs(shapley_values[m]) for m in modalities], axis=1)  # (n, M)
    total = abs_phi.sum(axis=1, keepdims=True)
    safe_total = np.where(total == 0, 1.0, total)
    per_sample = abs_phi / safe_total  # (n, M)
    global_contrib = per_sample.mean(axis=0)  # (M,)
    return {
        "modalities": list(modalities),
        "per_sample_contribution": per_sample,          # Patient rows, with all-zero rows unchanged
        "global_contribution": {                        # Mean shares across patients
            m: float(global_contrib[i]) for i, m in enumerate(modalities)
        },
        "mean_abs_shapley": {
            m: float(np.abs(shapley_values[m]).mean()) for m in modalities
        },
    }


def compute_mm_shap(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    modalities: Sequence[str] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Calculate input contributions, their shares and predictions for one batch."""
    modalities = _modalities_from_batch(batch) if modalities is None else tuple(modalities)
    coalitions = modality_coalition_values(
        model, batch, modalities, show_progress=show_progress,
    )
    phi = shapley_from_coalitions(coalitions, modalities)
    summary = estimate_mm_summary(phi, modalities)

    v_all = coalitions[frozenset(modalities)]
    v_none = coalitions[frozenset()]
    phi_sum = np.sum([phi[m] for m in modalities], axis=0)
    efficiency_residual = float(np.max(np.abs(phi_sum - (v_all - v_none))))

    summary["shapley_values"] = {m: phi[m] for m in modalities}
    summary["efficiency_residual"] = efficiency_residual
    # Include the prediction with all inputs present so callers can show it beside the contributions
    summary["v_all"] = v_all
    summary["v_all_mean"] = float(v_all.mean())
    summary["v_none_mean"] = float(v_none.mean())
    return summary


# --------------------------------------------------------------------------------------
# 4b. Explain M3Care while keeping links between patients fixed
# --------------------------------------------------------------------------------------
# M3Care uses other patients to fill in missing data, so keep the reference batch
# and its patient links while changing the target patient's inputs
# src.explanation.freeze_m3care handles the model calls


class M3CareFrozenGraphScorer:
    """Score changes to one patient while keeping M3Care's patient links fixed."""

    def __init__(self, model: torch.nn.Module, graph_batch: Mapping[str, torch.Tensor],
                 tabular_is_sequential: bool = False):
        self._frozen = freeze_m3care
        self.inner = model.model
        self.inner.eval()
        self.device = next(self.inner.parameters()).device
        self.tabular_is_sequential = bool(tabular_is_sequential)
        self.graph_batch = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in graph_batch.items()
        }
        ref = freeze_m3care.capture_reference_graph(self.inner, self.graph_batch)
        self.emb = ref["embeddings"]          # {modality: (B, H)}
        self.masks = ref["masks"]             # {modality: (B, 1)}
        self.adj = ref["adjacency"]           # (B, B)
        self.modality_names = tuple(getattr(self.inner, "modality_names", self.emb.keys()))
        self.tabular_modality = _tabular_modality_name(self.modality_names)
        self.B = int(self.adj.shape[0])
        self.tab_lengths = self.graph_batch.get("tab_lengths")
        self.max_t = (int(self.graph_batch["tab_feats"].shape[1])
                      if self.tabular_is_sequential else None)

    def batch_len(self) -> int:
        """Return the number of patients in the reference batch."""
        return self.B

    def identifiers(self) -> np.ndarray | None:
        """Return admission IDs, or emergency department stay IDs if admissions are unavailable."""
        ids = next(
            (self.graph_batch[column] for column in ("hadm_id", "ed_stay_id")
             if column in self.graph_batch),
            None,
        )
        return (ids.detach().cpu().numpy().reshape(-1)
                if isinstance(ids, torch.Tensor) else None)

    def ed_stay_ids(self) -> np.ndarray | None:
        """Keep the older method name for callers that request stay IDs."""
        return self.identifiers()

    def reference_tab_matrix(self, target_idx: int) -> np.ndarray:
        """Return the patient's tabular row as float32, averaging time steps if needed."""
        tab = self.graph_batch["tab_feats"][target_idx:target_idx + 1]
        if self.tabular_is_sequential:
            lengths = (None if self.tab_lengths is None
                       else self.tab_lengths[target_idx:target_idx + 1])
            tab = aggregate_ts(tab, lengths)
        return tab.detach().cpu().numpy().astype(np.float32)

    def _broadcast_target(self, target_idx: int,
                          replaced_emb: Mapping[str, torch.Tensor],
                          replaced_mask: Mapping[str, torch.Tensor] | None = None):
        """Copy the reference batch for each replacement and change only the target patient."""
        any_rep = next(iter(replaced_emb.values()))
        C = int(any_rep.shape[0])
        node = {n: self.emb[n].unsqueeze(0).expand(C, -1, -1).clone() for n in self.emb}
        mask = {n: self.masks[n].unsqueeze(0).expand(C, -1, -1).clone() for n in self.masks}
        for n, val in replaced_emb.items():
            node[n][:, target_idx, :] = val
        if replaced_mask is not None:
            for n, val in replaced_mask.items():
                mask[n][:, target_idx, :] = val
        return node, mask

    def tabular_predict_fn(self, target_idx: int, chunk_size: int = 2048):
        """Build a Kernel SHAP prediction function for one patient's tabular data.

        Keep their other inputs, the other patients and the patient links fixed.
        """
        inner = self.inner
        frozen = self._frozen
        tab_lengths_t = None
        if self.tabular_is_sequential and self.tab_lengths is not None:
            tab_lengths_t = self.tab_lengths[target_idx]

        @torch.no_grad()
        def f(tab_matrix: np.ndarray) -> np.ndarray:
            tab_matrix = np.atleast_2d(np.asarray(tab_matrix, dtype=np.float32))
            n = tab_matrix.shape[0]
            out = np.empty(n, dtype=np.float64)
            for s in range(0, n, chunk_size):
                chunk = tab_matrix[s:s + chunk_size]
                c = chunk.shape[0]
                tt = torch.from_numpy(chunk).to(self.device)                 # (c, F)
                if self.tabular_is_sequential:
                    tt = tt.unsqueeze(1).repeat(1, self.max_t, 1)            # (c, T, F)
                    lengths = None if tab_lengths_t is None else tab_lengths_t.repeat(c)
                    tab_emb = inner.modality_encoder(self.tabular_modality)(tt, lengths)  # (c, H)
                else:
                    tab_emb = inner.modality_encoder(self.tabular_modality)(tt, None)  # (c, H)
                node, mask = self._broadcast_target(target_idx, {self.tabular_modality: tab_emb})
                logits = frozen.forward_target_from_embeddings(
                    inner, node, mask, self.adj, target_idx)                # (c,)
                out[s:s + c] = torch.sigmoid(logits).double().cpu().numpy().reshape(-1)
            return out

        return f

    def gradient_module(self, target_idx: int) -> torch.nn.Module:
        """Build the Gradient SHAP wrapper for one patient with patient links kept fixed."""
        return _FrozenTabProbModule(self, target_idx)

    @torch.no_grad()
    def coalition_values(self, target_idx: int,
                         modalities: Sequence[str] | None = None) -> dict[frozenset, np.ndarray]:
        """Predict for one patient with each combination of input types present.

        Use the model's missing-data handling for absent inputs and keep patient links fixed.
        """
        inner = self.inner
        frozen = self._frozen
        modalities = self.modality_names if modalities is None else tuple(modalities)
        subsets = _all_coalitions(modalities)
        ncoal = len(subsets)

        # Encode all missing inputs once, since each input type is encoded independently
        trow = _slice_batch(self.graph_batch, target_idx)
        missing_batch = _mask_batch_to_coalition(trow, ())
        missing_batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                         for k, v in missing_batch.items()}
        e_missing = inner.encode(missing_batch)                     # {modality: (1, H)}
        m_missing = inner._modality_mask(missing_batch, e_missing)  # {modality: (1, 1)}

        node = {n: self.emb[n].unsqueeze(0).expand(ncoal, -1, -1).clone() for n in self.emb}
        mask = {n: self.masks[n].unsqueeze(0).expand(ncoal, -1, -1).clone() for n in self.masks}
        for ci, subset in enumerate(subsets):
            present = set(subset)
            for n in modalities:
                if n in present:
                    node[n][ci, target_idx, :] = self.emb[n][target_idx]
                    mask[n][ci, target_idx, :] = self.masks[n][target_idx]
                else:
                    node[n][ci, target_idx, :] = e_missing[n][0]
                    mask[n][ci, target_idx, :] = m_missing[n][0]

        logits = frozen.forward_target_from_embeddings(inner, node, mask, self.adj, target_idx)
        probs = torch.sigmoid(logits).double().cpu().numpy().reshape(-1)
        return {frozenset(subset): np.array([probs[ci]], dtype=np.float64)
                for ci, subset in enumerate(subsets)}


class SplitBatches:
    """Load patients in the same order and batches used for model evaluation.

    Keep the other patients in each batch when explaining a model that links
    their predictions. Reuse scorers for patients in the same batch.
    """

    def __init__(self, config_path: str, split: str):
        """Load a split and prepare its original patient batches."""
        cfg = dict(load_config(config_path))
        cfg["data"] = dict(cfg.get("data") or {})
        cfg["data"]["num_workers"] = 0          # Read patients in this process
        model_cfg = cfg.get("model") or {}
        self.dataset = get_datasets(cfg, splits=(split,))[split]
        self.collate_fn = get_collate_fn(model_cfg.get("collate_style") or model_cfg.get("name"))
        self.batch_size = int(cfg["data"].get("batch_size", 32))
        self.identifier_column = self.dataset.identifier_column
        self.source_identifier_column = self.dataset.source_identifier_column
        self._ids = self.dataset.frame[self.source_identifier_column].to_numpy()
        self._scorers: dict[tuple[int, str], object] = {}

    def index_of(self, identifier: int) -> int | None:
        """Find a patient's first row in the split, if present."""
        hits = np.nonzero(self._ids == int(identifier))[0]
        return int(hits[0]) if hits.size else None

    def first_identifiers(self, count: int = 10) -> list[int]:
        """Return patient identifiers in their saved order."""
        return [int(value) for value in self._ids[:count]]

    def co_batch(self, idx: int) -> tuple[dict, int, int, int]:
        """Return the patient's batch, row range and position within it."""
        start = (idx // self.batch_size) * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))
        batch = self.collate_fn([self.dataset[i] for i in range(start, end)])
        return batch, start, end - 1, idx - start

    def single_row(self, idx: int) -> dict:
        """Prepare one patient as a model batch."""
        return self.collate_fn([self.dataset[idx]])

    def scorer(self, model, batch: dict, start: int, cxr_modality: str = "cxr"):
        """Reuse the scorer for this patient batch and image input."""
        cache_key = (start, cxr_modality)
        cached = self._scorers.get(cache_key)
        if cached is None:
            cached = M3CareFrozenGraphScorer(model, batch, tabular_is_sequential=False)
            self._scorers[cache_key] = cached
        return cached


class _FrozenTabProbModule(torch.nn.Module):
    """Wrap one patient's tabular predictions for Gradient SHAP with patient links fixed."""

    def __init__(self, scorer: M3CareFrozenGraphScorer, target_idx: int):
        super().__init__()
        self.scorer = scorer
        self.target_idx = int(target_idx)
        self._seq = scorer.tabular_is_sequential
        self._max_t = scorer.max_t
        self._lengths = (scorer.tab_lengths[target_idx]
                         if (self._seq and scorer.tab_lengths is not None) else None)

    def forward(self, tab_matrix: torch.Tensor) -> torch.Tensor:
        scorer = self.scorer
        inner = scorer.inner
        tab_matrix = tab_matrix.to(device=scorer.device, dtype=torch.float32)
        n = tab_matrix.shape[0]
        if self._seq:
            tab = tab_matrix.unsqueeze(1).repeat(1, self._max_t, 1)          # (n, T, F)
            lengths = None if self._lengths is None else self._lengths.repeat(n)
            tab_emb = inner.modality_encoder(scorer.tabular_modality)(tab, lengths)  # (n, H)
        else:
            tab_emb = inner.modality_encoder(scorer.tabular_modality)(tab_matrix, None)  # (n, H)
        node, mask = scorer._broadcast_target(
            self.target_idx,
            {scorer.tabular_modality: tab_emb},
        )
        logit = scorer._frozen.forward_target_from_embeddings(
            inner, node, mask, scorer.adj, self.target_idx)                 # (n,)
        return torch.sigmoid(logit).unsqueeze(-1)                           # (n, 1)


def get_shap_values_frozen_graph(
    model: torch.nn.Module,
    test_ds,
    collate_fn: Callable,
    background_batch: Mapping[str, torch.Tensor],
    feature_names: Sequence[str],
    *,
    batch_size: int,
    tabular_is_sequential: bool = False,
    engine: str = "gradient",
    kernel_nsamples: int | str = "auto",
    kernel_background_summary: int = 50,
    gradient_nsamples: int | str = "auto",
    chunk_size: int = 2048,
    n_explain: int = -1,
    shap_seed: int = DEFAULT_SHAP_SEED,
):
    """Explain M3Care tabular features in dataset order while keeping patient links fixed."""
    import shap

    inner = model.model
    inner.eval()
    device = next(inner.parameters()).device
    feature_names = list(feature_names)
    n_features = len(feature_names)
    chosen = _select_engine(engine)

    background, grad_ns = _prepare_background(
        background_batch, tabular_is_sequential, chosen, device=device,
        kernel_background_summary=kernel_background_summary,
        gradient_nsamples=gradient_nsamples,
    )

    N = len(test_ds)
    limit = N if (n_explain is None or n_explain < 0) else min(int(n_explain), N)
    all_values = np.zeros((limit, n_features), dtype=np.float64)
    all_data = np.zeros((limit, n_features), dtype=np.float64)
    base_values = np.zeros(limit, dtype=np.float64)
    ed_ids = np.zeros(limit, dtype=np.int64)

    done = 0
    desc = "M3Care frozen-graph tabular SHAP" + (" (gradient)" if chosen == "gradient" else "")
    progress = tqdm(total=limit, desc=desc, unit="patients")
    for start in range(0, N, batch_size):
        if done >= limit:
            break
        idxs = list(range(start, min(start + batch_size, N)))
        graph_batch = collate_fn([test_ds[int(i)] for i in idxs])
        scorer = M3CareFrozenGraphScorer(model, graph_batch, tabular_is_sequential)
        batch_ed = scorer.identifiers()
        for local_t in range(len(idxs)):
            if done >= limit:
                break
            ref_tab = scorer.reference_tab_matrix(local_t)  # (1, F)
            if chosen == "kernel":
                all_values[done], base_values[done] = _kernel_row(
                    scorer.tabular_predict_fn(local_t, chunk_size=chunk_size),
                    ref_tab, background, kernel_nsamples, int(shap_seed) + done)
            else:
                all_values[done], base_values[done] = _gradient_row(
                    scorer.gradient_module(local_t),
                    torch.from_numpy(ref_tab).to(device=device, dtype=torch.float32),
                    background, grad_ns, int(shap_seed) + done)
            all_data[done] = ref_tab.reshape(-1)
            if batch_ed is not None:
                ed_ids[done] = int(batch_ed[local_t])
            done += 1
            progress.update(1)
    progress.close()

    explanation = shap.Explanation(
        values=all_values, base_values=base_values, data=all_data, feature_names=feature_names,
    )
    return explanation, ed_ids


def compute_mm_shap_frozen_graph(
    model: torch.nn.Module,
    test_ds,
    collate_fn: Callable,
    *,
    batch_size: int,
    modalities: Sequence[str] | None = None,
    n_mmshap: int = -1,
    tabular_is_sequential: bool = False,
):
    """Calculate M3Care input contributions in dataset order with patient links fixed."""
    inner = model.model
    inner.eval()
    modalities = (
        tuple(getattr(inner, "modality_names", MODALITIES))
        if modalities is None
        else tuple(modalities)
    )

    N = len(test_ds)
    limit = N if (n_mmshap is None or n_mmshap < 0) else min(int(n_mmshap), N)
    coalition_rows: dict[frozenset, list[float]] = {
        frozenset(s): [] for s in _all_coalitions(modalities)
    }
    ed_ids: list[int] = []

    done = 0
    progress = tqdm(total=limit, desc="M3Care frozen-graph MM-SHAP", unit="patients")
    for start in range(0, N, batch_size):
        if done >= limit:
            break
        idxs = list(range(start, min(start + batch_size, N)))
        graph_batch = collate_fn([test_ds[int(i)] for i in idxs])
        scorer = M3CareFrozenGraphScorer(
            model,
            graph_batch,
            tabular_is_sequential=tabular_is_sequential,
        )
        batch_ed = scorer.identifiers()
        for local_t in range(len(idxs)):
            if done >= limit:
                break
            cv = scorer.coalition_values(local_t, modalities)
            for key, arr in cv.items():
                coalition_rows[key].append(float(arr[0]))
            if batch_ed is not None:
                ed_ids.append(int(batch_ed[local_t]))
            done += 1
            progress.update(1)
    progress.close()

    coalition_values = {key: np.asarray(vals, dtype=np.float64) for key, vals in coalition_rows.items()}
    phi = shapley_from_coalitions(coalition_values, modalities)
    summary = estimate_mm_summary(phi, modalities)

    v_all = coalition_values[frozenset(modalities)]
    v_none = coalition_values[frozenset()]
    phi_sum = np.sum([phi[m] for m in modalities], axis=0)
    summary["shapley_values"] = {m: phi[m] for m in modalities}
    summary["efficiency_residual"] = float(np.max(np.abs(phi_sum - (v_all - v_none))))
    # Include the prediction made with all inputs present and the same fixed patient links
    summary["v_all"] = v_all
    summary["v_all_mean"] = float(v_all.mean())
    summary["v_none_mean"] = float(v_none.mean())
    return summary, np.asarray(ed_ids, dtype=np.int64)


# --------------------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------------------
def patient_title(
    stay_id: int | None = None,
    label: float | None = None,
    probability: float | None = None,
    contribution_pct: Mapping[str, float] | None = None,
    identifier_label: str = "stay",
    contribution_wrap: int | None = None,
) -> str:
    """Build a patient title with the outcome, prediction and input contributions.

    Show ? for an unknown ID or outcome and omit the probability if it is missing.
    Keep contributions in the supplied order and wrap them if requested.
    """
    gt = "?" if label is None else f"{int(round(float(label)))}"
    title = (
        f"{identifier_label} {stay_id if stay_id is not None else '?'}  "
        f"ground truth: {gt}"
    )
    if probability is not None:
        title += f"  predicted probability: {float(probability):.3f}"
    parts = []
    if contribution_pct is not None:
        for modality, value in contribution_pct.items():
            label = _MMSHAP_LABELS.get(
                modality,
                f"{str(modality).replace('_', ' ').upper()}-SHAP",
            )
            parts.append(f"{label}={value:.1f}%")
    if parts:
        if contribution_wrap is None or contribution_wrap <= 0:
            contribution_lines = ["  ".join(parts)]
        else:
            contribution_lines = [
                "  ".join(parts[index:index + contribution_wrap])
                for index in range(0, len(parts), contribution_wrap)
            ]
        title += "\n" + "\n".join(contribution_lines)
    return title


def get_shap_summary_plot(
    shap_obj,
    outcome: str = "Deterioration",
    fusion_type: str | None = None,
    modality: str | None = "tabular",
    max_features: int = 20,
    figsize: tuple[int, int] = (7, 8),
    save_path: str | Path | None = None,
    heatmap: bool = False,
):
    """Plot feature contributions across patients and save the plot if a path is provided."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    # Rename features for this plot, then restore their names so later plots do not rename them twice
    original_names = shap_obj.feature_names
    display_names = [_display_feature_name(n) for n in original_names]
    plt.figure()
    try:
        shap_obj.feature_names = display_names
        if heatmap:
            # Keep small contributions precise so rounding does not turn them into zeros
            shap.plots.heatmap(shap_obj, max_display=max_features, plot_width=9, show=False)
            plt.title(f"Heatmap view for {modality} modality: {outcome}, {fusion_type}.")
        else:
            shap.plots.beeswarm(shap_obj, plot_size=figsize, max_display=max_features, show=False)
            plt.grid("both", linestyle="--", alpha=0.7)
            plt.title(
                f"SHAP Global Importance for {modality} modality: {outcome}, {fusion_type}."
            )
    finally:
        shap_obj.feature_names = original_names
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary plot for {modality} modality saved to {save_path}.")
    return save_path


def _decision_plot_xlim(
    base_value: float, values, n_display: int, margin: float = 0.02,
) -> tuple[float, float]:
    """Choose axis limits around the displayed prediction line, with a small margin."""
    phi = np.asarray(values, dtype=float).reshape(-1)
    base = float(base_value)
    if phi.size == 0:
        return (0.0, 1.0)
    # Add contributions in the plot's order, from smallest to largest absolute value
    cumulative = base + np.nancumsum(phi[np.argsort(np.abs(phi))])
    # Include the point just before the first displayed feature
    drawn = np.concatenate(([base], cumulative[max(0, phi.size - int(n_display) - 1):]))
    low = float(np.nanmin(drawn))
    high = float(np.nanmax(drawn))
    # Keep a visible margin when all values are equal
    pad = max((high - low) * margin, 1e-3)
    return _snap_outward(min(low, max(0.0, low - pad)), max(high, min(1.0, high + pad)))


# Ticks for the decision plot's x-axis
_NICE_AXIS_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0)


def _snap_outward(low: float, high: float, max_ticks: int = 8) -> tuple[float, float]:
    """Widen the axis limits to round values with roughly max_ticks intervals."""
    span = high - low
    step = next((s for s in _NICE_AXIS_STEPS if span / s <= max_ticks), _NICE_AXIS_STEPS[-1])
    # Round away extra decimal places, such as 0.060000000000000005
    return (round(np.floor(low / step) * step, 10), round(np.ceil(high / step) * step, 10))


# Space axis labels using steps of 1, 2 or 5 to avoid unnecessary decimal places
_AXIS_LOCATOR_BINS = 6
_AXIS_LOCATOR_STEPS = (1, 2, 5, 10)


def get_shap_local_decision_plot(
    shap_obj,
    figsize: tuple[int, int] = (6, 7),
    save_tab_path: str | Path | None = None,
    shap_range=None,
    mm_scores=None,
    stay_id: int | None = None,
    label: float | None = None,
    probability: float | None = None,
    original_values: Mapping[str, float] | None = None,
    imputed: Mapping[str, bool] | None = None,
):
    """Plot how tabular features contribute to one patient's prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import shap

    # Put scores into a dictionary for the plot title
    score_map: dict[str, float] | None = None
    if mm_scores is not None:
        raw = (dict(mm_scores) if isinstance(mm_scores, Mapping)
               else {m: mm_scores[i] for i, m in enumerate(MODALITIES) if i < len(mm_scores)})
        score_map = {m: v for m, v in raw.items() if v is not None}

    # Keep the starting value precise, since rounding it would shift the whole prediction line
    base_values = np.asarray(shap_obj.base_values).reshape(-1)[0]

    # Show yes/no values and original measurements where available
    # Allow both text and numbers in the displayed values
    raw_names = list(shap_obj.feature_names)
    feature_names = [_display_feature_name(n) for n in raw_names]
    raw_data = np.asarray(shap_obj.data).reshape(-1)
    display_data = np.empty(raw_data.shape, dtype=object)
    for j, name in enumerate(feature_names):
        value = raw_data[j]
        text = str(name)
        raw_name = raw_names[j]
        on = np.isclose(value, 1.0)
        off = np.isclose(value, 0.0)
        if _is_onehot_indicator(raw_name):  # The name already includes the category
            display_data[j] = "Yes" if on else "No" if off else np.round(value, 2)
        elif text == "demographics_gender" and (on or off):
            display_data[j] = "Male" if on else "Female"
        elif text.startswith("demographics_") and (on or off):  # Combined yes/no category
            display_data[j] = "Yes" if on else "No"
        elif imputed is not None and imputed.get(raw_name):
            display_data[j] = "imputed"
        elif original_values is not None and raw_name in original_values:
            display_data[j] = _format_original_value(original_values[raw_name])
        else:  # Use the model input when the original value is unavailable
            display_data[j] = np.round(value, 2)

    n_display = max(2, min(16, len(feature_names)))

    # Choose axis limits around the displayed probabilities
    if shap_range is None:
        shap_range = _decision_plot_xlim(base_values, shap_obj.values, n_display)

    plt.figure(figsize=figsize)
    shap.plots.decision(
        base_values,
        shap_obj.values, display_data, feature_names,
        feature_display_range=slice(None, -n_display, -1),
        highlight=0,
        show=False,
        xlim=shap_range,
    )
    # Use the label spacing defined in _AXIS_LOCATOR_STEPS
    plt.gca().xaxis.set_major_locator(
        MaxNLocator(nbins=_AXIS_LOCATOR_BINS, steps=list(_AXIS_LOCATOR_STEPS))
    )
    plt.title(patient_title(stay_id, label, probability, score_map))
    plt.tight_layout()
    if save_tab_path is not None:
        plt.savefig(save_tab_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP local-level decision plot saved to {save_tab_path}.")
    return save_tab_path


# Chest X-ray availability groups in bar order
# cxr_missing is 0 when an image is present
_CXR_GROUPS: tuple[tuple[int, str], ...] = ((0, "CXR available"), (1, "CXR missing"))


def contribution_by_cxr_availability(
    mm_summary: Mapping[str, Any],
    cxr_missing: Sequence[int] | np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[tuple[int, str], dict[str, float]]:
    """Summarise input contributions for patients with and without chest X-rays.

    Return mean percentages, confidence bounds and patient counts for each group
    and input type. Match the availability flags to the order of the explained patients.
    """
    from src.evaluation.eval_utils import bias_corrected_ci

    modalities = tuple(mm_summary["modalities"])
    per_sample = np.asarray(mm_summary["per_sample_contribution"], dtype=np.float64) * 100.0
    flags = np.asarray(cxr_missing).reshape(-1)
    if len(flags) != per_sample.shape[0]:
        raise ValueError(
            f"cxr_missing has {len(flags)} entries but MM-SHAP explained {per_sample.shape[0]} "
            "patients. The flags must be in the same row order as per_sample_contribution."
        )

    rng = np.random.default_rng(seed)
    results: dict[tuple[int, str], dict[str, float]] = {}
    for flag, label in _CXR_GROUPS:
        values = per_sample[flags == flag]
        n = len(values)
        if n == 0:
            raise ValueError(f"No explained patients in the '{label}' group (cxr_missing == {flag}).")
        # Use the same randomly selected patients for every input type in this group
        boot = np.stack([values[rng.integers(0, n, size=n)].mean(axis=0) for _ in range(n_boot)])
        point = values.mean(axis=0)
        for j, modality in enumerate(modalities):
            samples = boot[:, j]
            if np.ptp(samples) == 0.0:  # All sample values are equal
                lower, upper = float(point[j]), float(point[j])
            else:
                lower, upper = bias_corrected_ci(samples, point[j])
            results[(flag, modality)] = {
                "point": float(point[j]), "ci_lower": lower, "ci_upper": upper, "n": n,
            }
    return results


def plot_modality_contribution(
    mm_summary: Mapping[str, Any],
    save_path: str | Path,
    cxr_missing: Sequence[int] | np.ndarray,
    fusion_type: str | None = None,
    palette: Sequence[str] = ("#6D2E8E", "#0E7C84", "#BE7C1C", "#3B4C61"),
    n_boot: int = 1000,
    seed: int = 42,
):
    """Compare input contributions for patients with and without chest X-rays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    modalities = list(mm_summary["modalities"])
    stats = contribution_by_cxr_availability(mm_summary, cxr_missing, n_boot=n_boot, seed=seed)
    x = np.arange(len(modalities))
    width = 0.38
    top = max(s["ci_upper"] for s in stats.values())

    fig, ax = plt.subplots(figsize=(6, 3), dpi=600)
    # Use a transparent background so the plot fits onto report slides
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    for mi, modality in enumerate(modalities):
        for gi, (flag, _) in enumerate(_CXR_GROUPS):
            stat = stats[(flag, modality)]
            point = stat["point"]
            lo = point - stat["ci_lower"]
            hi = stat["ci_upper"] - point
            ax.bar(x[mi] + (gi - 0.5) * width, point, width,
                   color=palette[mi % len(palette)],
                   alpha=1.0 if flag else 0.55,
                   hatch="" if flag else "//",
                   edgecolor="white", linewidth=0.5,
                   yerr=[[lo], [hi]], capsize=2,
                   error_kw={"ecolor": "lightgray", "elinewidth": 1})
            ax.text(x[mi] + (gi - 0.5) * width, point + hi + top * 0.02, f"{point:.1f}%",
                    ha="center", va="bottom", color="#404040", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("MM-SHAP contribution (%)")
    ax.set_ylim(0, top * 1.25)
    ax.set_title(f"Modality contribution: {fusion_type}")
    ax.spines[["top", "right"]].set_visible(False)

    # The legend shows availability groups and colours show input types
    handles = [
        Patch(facecolor="0.55", alpha=0.55, hatch="//", edgecolor="white",
              label=f"{_CXR_GROUPS[0][1]} (n={stats[(_CXR_GROUPS[0][0], modalities[0])]['n']})"),
        Patch(facecolor="0.55", edgecolor="white",
              label=f"{_CXR_GROUPS[1][1]} (n={stats[(_CXR_GROUPS[1][0], modalities[0])]['n']})"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right", frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=600, transparent=True)
    plt.close(fig)
    return save_path


def plot_global_modality_contribution(
    mm_summary: Mapping[str, Any],
    save_path: str | Path,
    fusion_type: str | None = None,
    palette: Sequence[str] = ("#6D2E8E", "#0E7C84", "#BE7C1C", "#3B4C61"),
    n_boot: int = 1000,
    seed: int = 42,
):
    """Plot average input contributions across all explained patients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modalities = list(mm_summary["modalities"])
    values = np.asarray(mm_summary["per_sample_contribution"], dtype=np.float64) * 100.0
    n = len(values)
    point = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.stack([values[rng.integers(0, n, size=n)].mean(axis=0) for _ in range(n_boot)])

    from src.evaluation.eval_utils import bias_corrected_ci

    intervals = [
        (point[j], *bias_corrected_ci(boot[:, j], point[j]))
        if np.ptp(boot[:, j]) else (point[j], point[j], point[j])
        for j in range(len(modalities))
    ]
    top = max(upper for _, _, upper in intervals)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=600)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    x = np.arange(len(modalities))
    for i, (value, lower, upper) in enumerate(intervals):
        ax.bar(x[i], value, color=palette[i % len(palette)],
               edgecolor="white", linewidth=0.5,
               yerr=[[value - lower], [upper - value]], capsize=2,
               error_kw={"ecolor": "lightgray", "elinewidth": 1})
        ax.text(x[i], upper + top * 0.02, f"{value:.1f}%",
                ha="center", va="bottom", color="#404040", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(modalities)
    ax.set_ylabel("MM-SHAP contribution (%)")
    ax.set_ylim(0, top * 1.25)
    ax.set_title(f"Modality contribution: {fusion_type}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=600, transparent=True)
    plt.close(fig)
    return save_path
