"""Create chest X-ray heatmaps for deterioration and hospitalisation models.

Rebuild the image features used by a saved model, then highlight regions that
affect its prediction. Keep other images from the same patient in the score.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.collate import modality_available
from src.data.splits import find_split_summary, resolve_split_dir
from src.evaluation.eval_utils import load_config
from src.explanation.shap_utils import SplitBatches, patient_title

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
# Match the image directory used during feature extraction
DEFAULT_CXR_ROOT = str(WORKSPACE_ROOT / "data" / "MIMIC-IV" / "mimic-cxr-jpg")

# Fit image feature scaling from the training data
CXR_NORMALISATION = "fit"


@dataclass
class CXRGradCAMConfig:
    """Settings for the saved model, image files and feature scaling.

    checkpoint_path identifies the model, keys_csv links patients to images, and
    cxr_root locates them. raw_cxr_pkl supplies training features when normalise
    is "fit". train_folds or train_identifiers select training patients, and
    cxr_modality selects the model input to explain.
    """

    checkpoint_path: str
    cxr_root: str
    keys_csv: str
    config_path: str | None = None
    raw_cxr_pkl: str | None = None
    normalise: Any = "fit"
    train_folds: Sequence[int] = tuple(range(18))
    densenet_weights: str = "densenet121-res224-chex"
    device: str = "cpu"
    cxr_modality: str = "cxr"
    identifier_column: str = "ed_stay_id"
    source_identifier_column: str = "ed_stay_id"
    paths_column: str = "cxr_paths"
    train_identifiers: Sequence[int] | None = None


@dataclass
class CAMResult:
    """Store one image, its prediction and both heatmap resolutions."""

    ed_stay_id: int
    dicom_id: str
    image_path: str
    logit: float
    prob: float
    cam: np.ndarray          # (H, W) float in [0, 1], upsampled to image size
    cam_lowres: np.ndarray   # (7, 7) raw CAM before upsampling
    original: np.ndarray     # (H, W) display image (xrv-normalised, for overlay)
    label: float | None = None  # ground-truth outcome for the stay, if available


# Prepare images as extract_cxr.py does
def load_and_preprocess_image(path: str, out_size: int = 224) -> np.ndarray:
    """Read an X-ray and return a float32 array shaped (1, 1, size, size).

    Use the same greyscale conversion, scaling and resizing as extract_cxr.py.
    """
    import cv2
    import skimage.io
    import torchxrayvision as xrv

    img = skimage.io.imread(path)
    # Convert colour images to greyscale
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = xrv.datasets.normalize(img, 255)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img[None, None, :, :].astype(np.float32)


# Calculate image feature scaling from training data
def fit_cxr_norm_stats(
    raw_cxr_pkl: str,
    keys_csv: str,
    train_folds: Sequence[int],
    *,
    train_identifiers: Sequence[int] | None = None,
    source_identifier_column: str = "ed_stay_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Find each chest X-ray feature’s minimum and maximum in training data.

    Use train_identifiers when supplied, otherwise select rows by train_folds.
    Ignore missing values and preserve the saved feature order.
    """

    embeddings = pd.read_pickle(raw_cxr_pkl)  # Saved sequences of image features
    keys = pd.read_csv(keys_csv)
    if train_identifiers is not None:
        if source_identifier_column not in keys:
            raise ValueError(
                f"keys.csv lacks {source_identifier_column!r}; cannot select train rows."
            )
        train_mask = keys[source_identifier_column].isin(train_identifiers).to_numpy()
    else:
        if "strat_fold" not in keys.columns:
            raise ValueError("keys.csv lacks 'strat_fold'; cannot select train folds.")
        train_mask = keys["strat_fold"].isin(list(train_folds)).to_numpy()
    row_indices = np.nonzero(train_mask)[0]

    min_values: np.ndarray | None = None
    max_values: np.ndarray | None = None
    for idx in row_indices:
        seq = embeddings[idx]
        arr = np.asarray(seq, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feat_min = np.nanmin(arr, axis=0)
            feat_max = np.nanmax(arr, axis=0)
        if min_values is None:
            min_values, max_values = feat_min, feat_max
        else:
            min_values = np.fmin(min_values, feat_min)
            max_values = np.fmax(max_values, feat_max)
    if min_values is None:
        raise ValueError("No train-fold CXR embeddings found to fit stats.")
    return min_values.astype(np.float32), max_values.astype(np.float32)


def sample_cxr_background(
    raw_cxr_pkl: str,
    keys_csv: str,
    train_folds: Sequence[int],
    n_samples: int = 50,
    seed: int = 0,
    *,
    train_identifiers: Sequence[int] | None = None,
    source_identifier_column: str = "ed_stay_id",
) -> np.ndarray:
    """Choose average chest X-ray features from training patients for SHAP.

    Use seed for repeatable order and skip empty or invalid rows.
    """

    embeddings = pd.read_pickle(raw_cxr_pkl)
    keys = pd.read_csv(keys_csv)
    if train_identifiers is not None:
        if source_identifier_column not in keys:
            raise ValueError(
                f"keys.csv lacks {source_identifier_column!r}; cannot select train rows."
            )
        train_mask = keys[source_identifier_column].isin(train_identifiers).to_numpy()
    else:
        if "strat_fold" not in keys.columns:
            raise ValueError("keys.csv lacks 'strat_fold'; cannot select train folds.")
        train_mask = keys["strat_fold"].isin(list(train_folds)).to_numpy()
    row_indices = np.nonzero(train_mask)[0]

    means: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(row_indices):
        arr = np.asarray(embeddings[idx], dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row = np.nanmean(arr, axis=0)
        if np.isnan(row).any():
            continue
        means.append(row.astype(np.float32))
        if len(means) >= n_samples:
            break
    if not means:
        raise ValueError("No train-fold CXR embeddings available for a SHAP background.")
    return np.stack(means, axis=0)


def apply_cxr_norm(
    emb: torch.Tensor,
    min_values: np.ndarray,
    max_values: np.ndarray,
) -> torch.Tensor:
    """Scale chest X-ray features with the training minimum and maximum."""
    rng = (max_values - min_values).astype(np.float32)
    rng[rng == 0] = 1.0
    tmin = torch.as_tensor(min_values, dtype=emb.dtype, device=emb.device)
    trng = torch.as_tensor(rng, dtype=emb.dtype, device=emb.device)
    return (emb - tmin) / trng


def bundle_files_from_config(
    config_path: str, cxr_modality: str = "cxr",
) -> tuple[str | None, str | None]:
    """Find raw image features and patient keys for the run.

    Use paths in the split summary when present. Otherwise build both paths
    from the folder above the split directory.
    """
    if not Path(config_path).exists():
        return None, None
    split_dir = resolve_split_dir(load_config(config_path))
    if split_dir is None:
        return None, None
    summary_path = find_split_summary(split_dir)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        raw_pkl = summary.get(f"input_{cxr_modality}_path")
        keys_csv = summary.get("input_keys_path")
        if raw_pkl and keys_csv:
            return str(raw_pkl), str(keys_csv)

    parent = split_dir.parent
    prefix = parent.name
    suffix = (
        "cxr_embeddings"
        if cxr_modality == "cxr"
        else f"{cxr_modality}_embeddings"
    )
    raw_pkl = parent / f"{prefix}_{suffix}.pkl"
    keys_csv = parent / f"{prefix}_keys.csv"
    return str(raw_pkl), str(keys_csv)


def cxr_modalities(batches: SplitBatches | None) -> list[str]:
    """List the run's chest X-ray inputs, or the single default input without a split."""
    if batches is None:
        return ["cxr"]
    return [
        modality for modality in batches.dataset.modality_names
        if modality == "cxr" or modality.startswith("cxr_")
    ]


def cxr_available(batches: SplitBatches, idx: int, modality: str) -> bool:
    """Check whether this patient has the requested chest X-ray input."""
    return bool(modality_available(batches.single_row(idx), modality, batch_size=1)[0])


def substitute_live_cxr(
    batch: dict, row: int, stay_embedding, cxr_modality: str = "cxr",
) -> None:
    """Replace one saved image feature row with features from the image."""
    feats_key = f"{cxr_modality}_feats"
    lengths_key = f"{cxr_modality}_lengths"
    if lengths_key in batch and int(batch[lengths_key][row]) > 1:
        raise ValueError(
            f"{cxr_modality} has multiple model timesteps; Grad-CAM currently supports "
            "single-record CXR inputs only."
        )
    if batch[feats_key].shape[1] == 0:
        return
    target = batch[feats_key]
    batch[feats_key][row, 0, :] = stay_embedding.to(target.device, target.dtype)
    missing_key = f"{cxr_modality}_missing"
    if lengths_key in batch:
        batch[lengths_key][row] = 1
    if missing_key in batch:
        batch[missing_key][row] = 0


def train_identifiers(config_path: str, task: str) -> list[int] | None:
    """Return the training patients used to scale image features.

    Hospitalisation selects them from the train split. Deterioration returns
    None so scaling selects training rows by fold.
    """
    if task != "hospitalisation":
        return None
    train_split = SplitBatches(config_path, "train")
    return train_split.first_identifiers(len(train_split.dataset))


def gradcam_config(
    *,
    checkpoint_path: str,
    config_path: str,
    cxr_root: str,
    cxr_modality: str,
    batches: SplitBatches | None,
    train_identifiers: list[int] | None,
    device: str,
    raw_pkl: str | None = None,
    keys_csv: str | None = None,
) -> CXRGradCAMConfig:
    """Build the heatmap settings for one chest X-ray input of a run.

    Find raw features and patient keys from the run configuration unless
    raw_pkl or keys_csv are given.
    """
    derived_raw_pkl, derived_keys_csv = bundle_files_from_config(config_path, cxr_modality)
    keys_csv = keys_csv or derived_keys_csv
    if keys_csv is None:
        raise SystemExit("Could not derive keys CSV; pass --keys-csv explicitly.")
    return CXRGradCAMConfig(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        cxr_root=cxr_root,
        keys_csv=keys_csv,
        raw_cxr_pkl=raw_pkl or derived_raw_pkl,
        normalise=CXR_NORMALISATION,
        device=device,
        cxr_modality=cxr_modality,
        identifier_column=(
            batches.identifier_column if batches is not None else "ed_stay_id"
        ),
        source_identifier_column=(
            batches.source_identifier_column if batches is not None else "ed_stay_id"
        ),
        paths_column=f"{cxr_modality}_paths",
        train_identifiers=train_identifiers,
    )


class CXRGradCAM:
    """Build chest X-ray heatmaps for a saved model.

    Use the configured model, image paths and patient context to explain
    deterioration or hospitalisation predictions.
    """

    def __init__(self, config: CXRGradCAMConfig) -> None:
        """Load the model, image encoder, image links and scaling values."""
        self.config = config
        self.device = torch.device(config.device)

        # Use saved paths as they are
        self._cxr_root = str(config.cxr_root)
        self._raw_cxr_pkl = (
            str(config.raw_cxr_pkl) if config.raw_cxr_pkl is not None else None
        )
        self._keys_csv = str(config.keys_csv)

        # Load the trained model
        from src.training.lightning import MMModel

        ckpt = config.checkpoint_path
        self.mm = MMModel.load_from_checkpoint(
            str(ckpt), map_location=str(self.device)
        )
        self.mm.eval().to(self.device)
        self.model = self.mm.model
        self.fusion_method = getattr(self.mm, "fusion_method", "unknown")
        self.feature_dims = dict(getattr(self.mm, "feature_dims", {}))
        self.cxr_modality = config.cxr_modality
        self.identifier_column = config.identifier_column
        self.source_identifier_column = config.source_identifier_column
        self.paths_column = config.paths_column
        self.cxr_dim = int(self.feature_dims.get(self.cxr_modality, 1024))

        self.config_path = (
            config.config_path
            if config.config_path is not None
            else str(Path(ckpt).parent / "config.yaml")
        )

        # Read the outcome for plot titles when the config is available
        self.outcome: str | None = None
        try:
            cfg = dict(load_config(self.config_path))
            self.outcome = str(
                cfg.get("data", {}).get("outcome", cfg.get("outcome", "deterioration"))
            )
        except Exception:
            self.outcome = None

        # Load the image encoder
        import torchxrayvision as xrv

        self.densenet = (
            xrv.models.DenseNet(weights=config.densenet_weights).eval().to(self.device)
        )
        for p in self.densenet.parameters():
            p.requires_grad_(False)

        # Scale image features as training did
        self.min_values: np.ndarray | None = None
        self.max_values: np.ndarray | None = None
        if config.normalise == "fit":
            if self._raw_cxr_pkl is None:
                raise ValueError("normalise='fit' requires raw_cxr_pkl.")
            self.min_values, self.max_values = fit_cxr_norm_stats(
                self._raw_cxr_pkl,
                self._keys_csv,
                config.train_folds,
                train_identifiers=config.train_identifiers,
                source_identifier_column=self.source_identifier_column,
            )
        elif isinstance(config.normalise, (tuple, list)) and len(config.normalise) == 2:
            self.min_values = np.asarray(config.normalise[0], dtype=np.float32)
            self.max_values = np.asarray(config.normalise[1], dtype=np.float32)
        elif config.normalise not in (None, "none"):
            raise ValueError(f"Unrecognised normalise={config.normalise!r}.")

        # Link patients to image paths

        self.keys = pd.read_csv(self._keys_csv)

        # Load patient context only when requested
        self._context_frame: pd.DataFrame | None = None

        # Reuse training image samples for repeated SHAP calls
        self._background_cache: dict[tuple[int, int], torch.Tensor] = {}

    def cxr_background(self, n_samples: int = 50, seed: int = 0) -> torch.Tensor:
        """Return training image features in model input units for SHAP.

        Cache each sample size and seed so repeated calls use the same reference.
        """
        if self._raw_cxr_pkl is None:
            raise ValueError("A SHAP background needs raw_cxr_pkl; none was configured.")
        key = (int(n_samples), int(seed))
        cached = self._background_cache.get(key)
        if cached is not None:
            return cached
        raw = sample_cxr_background(
            self._raw_cxr_pkl, self._keys_csv, self.config.train_folds,
            n_samples=int(n_samples), seed=int(seed),
            train_identifiers=self.config.train_identifiers,
            source_identifier_column=self.source_identifier_column,
        )
        bg = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        if self.min_values is not None:
            bg = apply_cxr_norm(bg, self.min_values, self.max_values)
        self._background_cache[key] = bg
        return bg

    @property
    def raw_cxr_pkl(self) -> str | None:
        """Return the resolved path to raw chest X-ray features, if configured."""
        return self._raw_cxr_pkl

    def attach_context_parquet(self, parquet_path: str) -> None:
        """Load patient context from a split parquet file."""

        self._context_frame = pd.read_parquet(parquet_path)

    def context_from_parquet(self, identifier: int) -> dict | None:
        """Return the patient’s label and available tabular and ECG data.

        Keep the saved tabular missingness mask so scoring knows which values were filled in.
        """
        if self._context_frame is None:
            return None
        rows = self._context_frame[
            self._context_frame[self.source_identifier_column] == identifier
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        ctx: dict[str, Any] = {self.identifier_column: int(identifier)}
        # Use the configured outcome, or the first outcome column if it is missing
        label_col = self.outcome if self.outcome in self._context_frame.columns else None
        if label_col is None:
            outcome_cols = [
                column
                for column in self._context_frame.columns
                if column.startswith(("deterioration_", "hospitalisation_"))
            ]
            label_col = outcome_cols[0] if outcome_cols else None
        if label_col is not None:
            val = row[label_col]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ctx["label"] = float(val)
        if "tab_feats" in row:
            ctx["tab_feats"] = np.asarray(row["tab_feats"], dtype=np.float32).reshape(-1)
            # Keep the saved mask so filled-in values are not treated as observed
            if "tab_missing" in row:
                ctx["tab_missing"] = np.asarray(row["tab_missing"], dtype=np.int64).reshape(-1)
        ecg = row.get("ecg_feats") if hasattr(row, "get") else None
        if ecg is not None and len(ecg):
            arr = np.stack(list(ecg)).astype(np.float32)
            ctx["ecg_feats"] = arr
        return ctx

    # Rebuild image features with gradients
    def _densenet_spatial(self, img: torch.Tensor) -> torch.Tensor:
        """Return the image encoder features used to build the heatmap."""
        feats = self.densenet.features(img)
        return F.relu(feats, inplace=False)

    def raw_gap_from_image(self, path: str, keep_graph: bool = False) -> torch.Tensor:
        """Read one image and average its spatial features before scaling."""
        arr = load_and_preprocess_image(path)
        img = torch.from_numpy(arr).to(self.device)
        ctx = torch.enable_grad() if keep_graph else torch.no_grad()
        with ctx:
            spatial = self._densenet_spatial(img)                        # (1,1024,7,7)
            gap = F.adaptive_avg_pool2d(spatial, (1, 1)).reshape(1, -1)   # (1,1024)
        return gap.squeeze(0)

    def embedding_from_image(self, path: str, keep_graph: bool = False) -> torch.Tensor:
        """Return one image’s features in model input units.

        This matches the saved patient feature only when that patient has one image.
        """
        gap = self.raw_gap_from_image(path, keep_graph=keep_graph)[None, :]
        if self.min_values is not None:
            gap = apply_cxr_norm(gap, self.min_values, self.max_values)
        return gap.squeeze(0)

    def stay_embedding(self, paths: Sequence[str]) -> torch.Tensor:
        """Average a patient’s images, then scale the result for the model."""
        with torch.no_grad():
            raws = torch.stack([self.raw_gap_from_image(p) for p in paths], dim=0)
            mean_raw = raws.mean(dim=0, keepdim=True)                     # (1,1024)
            if self.min_values is not None:
                mean_raw = apply_cxr_norm(mean_raw, self.min_values, self.max_values)
        return mean_raw.squeeze(0)

    # Build a patient batch
    def _base_batch_from_row(
        self,
        cxr_feats: torch.Tensor,
        context_row: dict | None,
    ) -> dict[str, torch.Tensor]:
        """Build a one-patient model batch from image features and available context.

        Mark missing tabular and ECG inputs as absent when context is unavailable.
        """
        tab_dim = int(self.feature_dims.get("tabular", 1))
        ecg_dim = int(self.feature_dims.get("ecg", 1))
        T = cxr_feats.shape[0]
        dev = self.device

        def _seq(row_key: str, dim: int) -> tuple[torch.Tensor, int, int]:
            """Return one sequence with its length and missingness flag."""
            if context_row is not None and row_key in context_row:
                a = np.asarray(context_row[row_key], dtype=np.float32)
                if a.ndim == 1:
                    a = a.reshape(-1, dim) if a.size else a.reshape(0, dim)
                if a.size == 0:
                    return torch.zeros(1, dim, device=dev), 0, 1
                t = torch.as_tensor(a, dtype=torch.float32, device=dev)
                return t, int(t.shape[0]), 0
            return torch.zeros(1, dim, device=dev), 0, 1

        ecg_feats, ecg_len, ecg_missing = _seq("ecg_feats", ecg_dim)

        # Use the patient's tabular values when available
        if context_row is not None and "tab_feats" in context_row:
            tab = np.asarray(context_row["tab_feats"], dtype=np.float32).reshape(-1)
            tab_feats = torch.as_tensor(tab, dtype=torch.float32, device=dev)[None, :]
            # Use the saved missingness mask when available
            miss = None if context_row is None else context_row.get("tab_missing")
            miss = None if miss is None else np.asarray(miss, dtype=np.int64).reshape(-1)
            if miss is not None and miss.size == tab.size:
                tab_missing = torch.as_tensor(miss, dtype=torch.int64, device=dev)[None, :]
            else:
                tab_missing = torch.zeros_like(tab_feats, dtype=torch.int64)
        else:
            tab_feats = torch.zeros(1, tab_dim, device=dev)
            tab_missing = torch.ones(1, tab_dim, dtype=torch.int64, device=dev)

        return {
            "ed_stay_id": torch.tensor([0], dtype=torch.int64, device=dev),
            # Keep labels flat so missingness masks index the right patient
            "label": torch.zeros(1, dtype=torch.float32, device=dev),
            "tab_feats": tab_feats,
            "tab_missing": tab_missing,
            "ecg_feats": ecg_feats[None, ...],
            "ecg_time": torch.zeros(1, ecg_feats.shape[0], device=dev),
            "ecg_lengths": torch.tensor([ecg_len], dtype=torch.int64, device=dev),
            "ecg_missing": torch.tensor([ecg_missing], dtype=torch.int64, device=dev),
            "cxr_feats": cxr_feats[None, ...],            # Keep gradients linked to the image
            "cxr_time": torch.zeros(1, T, device=dev),
            "cxr_lengths": torch.tensor([T], dtype=torch.int64, device=dev),
            "cxr_missing": torch.tensor([0], dtype=torch.int64, device=dev),
        }

    # Calculate one image heatmap
    def _cam_from_spatial(
        self,
        spatial: torch.Tensor,
        logit: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the score gradient to make a small heatmap from image features."""
        grads = torch.autograd.grad(logit, spatial, retain_graph=False)[0]  # (1,1024,7,7)
        weights = grads.mean(dim=(2, 3), keepdim=True)                       # (1,1024,1,1)
        cam = F.relu((weights * spatial).sum(dim=1, keepdim=True))           # (1,1,7,7)
        cam_low = cam.squeeze().detach().cpu().numpy()
        cam_n = cam_low - cam_low.min()
        denom = cam_n.max()
        if denom > 0:
            cam_n = cam_n / denom
        return cam_n, cam_low

    def heatmap_for_image(
        self,
        image_path: str,
        context_row: dict | None = None,
        other_raw_embeddings: torch.Tensor | None = None,
        n_images: int = 1,
        logit_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> CAMResult:
        """Explain one image while accounting for the patient’s other images.

        Average image features before scaling. Keep the selected image linked to
        the score calculation. Use logit_fn to score with a chosen model context.
        """
        # Keep the image linked to the score so its contribution can be measured
        arr = load_and_preprocess_image(image_path)
        img = torch.from_numpy(arr).to(self.device).requires_grad_(True)
        with torch.enable_grad():
            spatial = self._densenet_spatial(img)      # (1,1024,7,7), attached
        spatial.retain_grad()
        raw_target = F.adaptive_avg_pool2d(spatial, (1, 1)).reshape(1, -1)  # (1,1024)

        # Average a patient's images before scaling
        if other_raw_embeddings is not None and other_raw_embeddings.shape[0] > 0:
            others = other_raw_embeddings.to(self.device).detach()          # (T-1,1024)
            T = others.shape[0] + 1
            mean_raw = (raw_target + others.sum(dim=0, keepdim=True)) / T    # (1,1024)
        else:
            T = 1
            mean_raw = raw_target
        assert n_images == T, (
            f"n_images={n_images} but reconstructed T={T} from other_raw_embeddings"
        )

        gap = mean_raw
        if self.min_values is not None:
            gap = apply_cxr_norm(gap, self.min_values, self.max_values)
        cxr_feats = gap                                  # (1,1024) -- one row per stay

        if logit_fn is None:
            batch = self._base_batch_from_row(cxr_feats, context_row)
            logit = self.mm(batch)                       # (1,) scalar logit
        else:
            logit = logit_fn(cxr_feats)
        logit = logit.reshape(-1)[0]
        prob = torch.sigmoid(logit).item()

        cam_n, cam_low = self._cam_from_spatial(spatial, logit)

        cam_t = torch.as_tensor(cam_n, dtype=torch.float32)[None, None]
        H = W = arr.shape[-1]
        cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze().numpy()

        return CAMResult(
            ed_stay_id=(
                int(context_row.get(self.identifier_column, 0))
                if context_row else 0
            ),
            dicom_id=Path(image_path).stem,
            image_path=image_path,
            logit=float(logit.item()),
            prob=float(prob),
            cam=cam_up,
            cam_lowres=cam_low,
            original=arr.squeeze(),
            label=(
                float(context_row["label"])
                if context_row is not None and context_row.get("label") is not None
                else None
            ),
        )

    # Explain each image linked to a patient
    def resolve_stay_images(self, identifier: int) -> list[str]:
        """Find this patient’s image paths in the saved key table."""
        rows = self.keys[
            self.keys[self.source_identifier_column] == identifier
        ]
        if rows.empty:
            raise KeyError(
                f"{self.identifier_column} {identifier} not in keys.csv."
            )
        row = rows.iloc[0]
        raw = row.get(self.paths_column)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        rels = [p.strip() for p in str(raw).split("|") if p.strip()]
        return [str(Path(self._cxr_root) / r) for r in rels]

    def heatmap_for_stay(
        self,
        ed_stay_id: int,
        context_row: dict | None = None,
        logit_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> list[CAMResult]:
        """Return one heatmap for each image linked to a patient.

        Use saved patient context when none is supplied. Pass logit_fn to score
        with a chosen context, including M3Care’s fixed patient graph.
        """
        if context_row is None:
            context_row = self.context_from_parquet(ed_stay_id)
        paths = self.resolve_stay_images(ed_stay_id)
        if not paths:
            return []
        # Read image features once before making individual heatmaps
        with torch.no_grad():
            raws = [self.raw_gap_from_image(p) for p in paths]
        results: list[CAMResult] = []
        for i, path in enumerate(paths):
            others = torch.stack([e for j, e in enumerate(raws) if j != i], dim=0) \
                if len(raws) > 1 else None
            res = self.heatmap_for_image(
                path, context_row=context_row,
                other_raw_embeddings=others, n_images=len(paths),
                logit_fn=logit_fn,
            )
            res.ed_stay_id = int(ed_stay_id)
            results.append(res)
        return results

    # Draw and save a heatmap over the image
    def overlay(self, result: CAMResult, out_path: str, alpha: float = 0.4,
                modality_contribution: dict | None = None) -> str:
        """Save an image with its heatmap and return the output path.

        Show the prediction and label in the title. Add input contribution
        percentages when supplied, in their supplied order.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = result.original.astype(np.float32)
        base = (base - base.min()) / (np.ptp(base) + 1e-8)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(base, cmap="gray")
        ax.imshow(result.cam, cmap="jet", alpha=alpha)
        # Show missing contribution values as nan% so incomplete results are visible
        pct = (
            None
            if modality_contribution is None
            else {
                modality: value * 100
                for modality, value in modality_contribution.items()
            }
        )
        ax.set_title(
            patient_title(
                result.ed_stay_id,
                result.label,
                result.prob,
                pct,
                identifier_label=self.identifier_column.removesuffix("_id"),
                contribution_wrap=(
                    4 if self.identifier_column == "hadm_id" else None
                ),
            ),
            fontsize=9,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path
