"""Explain chest X-ray contributions and highlight image regions.

Measure each input type's contribution and use SHAP values from chest X-ray
features to weight image regions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.eval_utils import _move_batch_to_device
from src.explanation import freeze_m3care
from src.explanation.gradcam import CAMResult, CXRGradCAM, substitute_live_cxr
from src.explanation.shap_utils import (
    DEFAULT_SHAP_SEED,
    MODALITIES,
    M3CareFrozenGraphScorer,
    SplitBatches,
    _all_coalitions,
    _mask_batch_to_coalition,
    _select_engine,
    estimate_mm_summary,
    predict_probability,
    shapley_from_coalitions,
)

# Keep unavailable inputs missing when scoring different input combinations


@dataclass
class ModalityShapley:
    """Store input contributions and the chest X-ray share."""

    values: dict[str, float]  # Contribution by input type, in probability units
    base_value: float  # Probability with no inputs
    full_value: float  # Probability with all inputs
    cxr_fraction: float  # Chest X-ray share of total absolute contribution


def modality_shapley(
    g: CXRGradCAM,
    batch: dict,
    *,
    modalities: tuple[str, ...] | list[str] | None = None,
    target_modality: str = "cxr",
) -> ModalityShapley:
    """Measure how each input type changes the predicted probability.

    Compare predictions across every input combination. The chest X-ray share is
    its absolute contribution divided by the total absolute contribution.
    """
    g.mm.eval()
    modalities = tuple(modalities or MODALITIES)

    def value(coal: tuple[str, ...]) -> float:
        b = _mask_batch_to_coalition(batch, coal)  # Keep only these inputs
        return float(predict_probability(g.mm, b).reshape(-1)[0])

    coalitions = {
        frozenset(subset): np.array([value(subset)], dtype=np.float64)
        for subset in _all_coalitions(modalities)
    }
    phi = {
        modality: float(value[0])
        for modality, value in shapley_from_coalitions(coalitions, modalities).items()
    }

    base = float(coalitions[frozenset()][0])
    full = float(coalitions[frozenset(modalities)][0])
    total_abs = sum(abs(v) for v in phi.values()) + 1e-12
    cxr_frac = abs(phi[target_modality]) / total_abs
    return ModalityShapley(
        values=phi,
        base_value=base,
        full_value=full,
        cxr_fraction=cxr_frac,
    )


# Keep the other patients fixed when explaining M3Care predictions
def modality_shapley_frozen(
    scorer,
    target_idx: int,
    *,
    modalities: tuple[str, ...] | list[str] | None = None,
    target_modality: str = "cxr",
) -> ModalityShapley:
    """Measure input contributions while keeping M3Care's patient links fixed.

    Use the same patient batch as the model and change only the selected patient's
    inputs. This keeps other patients available when the model fills in missing data.
    """

    modalities = tuple(modalities or MODALITIES)
    coalitions = scorer.coalition_values(int(target_idx), modalities)
    phi_arrays = shapley_from_coalitions(coalitions, modalities)
    phi = {modality: float(phi_arrays[modality][0]) for modality in modalities}
    contribution = estimate_mm_summary(phi_arrays, modalities)["global_contribution"]
    return ModalityShapley(
        values=phi,
        base_value=float(coalitions[frozenset()][0]),
        full_value=float(coalitions[frozenset(modalities)][0]),
        cxr_fraction=float(contribution[target_modality]),
    )


def frozen_cxr_logit_fn(
    scorer,
    target_idx: int,
    cxr_modality: str = "cxr",
):
    """Return a score function for chest X-ray features with patient links fixed.

    Only the selected patient's image features change. Gradients can flow back
    to the image while the other patients and their links stay fixed.
    """

    inner = scorer.inner
    target_idx = int(target_idx)

    def fn(cxr_emb: torch.Tensor) -> torch.Tensor:
        emb = cxr_emb.to(scorer.device)
        lengths = torch.ones(emb.shape[0], dtype=torch.int64, device=scorer.device)
        encoder = inner.modality_encoder(cxr_modality)
        encoded = encoder(emb.unsqueeze(1), lengths)   # (C,1,D) -> (C,H)
        node, mask = scorer._broadcast_target(
            target_idx, {cxr_modality: encoded}
        )
        return freeze_m3care.forward_target_from_embeddings(
            inner, node, mask, scorer.adj, target_idx
        ).reshape(-1, 1)

    return fn


def live_cxr_logit_fn(
    g: CXRGradCAM,
    batch: dict,
    target_idx: int,
    cxr_modality: str = "cxr",
):
    """Return a score function that changes one patient's chest X-ray features."""
    prefix = cxr_modality
    target_idx = int(target_idx)

    def fn(cxr_emb: torch.Tensor) -> torch.Tensor:
        emb = cxr_emb.reshape(-1, cxr_emb.shape[-1])
        outs: list[torch.Tensor] = []
        for i in range(emb.shape[0]):
            edited = dict(batch)
            feats = batch[f"{prefix}_feats"].clone()
            feats[target_idx, 0, :] = emb[i].to(feats.device, feats.dtype)
            edited[f"{prefix}_feats"] = feats
            for suffix, value in (("lengths", 1), ("missing", 0)):
                key = f"{prefix}_{suffix}"
                if key in batch:
                    values = batch[key].clone()
                    values[target_idx] = value
                    edited[key] = values
            outs.append(g.mm(edited).reshape(-1)[target_idx])
        return torch.stack(outs).unsqueeze(-1)
    return fn


@dataclass
class CXRPatientContext:
    """Store how one patient is scored with chest X-ray features rebuilt from images."""

    logit_fn: Callable[[torch.Tensor], torch.Tensor]
    scorer: M3CareFrozenGraphScorer | None  # Fixed-graph scorer, M3Care with imputation only
    target_idx: int  # Patient's row in batch
    batch: dict | None  # One-row batch, models that score patients separately only
    co_batch: tuple[int, int] | None  # First and last split rows of the fixed-graph batch


def patient_cxr_context(
    g: CXRGradCAM,
    batches: SplitBatches,
    idx: int,
    stay_emb: torch.Tensor,
    use_frozen: bool,
) -> CXRPatientContext:
    """Score one patient in the batch context its saved prediction came from.

    Replace the patient's saved image features with stay_emb so the score is
    tied to the features rebuilt from the pixels rather than to the saved row.
    M3Care with imputation uses the patient's real batch with its links fixed.
    Other models use the patient's own row with all its inputs.
    """
    cxr_modality = g.cxr_modality
    if use_frozen:
        batch, start, end, target_idx = batches.co_batch(idx)
        substitute_live_cxr(batch, target_idx, stay_emb, cxr_modality)
        scorer = batches.scorer(g.mm, batch, start, cxr_modality)
        return CXRPatientContext(
            logit_fn=frozen_cxr_logit_fn(scorer, target_idx, cxr_modality),
            scorer=scorer, target_idx=target_idx, batch=None, co_batch=(start, end),
        )
    # Move every input to the model's device before replacing image features
    batch = _move_batch_to_device(batches.single_row(idx), g.device)
    substitute_live_cxr(batch, 0, stay_emb, cxr_modality)
    return CXRPatientContext(
        logit_fn=live_cxr_logit_fn(g, batch, 0, cxr_modality),
        scorer=None, target_idx=0, batch=batch, co_batch=None,
    )


# Weight image regions using SHAP contributions from chest X-ray features
def _resolve_background(
    g: CXRGradCAM,
    n_features: int,
    background_size: int,
    background_seed: int,
) -> torch.Tensor:
    """Use training chest X-ray features as a reference when available.

    Warn and use zeros if the requested reference cannot be read. Return zeros
    without a warning when no reference was requested.
    """
    if background_size and int(background_size) > 0:
        build = getattr(g, "cxr_background", None)
        if build is not None:
            try:
                return build(n_samples=int(background_size), seed=int(background_seed))
            except (ValueError, FileNotFoundError, OSError) as exc:
                warnings.warn(
                    f"Falling back to an all-zero CXR SHAP baseline: {exc}. Channel "
                    "attributions are then measured against a vector that is not a "
                    "chest X-ray; pass `background` explicitly for a faithful reference.",
                    RuntimeWarning, stacklevel=3,
                )
    return torch.zeros(1, n_features, device=g.device)


def shap_weighted_cam(
    g: CXRGradCAM,
    image_path: str,
    *,
    logit_fn,
    context_row: dict | None = None,
    background: torch.Tensor | None = None,
    other_raw_embeddings: torch.Tensor | None = None,
    explainer: str = "gradient",
    kernel_nsamples: int | str = "auto",
    background_size: int = 50,
    background_seed: int = 0,
    shap_seed: int = DEFAULT_SHAP_SEED,
    kernel_background_summary: int = 1,
) -> CAMResult:
    """Highlight image regions using SHAP contributions for chest X-ray features.

    Use the predicted score before conversion to a probability, as Grad-CAM does.
    Gradient SHAP uses all reference images and is the default. Kernel SHAP uses
    a smaller reference sample and may take longer. Supply background to choose
    a reference directly, or set background_size to zero to use a zero reference.
    logit_fn scores chest X-ray features in the patient's batch context, as
    patient_cxr_context builds it.

    background_seed selects training images for the reference. shap_seed controls
    SHAP sampling so repeated calls with the same settings produce the same map.
    """
    chosen = _select_engine(explainer)
    import shap

    # Keep the image features linked to the image while building the map
    from src.explanation.gradcam import load_and_preprocess_image

    arr = load_and_preprocess_image(image_path)
    img = torch.from_numpy(arr).to(g.device).requires_grad_(True)
    with torch.enable_grad():
        spatial = g._densenet_spatial(img)                 # (1,1024,7,7)
    raw_target = F.adaptive_avg_pool2d(spatial, (1, 1)).reshape(1, -1)
    if other_raw_embeddings is not None and other_raw_embeddings.shape[0] > 0:
        others = other_raw_embeddings.to(g.device).detach()
        T = others.shape[0] + 1
        mean_raw = (raw_target + others.sum(dim=0, keepdim=True)) / T
    else:
        mean_raw = raw_target
    from src.explanation.gradcam import apply_cxr_norm

    emb = mean_raw
    if g.min_values is not None:
        emb = apply_cxr_norm(emb, g.min_values, g.max_values)
    emb = emb.detach().requires_grad_(True)                # (1,1024)

    if background is None:
        background = _resolve_background(g, emb.shape[1], background_size, background_seed)

    fn = logit_fn

    if chosen == "kernel":
        # Let Kernel SHAP evaluate scores from chest X-ray features
        def f_np(x: np.ndarray) -> np.ndarray:
            x = np.atleast_2d(np.asarray(x, dtype=np.float32))
            with torch.no_grad():
                out = fn(torch.from_numpy(x).to(g.device))
            return out.detach().cpu().numpy().reshape(-1)

        # Use fewer reference samples to keep Kernel SHAP practical
        bg_np = background.detach().cpu().numpy()
        k = max(1, min(kernel_background_summary, bg_np.shape[0]))
        bg_summary = shap.kmeans(bg_np, k) if bg_np.shape[0] > k else bg_np
        kexp = shap.KernelExplainer(f_np, bg_summary)
        # Kernel SHAP reads NumPy's random seed when sampling features
        np.random.seed(int(shap_seed))
        sv = kexp.shap_values(
            emb.detach().cpu().numpy(), nsamples=kernel_nsamples, silent=True
        )
    else:  # Gradient SHAP uses all reference samples
        class _Wrap(torch.nn.Module):
            def forward(self, x):
                return fn(x)

        gexp = shap.GradientExplainer(_Wrap(), background)
        # Fix the seed so repeated calls produce the same map
        sv = gexp.shap_values(emb, rseed=int(shap_seed))

    if isinstance(sv, list):
        sv = sv[0]
    channel_shap = torch.as_tensor(np.asarray(sv).reshape(-1), dtype=torch.float32, device=g.device)

    # Weight each image region by its SHAP contribution
    A = spatial.squeeze(0)                                   # (1024,7,7)
    cam = F.relu((channel_shap.view(-1, 1, 1) * A).sum(dim=0))
    cam_low = cam.detach().cpu().numpy()
    cam_n = cam_low - cam_low.min()
    if cam_n.max() > 0:
        cam_n = cam_n / cam_n.max()
    H = W = arr.shape[-1]
    cam_up = F.interpolate(
        torch.as_tensor(cam_n)[None, None].float(), size=(H, W),
        mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    with torch.no_grad():
        logit = fn(emb).reshape(-1)[0]
    return CAMResult(
        ed_stay_id=(
            int(context_row.get(getattr(g, "identifier_column", "ed_stay_id"), 0))
            if context_row else 0
        ),
        dicom_id=image_path.split("/")[-1].split(".")[0],
        image_path=image_path,
        logit=float(logit.item()),
        prob=float(torch.sigmoid(logit).item()),
        cam=cam_up,
        cam_lowres=cam_low,
        original=arr.squeeze(),
        label=(
            float(context_row["label"])
            if context_row is not None and context_row.get("label") is not None
            else None
        ),
    )
