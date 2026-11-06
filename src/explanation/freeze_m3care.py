"""Explain M3Care predictions while keeping other patients fixed.

M3Care uses patients in the same batch to fill in missing inputs. Save their
connections and model inputs, then change one patient's inputs while the
other patients and their connections stay fixed.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from src.evaluation.eval_utils import _move_batch_to_device


def use_frozen_graph(model) -> bool:
    """Check whether M3Care links patient predictions through imputation.

    Other models and M3Care without imputation can score patients separately.
    """
    inner = getattr(model, "model", model)
    if type(inner).__name__ != "M3CareModel":
        return False
    return bool(getattr(inner, "perform_imputation", False))


@torch.no_grad()
def capture_reference_graph(
    inner: torch.nn.Module, batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Save model inputs and patient connections from one real batch.

    The result keeps input features under ``embeddings``, availability flags
    under ``masks``, and patient links under ``adjacency``. Their shapes are
    ``(B, H)``, ``(B, 1)`` and ``(B, B)``, where ``B`` is the patient count and
    ``H`` is the feature width. Tensors stay on the model's device without
    retaining gradients.
    """
    if not getattr(inner, "perform_imputation", False):
        raise ValueError(
            "capture_reference_graph requires a M3Care model with imputation enabled; "
            "without imputation there is no cross-patient graph to freeze."
        )
    device = next(inner.parameters()).device
    batch = _move_batch_to_device(batch, device)

    emb = inner.encode(batch)
    masks = inner._modality_mask(batch, emb)

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module, args):
        # The first argument is the patient similarity graph
        captured["adj"] = args[0].detach()

    handle = inner.imputation.GCN[0].register_forward_pre_hook(_pre_hook)
    try:
        inner.imputation(emb, masks)  # Capture the graph and discard the updated inputs
    finally:
        handle.remove()

    if "adj" not in captured:
        raise RuntimeError(
            "Failed to capture the M3Care adjacency: imputation.GCN[0] was not called. "
            "The model's GCN_imputation.forward layout may have changed."
        )

    return {
        "embeddings": {k: v.detach() for k, v in emb.items()},
        "masks": {k: v.detach() for k, v in masks.items()},
        "adjacency": captured["adj"],
    }


def _graph_conv(gcn: torch.nn.Module, adjacency: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """Apply one trained graph layer using saved patient connections.

    Features can describe one patient batch or several changed versions of it.
    Every version uses the same patient connections.
    """
    y = torch.matmul(feats, gcn.weight)
    out = torch.matmul(adjacency, y)
    if gcn.bias is not None:
        out = out + gcn.bias
    return out


def frozen_refine(
    inner: torch.nn.Module,
    node_feats: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    adjacency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Update model inputs through two layers without changing patient links.

    Extra leading dimensions let several input changes share the same patient
    links. Only input types selected for filling in are updated. Use the
    trained layers in the order of those selected types. Learned mixing
    weights apply only in adaptive mode.
    """
    imp = inner.imputation
    relu = imp.relu
    to_impute = getattr(imp, "modalities_to_impute", imp.modality_names)
    mode = imp.gcn_refine_mode
    refined: dict[str, torch.Tensor] = {}
    impute_idx = 0
    for name in imp.modality_names:
        x = node_feats[name]
        if name not in to_impute:
            refined[name] = x
            continue
        mask = masks[name].to(device=x.device, dtype=x.dtype)

        x0 = relu(_graph_conv(imp.GCN[impute_idx], adjacency, x))
        x0 = relu(_graph_conv(imp.GCN_2[impute_idx], adjacency, x0))

        if mode == "adaptive":
            w1 = torch.sigmoid(imp.weight1[impute_idx](x0))
            w2 = torch.sigmoid(imp.weight2[impute_idx](x))
            w1 = w1 / (w1 + w2 + 1e-8)
            w2 = 1.0 - w1

            w1 = w1 * mask + (1.0 - mask)
            w2 = w2 * mask
            x_final = w1 * x0 + w2 * x
        elif mode == "imputed_only":
            x_final = x0
        elif mode == "imputed_for_missingness":
            x_final = (1.0 - mask) * x0 + mask * x
        else:  # pragma: no cover - the model checks this during initialisation
            raise ValueError(
                f"Unknown gcn_refine_mode={mode!r}; expected 'adaptive', 'imputed_only', "
                "or 'imputed_for_missingness'."
            )
        refined[name] = x_final
        impute_idx += 1
    return refined


def forward_target_from_embeddings(
    inner: torch.nn.Module,
    node_feats: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    adjacency: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    """Score one patient across several input changes.

    Only the selected patient's inputs vary. Both graph layers still include
    every patient, so a change can pass through neighbours and return to the
    selected patient. Return one model score for each change.
    """
    refined = frozen_refine(inner, node_feats, masks, adjacency)
    target_rows = {name: refined[name][:, target_idx, :] for name in refined}  # (C, H)
    target_masks = {name: masks[name][:, target_idx, :] for name in masks}  # (C, 1)
    fused = inner.fuse(target_rows, target_masks)["embeds"]
    return inner.predict(fused)
