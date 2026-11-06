from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from src.models.concat import build_concat_model_from_config
from src.models.fusemoe import build_fusemoe_model_from_config
from src.models.m3care import build_m3care_model_from_config
from src.models.mtand_m3care import build_mtand_m3care_model_from_config
from src.models.smp import build_smp_model_from_config

ACTIVE_FUSION_METHODS = ("concat", "m3care", "fusemoe", "smp", "mtand_m3care")


def _require_feature_dim(feature_dims: Mapping[str, int], key: str) -> int:
    try:
        return int(feature_dims[key])
    except KeyError as exc:
        raise ValueError(
            f"feature_dims must include {key!r} to build the active fusion model."
        ) from exc


def _configured_feature_dims(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
    default_modalities: tuple[str, ...] = ("tabular", "ecg", "cxr"),
) -> dict[str, int]:
    """Select configured modalities, using feature names in their given order by default."""
    modalities = fusion_config.get("modality_names")
    if modalities is None:
        modalities = tuple(feature_dims) if feature_dims else default_modalities
    return {modality: _require_feature_dim(feature_dims, modality) for modality in modalities}


def _build_concat_model(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
) -> nn.Module:
    return build_concat_model_from_config(
        fusion_config, _configured_feature_dims(fusion_config, feature_dims),
    )


def _modal_feature_dims(feature_dims: Mapping[str, int]) -> dict[str, int]:
    return {
        "tabular": _require_feature_dim(feature_dims, "tabular"),
        "ecg": _require_feature_dim(feature_dims, "ecg"),
        "cxr": _require_feature_dim(feature_dims, "cxr"),
    }


def _build_m3care_model(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
) -> nn.Module:
    return build_m3care_model_from_config(
        fusion_config, _configured_feature_dims(fusion_config, feature_dims),
    )


def _build_fusemoe_model(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
) -> nn.Module:
    # Include the raw time-series width when the dataset provides it
    dims = _configured_feature_dims(fusion_config, feature_dims)
    if "ts" in feature_dims:
        dims["ts"] = _require_feature_dim(feature_dims, "ts")
    return build_fusemoe_model_from_config(fusion_config, dims)


def _build_smp_model(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
) -> nn.Module:
    return build_smp_model_from_config(
        fusion_config, _configured_feature_dims(fusion_config, feature_dims),
    )


def _build_mtand_m3care_model(
    fusion_config: Mapping[str, Any], feature_dims: Mapping[str, int],
) -> nn.Module:
    # Include the raw time-series width for models that enable it
    dims = _modal_feature_dims(feature_dims)
    if "ts" in feature_dims:
        dims["ts"] = _require_feature_dim(feature_dims, "ts")
    return build_mtand_m3care_model_from_config(fusion_config, dims)


def build_fusion_model(
    fusion_method: str, fusion_config: Mapping[str, Any] | None,
    feature_dims: Mapping[str, int] | None,
) -> nn.Module:
    """Build a fusion model from its name, settings and modality feature widths."""
    fusion_method = fusion_method.lower()
    config = dict(fusion_config or {})
    dims = dict(feature_dims or {})

    if fusion_method == "concat":
        return _build_concat_model(config, dims)
    if fusion_method == "m3care":
        return _build_m3care_model(config, dims)
    if fusion_method == "fusemoe":
        return _build_fusemoe_model(config, dims)
    if fusion_method == "mtand_m3care":
        return _build_mtand_m3care_model(config, dims)
    if fusion_method == "smp":
        return _build_smp_model(config, dims)

    available = ", ".join(ACTIVE_FUSION_METHODS)
    raise ValueError(
        f"Unknown active fusion method {fusion_method!r}. "
        f"Available fusion methods: {available}."
    )
