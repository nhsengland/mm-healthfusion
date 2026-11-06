"""Generate fictional average-record split bundles for the deterioration and hospitalisation pathways.

Every record, identifier and outcome written here is artificial. The values come from a small set
of random latent factors, so the outcomes are learnable, but they have no clinical meaning and do
not follow real clinical distributions.

The files follow the average-record layout of the real split bundles, the input of the static
fusion models. Each ECG, CXR and notes modality holds one averaged embedding. Vital-sign and
lab-like measurements are summarised into nine statistics each. For deterioration, the summaries
join the tabular vector. For hospitalisation, they form the ``ts_prev`` and ``ts_curr`` vectors.
No observation times are stored. The existing datasets, collators and static models read the
files without changes.

Example:
    python -m src.preprocessing.generate_synthetic --task both --n-samples 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.datasets import (
    DETERIORATION_MODALITIES,
    HOSPITALISATION_MODALITIES,
    SPLIT_NAMES,
    TABULAR_MODALITIES,
    _frame_modality_available,
    modality_prefix,
)
from src.preprocessing.split_deterioration import (
    build_arrow_schema as build_deterioration_schema,
)
from src.preprocessing.split_hospitalisation import (
    TS_STAT_NAMES,
    assign_shared_split,
    parse_modality_specs,
    validate_split_invariants,
)
from src.preprocessing.split_hospitalisation import (
    write_split_parquet as write_hospitalisation_parquet,
)

# Folder holding this repository beside data/, outputs/ and logs/
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "synthetic"

GENERATOR_VERSION = "2.1.0"
RECORD_LAYOUT = "average"
TASKS = ("deterioration", "hospitalisation")
OUTCOMES = {
    "deterioration": "deterioration_mechanical_ventilation",
    "hospitalisation": "hospitalisation_ext_stay_7",
}
MODALITIES = {
    "deterioration": DETERIORATION_MODALITIES,
    "hospitalisation": HOSPITALISATION_MODALITIES,
}
MANIFEST_FILENAME = "synthetic_manifest.json"
DEMOGRAPHICS_FILENAME = "synthetic_demographics.csv"
# Named like the split scripts' summaries, which src.explanation reads for tabular feature names
SPLIT_SUMMARY_FILENAME = "synthetic_split_summary.json"
MIN_SAMPLES = 100
MAX_MISSING_RATE = 0.9
# Measurements are split into a vital-sign-like half and a lab-like half
MIN_MEASUREMENTS = 2
# Statistics that need two observations. Hospitalisation leaves them missing after one.
TREND_STATS = ("rate_change", "coeff")

# Observation windows in minutes. Deterioration uses the first 90 minutes after ED arrival.
DETERIORATION_WINDOW_MINUTES = 90.0
CURRENT_WINDOW_RANGE_MINUTES = (120.0, 720.0)
PREVIOUS_WINDOW_RANGE_MINUTES = (1_440.0, 14 * 1_440.0)

# Embedding widths of the real extractors: ECG-FM, the TorchXRayVision DenseNet and
# Bio_Discharge_Summary_BERT
EMBEDDING_DIMS = {"ecg": 768, "cxr": 1024, "notes": 768}

# Maximum raw observations averaged into one embedding, and summarised per measurement series
MAX_OBSERVATIONS = {"ecg": 2, "cxr": 3, "notes": 4}
MAX_TS_STEPS = 12

# Outcome log-odds: intercept plus one weight per latent factor
OUTCOME_COEFFICIENTS = {
    "deterioration": {
        "intercept": -1.2, "acuity": 1.4, "frailty": 0.6, "cxr_factor": 0.6, "ecg_factor": 0.4,
    },
    "hospitalisation": {
        "intercept": -1.0, "acuity": 0.8, "frailty": 1.1, "notes_factor": 0.6, "cxr_factor": 0.4,
    },
}
OUTCOME_NOISE_SD = 0.5

# Fictional identifier offsets, well clear of the MIMIC-IV identifier ranges
ID_OFFSETS = {
    "deterioration": {"subject_id": 1_000_000, "ed_hadm_id": 2_000_000, "ed_stay_id": 3_000_000},
    "hospitalisation": {"subject_id": 5_000_000, "ed_hadm_id": 6_000_000, "ed_stay_id": 7_000_000},
}

SIGNAL_DESCRIPTION = (
    "Each fictional patient has two shared latent factors, acuity and frailty, plus one factor "
    "per modality family. Base tabular features and embeddings are noisy projections of the "
    "shared factors and their own factor. Each ECG, CXR or notes embedding is the average of one "
    "to several raw observations. Vital-sign-like and lab-like measurements are sampled at "
    "irregular times inside the observation window, drift with acuity, and are then reduced to "
    "nine summary statistics, so only the summaries are stored. The binary outcome is drawn from "
    "a logistic model of the factors listed under outcome_coefficients, so the modality-specific "
    "factors can only be recovered when that modality is present."
)
MISSINGNESS_DESCRIPTION = (
    "Missing values are stored as 0.0 with a mask of 1 (0 = observed, 1 = missing). Base tabular "
    "features go missing more often for low-acuity patients, and the first base feature is always "
    "observed. A measurement that was never taken leaves all nine of its statistics missing. In "
    "hospitalisation, a measurement taken once also leaves rate_change and coeff missing. ECG and "
    "CXR absence shares a common random effect and is more likely at low acuity, so the two are "
    "correlated and informative. For hospitalisation, patients without a previous admission have "
    "ts_prev, cxr_prev and notes_prev all absent together. An absent ECG, CXR, notes or "
    "time-series modality has an empty feats array, and ECG, CXR and notes also carry a "
    "modality-level missing flag of 1."
)
# Fictional demographics, written for src.evaluation.fairness with planted, documented biases.
# Ages rise with frailty, which raises outcome risk, so older groups have a higher prevalence.
AGE_MEAN, AGE_PER_FRAILTY, AGE_NOISE_SD, AGE_RANGE = 62.0, 15.0, 8.0, (18, 100)
# Same bands as src.evaluation.fairness, which this module does not import because it switches
# matplotlib to a non-interactive backend
AGE_BINS = (0, 30, 50, 65, 80, np.inf)
AGE_LABELS = ("<30", "30-49", "50-64", "65-79", "80+")
ETHNICITY_SHARES = {
    "White": 0.60, "Black": 0.15, "Hispanic/Latino": 0.08, "Asian": 0.07, "Unrecorded": 0.10,
}
# Patients with unrecorded ethnicity also have sparser, lower-quality records: optional modalities
# are absent and base features missing this much more often, and their features measure acuity and
# frailty with extra error of this standard deviation. Their outcome risk is unchanged.
UNRECORDED_ETHNICITY = "Unrecorded"
UNRECORDED_EXTRA_MISSING = 0.35
UNRECORDED_MEASUREMENT_NOISE = 1.0
# Medicare covers most patients aged 65 or over and the rest are privately insured
MEDICARE_AGE = 65
MEDICARE_SHARE = 0.9
INSURANCE_SHARES_UNDER_MEDICARE_AGE = {"Private": 0.55, "Medicaid": 0.35, "Other": 0.10}
# Ethnicity labels in each pathway's source format. src.evaluation.fairness reports both
# "UNKNOWN" and "Other" as the Other group.
ETHNICITY_LABELS = {
    "deterioration": {
        "White": "WHITE", "Black": "BLACK/AFRICAN AMERICAN", "Hispanic/Latino": "HISPANIC/LATINO",
        "Asian": "ASIAN", "Unrecorded": "UNKNOWN",
    },
    "hospitalisation": {
        "White": "White", "Black": "Black", "Hispanic/Latino": "Hispanic/Latino",
        "Asian": "Asian", "Unrecorded": "Other",
    },
}
DEMOGRAPHIC_BIAS = {
    "sex": "None. Sex is drawn independently of every factor, so it serves as a null control and "
    "fairness ratios should differ from 1 only by sampling noise.",
    "age": "Base-rate gap. Age rises with frailty and frailty raises outcome risk, so older age "
    "groups have a higher prevalence. Selection rates differ by age even for a well-calibrated "
    "model, which lowers the demographic parity ratio.",
    "ethnicity": "Information gap. Patients with unrecorded ethnicity (reported as Other by the "
    f"fairness script) lose optional modalities and base features {UNRECORDED_EXTRA_MISSING:.2f} "
    "more often, and their whole record, including which values are missing, follows acuity and "
    f"frailty measured with extra error (standard deviation {UNRECORDED_MEASUREMENT_NOISE:.1f}). "
    "Their outcome still follows the true factors, so their risk is unchanged. Models see less "
    "reliable signal for this group, so its discrimination is expected to be weaker, its error "
    "rates higher and the equalised odds and equal opportunity ratios lower.",
    "insurance": "Inherited base-rate gap (hospitalisation only). Medicare covers most patients "
    f"aged {MEDICARE_AGE} or over, so it inherits the higher prevalence of older groups. "
    "Insurance has no direct effect.",
    "cxr_availability": "Informative missingness (deterioration only). CXR absence is more likely "
    "at low acuity, so the CXR present and absent groups differ in prevalence.",
}

ARTIFICIAL_DATA_NOTICE = (
    "Synthetic data. Every record, identifier and outcome is artificial and has no clinical "
    "interpretation. No real patient information was used."
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read and validate the generator options."""
    parser = argparse.ArgumentParser(
        description="Generate fictional mm-healthfusion average-record split bundles."
    )
    parser.add_argument(
        "--task", choices=(*TASKS, "both"), default="both",
        help="Pathway to generate. Defaults to both.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help=f"Records per pathway, split 80/10/10. At least {MIN_SAMPLES}. Defaults to 1000.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Defaults to 42.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output root holding one folder per pathway. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--missing-rate", type=float, default=0.3,
        help=f"Baseline probability that an optional modality is absent, between 0 and "
        f"{MAX_MISSING_RATE}. Defaults to 0.3.",
    )
    parser.add_argument(
        "--tabular-dim", type=int, default=16,
        help="Number of base tabular and static features. Defaults to 16.",
    )
    parser.add_argument(
        "--ts-dim", type=int, default=8,
        help="Number of vital-sign and lab measurements, each summarised by nine statistics. "
        "Defaults to 8.",
    )
    for kind, width in EMBEDDING_DIMS.items():
        parser.add_argument(
            f"--{kind}-dim", type=int, default=width,
            help=f"Width of the {kind.upper() if kind != 'notes' else 'notes'} embeddings. "
            f"Defaults to {width}, as in the real data.",
        )
    args = parser.parse_args(argv)

    if args.n_samples < MIN_SAMPLES:
        parser.error(f"--n-samples must be at least {MIN_SAMPLES}, got {args.n_samples}.")
    if not 0.0 <= args.missing_rate <= MAX_MISSING_RATE:
        parser.error(
            f"--missing-rate must be between 0 and {MAX_MISSING_RATE}, got {args.missing_rate}."
        )
    for option in ("tabular_dim", *(f"{kind}_dim" for kind in EMBEDDING_DIMS)):
        if getattr(args, option) < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive.")
    if args.ts_dim < MIN_MEASUREMENTS:
        parser.error(
            f"--ts-dim must be at least {MIN_MEASUREMENTS} so both measurement groups are "
            "represented."
        )
    return args


def measurement_feature_names(ts_dim: int) -> list[str]:
    """Name each summary statistic, grouped by measurement in ``TS_STAT_NAMES`` order."""
    vital_count = (ts_dim + 1) // 2
    measurements = [
        f"vital_{index}" if index < vital_count else f"lab_{index - vital_count}"
        for index in range(ts_dim)
    ]
    return [f"{measurement}_{stat}" for measurement in measurements for stat in TS_STAT_NAMES]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _projection(rng: np.random.Generator, n_inputs: int, width: int) -> np.ndarray:
    return rng.normal(scale=1.0 / np.sqrt(n_inputs), size=(n_inputs, width))


def _draw_demographics(rng: np.random.Generator, frailty: np.ndarray) -> pd.DataFrame:
    """Draw fictional age, sex, ethnicity and insurance for each patient."""
    n_samples = len(frailty)
    age = np.clip(
        np.rint(AGE_MEAN + AGE_PER_FRAILTY * frailty + AGE_NOISE_SD * rng.normal(size=n_samples)),
        *AGE_RANGE,
    ).astype(np.int64)
    gender = rng.integers(0, 2, size=n_samples)
    ethnicity = rng.choice(
        list(ETHNICITY_SHARES), size=n_samples, p=list(ETHNICITY_SHARES.values())
    )
    older = age >= MEDICARE_AGE
    younger_insurance = rng.choice(
        list(INSURANCE_SHARES_UNDER_MEDICARE_AGE), size=n_samples,
        p=list(INSURANCE_SHARES_UNDER_MEDICARE_AGE.values()),
    )
    medicare = older & (rng.random(n_samples) < MEDICARE_SHARE)
    insurance = np.where(medicare, "Medicare", np.where(older, "Private", younger_insurance))
    return pd.DataFrame(
        {"age": age, "gender": gender, "ethnicity": ethnicity, "insurance": insurance}
    )


def _missing_rates(demographics: pd.DataFrame, missing_rate: float) -> np.ndarray:
    """Raise the baseline missing rate for patients with unrecorded ethnicity."""
    unrecorded = (demographics["ethnicity"] == UNRECORDED_ETHNICITY).to_numpy()
    return missing_rate + UNRECORDED_EXTRA_MISSING * unrecorded


def _measured_factors(
    rng: np.random.Generator, factors: dict[str, np.ndarray], demographics: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return acuity and frailty as the features record them.

    Records of patients with unrecorded ethnicity measure both with extra error. The outcome
    always uses the true factors.
    """
    unrecorded = (demographics["ethnicity"] == UNRECORDED_ETHNICITY).to_numpy()
    return tuple(
        factors[name]
        + UNRECORDED_MEASUREMENT_NOISE * unrecorded * rng.normal(size=unrecorded.shape)
        for name in ("acuity", "frailty")
    )


def _absence(
    rng: np.random.Generator,
    acuity: np.ndarray,
    shared_effect: np.ndarray,
    rate: float | np.ndarray,
) -> np.ndarray:
    """Draw modality absence that averages ``rate`` and rises at low acuity.

    ``shared_effect`` is common to related modalities, which correlates their absence. ``rate``
    is a baseline or one rate per patient.
    """
    scale = 2.0 * _sigmoid(-0.8 * acuity + shared_effect + 0.5 * rng.normal(size=acuity.shape))
    return rng.random(acuity.shape) < np.clip(rate * scale, 0.0, 0.95)


def _tabular_block(
    rng: np.random.Generator,
    latents: np.ndarray,
    acuity: np.ndarray,
    *,
    width: int,
    missing_rate: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [0, 1] base features and their element-wise mask, one row per patient."""
    values = _sigmoid(
        np.einsum("ni,ij->nj", latents, _projection(rng, latents.shape[1], width))
        + 0.5 * rng.normal(size=(latents.shape[0], width))
    )
    probability = np.clip(
        (missing_rate * (1.0 - 0.5 * np.tanh(acuity)))[:, None], 0.0, 0.95
    )
    missing = rng.random(values.shape) < probability
    missing[:, 0] = False
    values[missing] = 0.0
    return values.astype(np.float32), missing.astype(np.int8)


def _averaged_embeddings(
    rng: np.random.Generator,
    latents: np.ndarray,
    present: np.ndarray,
    *,
    width: int,
    max_observations: int,
) -> tuple[list[list[np.ndarray]], list[int], list[int]]:
    """Return one averaged embedding per present patient, scalar flags and observation counts."""
    weights = _projection(rng, latents.shape[1], width)
    feats, missing, counts = [], [], []
    for row, is_present in zip(latents, present, strict=True):
        if not is_present:
            feats.append([])
            missing.append(1)
            counts.append(0)
            continue
        count = int(rng.integers(1, max_observations + 1))
        observations = np.einsum("i,ij->j", row, weights) + 0.5 * rng.normal(size=(count, width))
        feats.append([observations.mean(axis=0).astype(np.float32)])
        missing.append(0)
        counts.append(count)
    return feats, missing, counts


def _summarise_series(
    values: np.ndarray,
    observed: np.ndarray,
    progress: np.ndarray,
    *,
    single_observation_trend_missing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one patient's series to nine statistics per measurement, with a matching mask.

    ``progress`` is each observation's position in the window, from 0 to 1. Trend statistics are
    rescaled from [-1, 1] to [0, 1] so every statistic shares the [0, 1] range.
    """
    n_measurements = values.shape[1]
    stats = np.zeros((n_measurements, len(TS_STAT_NAMES)), dtype=np.float32)
    missing = np.ones_like(stats, dtype=np.int8)
    trend_columns = [TS_STAT_NAMES.index(name) for name in TREND_STATS]
    for measurement in range(n_measurements):
        taken = observed[:, measurement]
        series, at = values[taken, measurement], progress[taken]
        if series.size == 0:
            continue
        if series.size > 1 and np.ptp(at) > 0:
            slope = np.polyfit(at, series, 1)[0]
            rate = (series[-1] - series[0]) / series[0]
        else:
            slope = rate = 0.0
        summary = {
            "mean": series.mean(), "median": np.median(series), "min": series.min(),
            "max": series.max(), "std": series.std(), "first": series[0], "last": series[-1],
            "rate_change": (np.clip(rate, -1.0, 1.0) + 1.0) / 2.0,
            "coeff": (np.clip(slope, -1.0, 1.0) + 1.0) / 2.0,
        }
        stats[measurement] = [summary[name] for name in TS_STAT_NAMES]
        missing[measurement] = 0
        if series.size == 1 and single_observation_trend_missing:
            stats[measurement, trend_columns] = 0.0
            missing[measurement, trend_columns] = 1
    return stats.ravel(), missing.ravel()


def _summarised_time_series(
    rng: np.random.Generator,
    latents: np.ndarray,
    acuity: np.ndarray,
    present: np.ndarray,
    *,
    n_measurements: int,
    missing_rate: float,
    single_observation_trend_missing: bool,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sample irregular measurement series and return only their summary statistics.

    Sicker patients are measured more often and drift further across the window. Vital-sign-like
    measurements are taken more often than lab-like ones. Rows of absent patients are all missing.
    """
    weights = _projection(rng, latents.shape[1], n_measurements)
    trend = rng.choice((-1.0, 1.0), size=n_measurements)
    vital_like = np.arange(n_measurements) < (n_measurements + 1) // 2
    observe_probability = np.where(
        vital_like, 1.0 - missing_rate / 3.0, np.clip(1.0 - 2.0 * missing_rate, 0.1, 1.0)
    )
    width = n_measurements * len(TS_STAT_NAMES)
    summaries = np.zeros((len(acuity), width), dtype=np.float32)
    masks = np.ones((len(acuity), width), dtype=np.int8)
    counts = []
    for index, (row, severity, is_present) in enumerate(
        zip(latents, acuity, present, strict=True)
    ):
        if not is_present:
            counts.append(0)
            continue
        count = int(min(MAX_TS_STEPS, 1 + rng.poisson(np.exp(0.8 + 0.5 * severity))))
        progress = np.sort(rng.uniform(0.0, 1.0, size=count))
        values = _sigmoid(
            row @ weights + 0.8 * severity * progress[:, None] * trend
            + 0.3 * rng.normal(size=(count, n_measurements))
        )
        observed = rng.random((count, n_measurements)) < observe_probability
        observed[:, 0] = True
        summaries[index], masks[index] = _summarise_series(
            values, observed, progress,
            single_observation_trend_missing=single_observation_trend_missing,
        )
        counts.append(count)
    return summaries, masks, counts


def _draw_outcome(
    rng: np.random.Generator,
    task: str,
    factors: dict[str, np.ndarray],
) -> np.ndarray:
    coefficients = OUTCOME_COEFFICIENTS[task]
    log_odds = np.full(len(factors["acuity"]), coefficients["intercept"])
    for name, weight in coefficients.items():
        if name != "intercept":
            log_odds = log_odds + weight * factors[name]
    log_odds = log_odds + OUTCOME_NOISE_SD * rng.normal(size=log_odds.shape)
    return (rng.random(log_odds.shape) < _sigmoid(log_odds)).astype(np.int64)


def _identifiers(task: str, n_samples: int) -> dict[str, np.ndarray]:
    index = np.arange(1, n_samples + 1, dtype=np.int64)
    return {column: offset + index for column, offset in ID_OFFSETS[task].items()}


def _no_times(n_samples: int) -> list[np.ndarray]:
    """Average records keep the time column but store no observation times."""
    return [np.empty(0, dtype=np.float32) for _ in range(n_samples)]


def generate_deterioration_frame(
    rng: np.random.Generator,
    n_samples: int,
    *,
    missing_rate: float,
    tabular_dim: int,
    ts_dim: int,
    embedding_dims: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, list[int]], pd.DataFrame]:
    """Return one fictional ED encounter per patient, raw observation counts and demographics.

    The tabular vector holds the base features followed by the measurement summaries.
    """
    names = ("acuity", "frailty", "tab_factor", "ecg_factor", "cxr_factor", "ts_factor")
    factors = {name: rng.normal(size=n_samples) for name in names}
    frailty = factors["frailty"]
    demographics = _draw_demographics(rng, frailty)
    rates = _missing_rates(demographics, missing_rate)
    # The whole record, including what is missing, follows the recorded factors. Only the
    # outcome uses their true values.
    recorded_acuity, recorded_frailty = _measured_factors(rng, factors, demographics)

    imaging_effect = rng.normal(size=n_samples)
    ecg_present = ~_absence(rng, recorded_acuity, imaging_effect, rates)
    cxr_present = ~_absence(
        rng, recorded_acuity, imaging_effect, np.minimum(MAX_MISSING_RATE, 1.5 * rates)
    )
    measured = ~_absence(rng, recorded_acuity, np.zeros(n_samples), rates / 6.0)

    def latents(factor: str) -> np.ndarray:
        return np.column_stack([recorded_acuity, recorded_frailty, factors[factor]])

    base, base_missing = _tabular_block(
        rng, latents("tab_factor"), recorded_acuity, width=tabular_dim, missing_rate=rates
    )
    summaries, summary_missing, ts_counts = _summarised_time_series(
        rng, latents("ts_factor"), recorded_acuity, measured, n_measurements=ts_dim,
        missing_rate=missing_rate, single_observation_trend_missing=False,
    )
    ecg_feats, ecg_missing, ecg_counts = _averaged_embeddings(
        rng, latents("ecg_factor"), ecg_present,
        width=embedding_dims["ecg"], max_observations=MAX_OBSERVATIONS["ecg"],
    )
    cxr_feats, cxr_missing, cxr_counts = _averaged_embeddings(
        rng, latents("cxr_factor"), cxr_present,
        width=embedding_dims["cxr"], max_observations=MAX_OBSERVATIONS["cxr"],
    )

    frame = pd.DataFrame(
        {
            **_identifiers("deterioration", n_samples),
            OUTCOMES["deterioration"]: _draw_outcome(rng, "deterioration", factors),
            "tab_feats": list(np.hstack([base, summaries])),
            "tab_missing": list(np.hstack([base_missing, summary_missing])),
            "ecg_feats": ecg_feats,
            "ecg_time": _no_times(n_samples),
            "ecg_missing": np.asarray(ecg_missing, dtype=np.int8),
            "cxr_feats": cxr_feats,
            "cxr_time": _no_times(n_samples),
            "cxr_missing": np.asarray(cxr_missing, dtype=np.int8),
        }
    )
    counts = {"tabular": ts_counts, "ecg": ecg_counts, "cxr": cxr_counts}
    return frame, counts, demographics


def generate_hospitalisation_frame(
    rng: np.random.Generator,
    n_samples: int,
    *,
    missing_rate: float,
    tabular_dim: int,
    ts_dim: int,
    embedding_dims: Mapping[str, int],
) -> tuple[pd.DataFrame, dict[str, list[int]], pd.DataFrame]:
    """Return one fictional current admission per patient, observation counts and demographics."""
    names = ("acuity", "frailty", "tab_factor", "ecg_factor", "cxr_factor", "notes_factor",
             "ts_factor")
    factors = {name: rng.normal(size=n_samples) for name in names}
    frailty = factors["frailty"]
    demographics = _draw_demographics(rng, frailty)
    rates = _missing_rates(demographics, missing_rate)
    # The whole record, including what is missing, follows the recorded factors. Only the
    # outcome uses their true values.
    recorded_acuity, recorded_frailty = _measured_factors(rng, factors, demographics)

    # Frailer patients are more likely to have a previous admission
    has_previous = rng.random(n_samples) < 0.35 + 0.4 * _sigmoid(recorded_frailty)
    imaging_effect = rng.normal(size=n_samples)
    cxr_rates = np.minimum(MAX_MISSING_RATE, 1.5 * rates)
    present = {
        "ts_prev": has_previous,
        "ts_curr": ~_absence(rng, recorded_acuity, np.zeros(n_samples), rates / 6.0),
        "ecg_curr": ~_absence(rng, recorded_acuity, imaging_effect, rates),
        "cxr_prev": has_previous & ~_absence(rng, recorded_acuity, imaging_effect, cxr_rates),
        "cxr_curr": ~_absence(rng, recorded_acuity, imaging_effect, cxr_rates),
        "notes_prev": has_previous & (rng.random(n_samples) >= rates / 3.0),
    }

    def latents(factor: str) -> np.ndarray:
        return np.column_stack([recorded_acuity, recorded_frailty, factors[factor]])

    tab_feats, tab_missing = _tabular_block(
        rng, latents("tab_factor"), recorded_acuity, width=tabular_dim, missing_rate=rates
    )
    columns: dict[str, object] = {
        **_identifiers("hospitalisation", n_samples),
        OUTCOMES["hospitalisation"]: _draw_outcome(rng, "hospitalisation", factors),
        "tab_feats": list(tab_feats),
        "tab_missing": list(tab_missing),
    }
    counts = {}
    for spec in parse_modality_specs(list(HOSPITALISATION_MODALITIES)):
        factor_latents = latents(f"{spec.kind}_factor")
        if spec.per_step_mask:
            summaries, masks, counts[spec.name] = _summarised_time_series(
                rng, factor_latents, recorded_acuity, present[spec.name], n_measurements=ts_dim,
                missing_rate=missing_rate, single_observation_trend_missing=True,
            )
            # One summary row per present patient, with an element-wise mask of the same shape
            feats = [[row] if is_present else [] for row, is_present in
                     zip(summaries, present[spec.name], strict=True)]
            missing = [[row] if is_present else [] for row, is_present in
                       zip(masks, present[spec.name], strict=True)]
        else:
            feats, flags, counts[spec.name] = _averaged_embeddings(
                rng, factor_latents, present[spec.name],
                width=embedding_dims[spec.kind], max_observations=MAX_OBSERVATIONS[spec.kind],
            )
            missing = np.asarray(flags, dtype=np.int8)
        columns[f"{spec.name}_feats"] = feats
        columns[f"{spec.name}_missing"] = missing
        columns[f"{spec.name}_time"] = _no_times(n_samples)
    return pd.DataFrame(columns), counts, demographics


def split_frame(frame: pd.DataFrame, outcome: str, seed: int) -> dict[str, pd.DataFrame]:
    """Split patients 80/10/10, stratified by outcome, with no patient in two splits."""
    assigned = frame[["subject_id"]].copy()
    assigned["split"] = assign_shared_split(
        frame, frame[outcome].astype(str).rename("stratum"), seed=seed
    ).to_numpy()
    split_tables = {
        split: frame.loc[assigned["split"].to_numpy() == split].reset_index(drop=True)
        for split in SPLIT_NAMES
    }
    validate_split_invariants(assigned, split_tables)
    for split, table in split_tables.items():
        if set(table[outcome]) != {0, 1}:
            raise ValueError(
                f"The {split} split does not contain both outcome classes. "
                "Increase --n-samples."
            )
        absent = [
            column.removesuffix("_feats")
            for column in table.columns
            if column.endswith("_feats")
            and column != "tab_feats"
            and not table[column].map(len).gt(0).any()
        ]
        if absent:
            raise ValueError(
                f"The {split} split has no observations for modalities {absent}. "
                "Increase --n-samples or lower --missing-rate."
            )
    return split_tables


def _count_stats(counts: Sequence[int]) -> dict[str, float]:
    observed = np.asarray([count for count in counts if count > 0], dtype=np.int64)
    if observed.size == 0:
        return {"min": 0, "median": 0.0, "max": 0, "mean": 0.0}
    return {
        "min": int(observed.min()),
        "median": float(np.median(observed)),
        "max": int(observed.max()),
        "mean": round(float(observed.mean()), 4),
    }


def demographics_table(
    task: str, frame: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    """Lay out demographics in the pathway's source format, as ``fairness --static`` reads it."""
    ethnicity = demographics["ethnicity"].map(ETHNICITY_LABELS[task])
    common = {
        "subject_id": frame["subject_id"],
        "demographics_gender": demographics["gender"],
        "demographics_age": demographics["age"],
    }
    if task == "deterioration":
        return pd.DataFrame(
            {"general_ed_stay_id": frame["ed_stay_id"], **common, "general_race": ethnicity}
        )
    return pd.DataFrame(
        {
            "last_hadm_id": frame["ed_hadm_id"],
            **common,
            "demographics_ethnicity": ethnicity,
            "demographics_insurance": demographics["insurance"],
        }
    )


def _group_summary(
    task: str, frame: pd.DataFrame, demographics: pd.DataFrame
) -> dict[str, dict[str, dict[str, float]]]:
    """Count records and measure prevalence and optional-modality availability per group."""
    optional = [modality for modality in MODALITIES[task] if modality not in TABULAR_MODALITIES]
    availability = np.mean(
        [_frame_modality_available(frame, modality) for modality in optional], axis=0
    )
    groups = {
        "sex": pd.Categorical(demographics["gender"].map({0: "Female", 1: "Male"})),
        "age": pd.cut(demographics["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False),
        "ethnicity": pd.Categorical(demographics["ethnicity"], categories=list(ETHNICITY_SHARES)),
    }
    if task == "hospitalisation":
        groups["insurance"] = pd.Categorical(demographics["insurance"])
    summary = {}
    for attribute, values in groups.items():
        table = pd.DataFrame(
            {"group": values, "outcome": frame[OUTCOMES[task]], "available": availability}
        )
        summary[attribute] = {
            str(group): {
                "count": int(len(rows)),
                "outcome_prevalence": round(float(rows["outcome"].mean()), 4),
                "optional_modality_availability": round(float(rows["available"].mean()), 4),
            }
            for group, rows in table.groupby("group", observed=True)
        }
    return summary


def build_manifest(
    task: str,
    frame: pd.DataFrame,
    split_tables: dict[str, pd.DataFrame],
    raw_counts: dict[str, list[int]],
    parameters: dict[str, object],
    demographics: pd.DataFrame,
) -> dict[str, object]:
    """Describe one generated pathway without any machine-specific paths."""
    outcome = OUTCOMES[task]
    modalities = MODALITIES[task]
    feature_dims = {}
    for modality in modalities:
        values = frame[f"{modality_prefix(modality)}_feats"]
        first = next(value for value in values if len(value) > 0)
        feature_dims[modality] = int(len(first) if modality in TABULAR_MODALITIES else len(first[0]))
    summary_names = measurement_feature_names(int(parameters["ts_dim"]))
    if task == "deterioration":
        base_names = [f"base_{index:02d}" for index in range(int(parameters["tabular_dim"]))]
        feature_names = {"tabular": base_names + summary_names}
        summary_masks = np.stack(frame["tab_missing"].to_list())[:, len(base_names):]
    else:
        feature_names = {
            "static": [f"base_{index:02d}" for index in range(int(parameters["tabular_dim"]))],
            "ts_prev": summary_names,
            "ts_curr": summary_names,
        }
        summary_masks = np.stack(
            [mask[0] for column in ("ts_prev_missing", "ts_curr_missing")
             for mask in frame[column] if len(mask) > 0]
        )
    return {
        "notice": ARTIFICIAL_DATA_NOTICE,
        "generator_version": GENERATOR_VERSION,
        "record_layout": RECORD_LAYOUT,
        "task": task,
        "seed": parameters["seed"],
        "parameters": parameters,
        "outcome": outcome,
        "modalities": list(modalities),
        "feature_dims": feature_dims,
        "feature_names": feature_names,
        "summary_statistics": list(TS_STAT_NAMES),
        "files": {split: f"{split}_synthetic.parquet" for split in SPLIT_NAMES},
        "split_counts": {split: len(table) for split, table in split_tables.items()},
        "outcome_prevalence": {
            split: round(float(table[outcome].mean()), 4)
            for split, table in split_tables.items()
        },
        "modality_availability": {
            modality: round(float(_frame_modality_available(frame, modality).mean()), 4)
            for modality in modalities
        },
        "tabular_feature_missing_rate": round(
            float(np.mean(np.stack(frame["tab_missing"].to_list()))), 4
        ),
        "summary_statistic_missing_rate": round(float(summary_masks.mean()), 4),
        "raw_observations_per_record": {
            modality: _count_stats(counts) for modality, counts in raw_counts.items()
        },
        "observation_windows_minutes": (
            {"ed": [0.0, DETERIORATION_WINDOW_MINUTES]}
            if task == "deterioration"
            else {
                "previous": list(PREVIOUS_WINDOW_RANGE_MINUTES),
                "current": list(CURRENT_WINDOW_RANGE_MINUTES),
            }
        ),
        "outcome_coefficients": {
            **OUTCOME_COEFFICIENTS[task], "noise_sd": OUTCOME_NOISE_SD,
        },
        "signal_description": SIGNAL_DESCRIPTION,
        "missingness_description": MISSINGNESS_DESCRIPTION,
        "demographics_file": DEMOGRAPHICS_FILENAME,
        "demographic_bias": {
            attribute: description
            for attribute, description in DEMOGRAPHIC_BIAS.items()
            if not (attribute == "insurance" and task == "deterioration")
            and not (attribute == "cxr_availability" and task == "hospitalisation")
        },
        "demographic_groups": _group_summary(task, frame, demographics),
    }


def _write_deterioration_parquet(frame: pd.DataFrame, path: Path) -> None:
    schema = build_deterioration_schema(
        [OUTCOMES["deterioration"]], use_sequence_tabular=False, include_ts=False
    )
    schema = schema.insert(2, pa.field("subject_id", pa.int64()))
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _check_no_other_splits(output_dir: Path, tasks: Sequence[str]) -> None:
    """Stop before writing beside other split files.

    Earlier synthetic files are replaced, but resolve_split_path expects exactly one parquet per
    split in each folder.
    """
    for task in tasks:
        foreign = [
            path
            for split in SPLIT_NAMES
            for path in sorted((output_dir / task).glob(f"{split}_*.parquet"))
            if path.name != f"{split}_synthetic.parquet"
        ]
        if foreign:
            raise FileExistsError(
                f"{output_dir / task} holds other split parquet files {foreign}. "
                "Choose an empty --output-dir."
            )


def generate(
    tasks: Sequence[str] = TASKS,
    *,
    n_samples: int = 1000,
    seed: int = 42,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    missing_rate: float = 0.3,
    tabular_dim: int = 16,
    ts_dim: int = 8,
    ecg_dim: int = EMBEDDING_DIMS["ecg"],
    cxr_dim: int = EMBEDDING_DIMS["cxr"],
    notes_dim: int = EMBEDDING_DIMS["notes"],
) -> dict[str, dict[str, object]]:
    """Generate, split and write each requested pathway, replacing earlier synthetic files.

    Returns each pathway's manifest.
    """
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Choose from {TASKS}.")
    output_dir = Path(output_dir)
    _check_no_other_splits(output_dir, tasks)

    parameters = {
        "n_samples": n_samples, "seed": seed, "missing_rate": missing_rate,
        "tabular_dim": tabular_dim, "ts_dim": ts_dim,
        "ecg_dim": ecg_dim, "cxr_dim": cxr_dim, "notes_dim": notes_dim,
    }
    embedding_dims = {"ecg": ecg_dim, "cxr": cxr_dim, "notes": notes_dim}
    builders = {
        "deterioration": generate_deterioration_frame,
        "hospitalisation": generate_hospitalisation_frame,
    }
    manifests = {}
    for task in tasks:
        # One stream per task keeps each pathway identical whether it is generated alone or both
        rng = np.random.default_rng(np.random.SeedSequence([seed, TASKS.index(task)]))
        frame, raw_counts, demographics = builders[task](
            rng, n_samples, missing_rate=missing_rate, tabular_dim=tabular_dim,
            ts_dim=ts_dim, embedding_dims=embedding_dims,
        )
        split_tables = split_frame(frame, OUTCOMES[task], seed)

        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for split, table in split_tables.items():
            path = task_dir / f"{split}_synthetic.parquet"
            if task == "deterioration":
                _write_deterioration_parquet(table, path)
            else:
                write_hospitalisation_parquet(
                    table, path, [OUTCOMES[task]],
                    parse_modality_specs(list(HOSPITALISATION_MODALITIES)),
                )

        demographics_table(task, frame, demographics).to_csv(
            task_dir / DEMOGRAPHICS_FILENAME, index=False
        )
        manifest = build_manifest(
            task, frame, split_tables, raw_counts, parameters, demographics
        )
        (task_dir / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # The tabular feature names, under the key the real split summaries use
        tabular_modality = MODALITIES[task][0]
        split_summary = {
            "notice": ARTIFICIAL_DATA_NOTICE,
            "tab_feats_columns": manifest["feature_names"][tabular_modality],
        }
        (task_dir / SPLIT_SUMMARY_FILENAME).write_text(
            json.dumps(split_summary, indent=2) + "\n", encoding="utf-8"
        )
        manifests[task] = manifest
    return manifests


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the requested synthetic pathways and print a short summary."""
    args = parse_args(argv)
    tasks = TASKS if args.task == "both" else (args.task,)
    try:
        manifests = generate(
            tasks, n_samples=args.n_samples, seed=args.seed, output_dir=args.output_dir,
            missing_rate=args.missing_rate, tabular_dim=args.tabular_dim, ts_dim=args.ts_dim,
            ecg_dim=args.ecg_dim, cxr_dim=args.cxr_dim, notes_dim=args.notes_dim,
        )
    except FileExistsError as error:
        raise SystemExit(str(error)) from None
    print(ARTIFICIAL_DATA_NOTICE)
    for task, manifest in manifests.items():
        counts = ", ".join(f"{split} {count}" for split, count in manifest["split_counts"].items())
        print(f"{task}: {counts} -> {args.output_dir / task}")


if __name__ == "__main__":
    main()
