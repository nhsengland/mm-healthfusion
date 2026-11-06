"""Shared bundle preparation and splitting helpers.

Keep cohort selection and pipeline-specific sequence conversion in the calling scripts.
Encoding and normalisation statistics must be fitted using training rows only.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

EmbeddingFormat: TypeAlias = Literal["pickle", "parquet"]
TS_STAT_NAMES = [
    "mean",
    "median",
    "min",
    "max",
    "std",
    "first",
    "last",
    "rate_change",
    "coeff",
]


def resolve_bundle_dir(bundle: Path, default_parent: Path) -> Path:
    """Use an absolute or existing path as given, otherwise look the name up in default_parent."""
    bundle_dir = bundle if bundle.is_absolute() or bundle.exists() else default_parent / bundle
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Bundle folder not found: {bundle_dir}. Pass the path to the bundle folder or a "
            f"bundle name under {default_parent}."
        )
    return bundle_dir


def normalise_nullable_integer_column(data: pd.DataFrame, column: str) -> None:
    """Store an identifier column as a nullable integer."""
    numeric_values = pd.to_numeric(data[column], errors="coerce")
    finite_values = numeric_values.dropna()
    if not finite_values.mod(1).eq(0).all():
        raise ValueError(f"{column} contains non-integer identifier values.")
    data[column] = numeric_values.astype("Int64")


def aggregate_cxr_studies(cxr_keys: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["subject_id", "study_id", "dicom_id", "path", "ViewPosition"]
    if "cxr_time" in cxr_keys.columns:
        cxr = cxr_keys[[*required_columns, "cxr_time"]].copy()
        cxr["cxr_time"] = pd.to_datetime(cxr["cxr_time"], errors="coerce")
    else:
        cxr = cxr_keys[[*required_columns, "StudyDate", "StudyTime"]].copy()
        study_date = cxr["StudyDate"].astype(str).str.split(".").str[0].str.zfill(8)
        study_time = cxr["StudyTime"].astype(str).str.split(".", n=1, expand=True)
        time_base = study_time[0].str.zfill(6)
        time_fraction = (
            study_time[1]
            if 1 in study_time.columns
            else pd.Series("0", index=cxr.index)
        )
        cxr["cxr_time"] = pd.to_datetime(
            study_date + time_base + "." + time_fraction.fillna("0"),
            format="%Y%m%d%H%M%S.%f",
            errors="coerce",
        ).dt.round("s")
    if cxr["cxr_time"].isna().any():
        raise ValueError("CXR keys contain rows with unparseable cxr_time values.")
    cxr["cxr_key_row_id"] = cxr.index

    return (
        cxr.sort_values(["subject_id", "study_id", "cxr_time", "cxr_key_row_id"])
        .groupby(["subject_id", "study_id", "cxr_time"], as_index=False)
        .agg(
            cxr_key_row_ids=("cxr_key_row_id", join_selected_values),
            cxr_first_key_row_id=("cxr_key_row_id", "min"),
            cxr_dicom_ids=("dicom_id", join_selected_values),
            cxr_image_times=("cxr_time", join_selected_values),
            cxr_paths=("path", join_selected_values),
            cxr_view_positions=(
                "ViewPosition",
                lambda values: "|".join(values.fillna("").astype(str)),
            ),
            n_cxr_images=("dicom_id", "nunique"),
        )
        .rename(columns={"study_id": "cxr_study_id"})
    )


def join_selected_values(values: pd.Series) -> str:
    """Join selected values in their existing order."""
    return "|".join(values.astype(str))


def get_feature_columns(embeddings: pd.DataFrame, prefix: str) -> list[str]:
    """Return ordered numeric feature columns for one modality."""
    feature_columns = [
        column for column in embeddings.columns if column.startswith(prefix)
    ]
    if not feature_columns:
        raise ValueError(
            f"Embeddings contain no feature columns with prefix '{prefix}'."
        )
    invalid_columns = [
        column
        for column in feature_columns
        if not column.removeprefix(prefix).isdigit()
    ]
    if invalid_columns:
        raise ValueError(
            f"Embeddings contain invalid feature columns for prefix '{prefix}': "
            f"{invalid_columns}."
        )
    return sorted(feature_columns, key=lambda column: int(column.removeprefix(prefix)))


def parse_row_ids(value: object, column: str, output_row: int) -> list[int]:
    """Parse one or more pipe-separated source row numbers."""
    if pd.isna(value):
        raise ValueError(f"{column} is missing at output row {output_row}.")
    row_ids: list[int] = []
    for token in str(value).split("|"):
        try:
            number = float(token)
            row_id = int(number)
        except ValueError as exc:
            raise ValueError(
                f"{column} contains an invalid row identifier '{token}' "
                f"at output row {output_row}."
            ) from exc
        if not number.is_integer() or row_id < 0:
            raise ValueError(
                f"{column} contains an invalid row identifier '{token}' "
                f"at output row {output_row}."
            )
        row_ids.append(row_id)
    return row_ids


def validate_row_ids(row_ids: list[int], source_rows: int, column: str) -> None:
    """Check that referenced source rows exist."""
    invalid_ids = [row_id for row_id in row_ids if row_id >= source_rows]
    if invalid_ids:
        raise ValueError(
            f"{column} references row identifiers outside the source embeddings: "
            f"{invalid_ids[:5]}."
        )


def identifier_matches(actual: object, expected: object) -> bool:
    """Compare identifiers after normalising numeric values."""
    if pd.isna(actual) and pd.isna(expected):
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def to_sequence_array(sequence_features: list[np.ndarray]) -> np.ndarray:
    """Return an array of embedding sequences aligned with the key table."""
    embeddings = np.empty(len(sequence_features), dtype=object)
    embeddings[:] = sequence_features
    return embeddings


def get_ecg_feature_columns(source_embeddings: pd.DataFrame) -> list[str]:
    """Return ECG embedding columns for a supported source schema."""
    for prefix in ("ecgfm_emb_", "ecg_ft"):
        try:
            return get_feature_columns(source_embeddings, prefix)
        except ValueError:
            continue
    raise ValueError(
        "No ECG embedding columns found. Expected columns matching either "
        "'ecgfm_emb_###' or legacy 'ecg_ft###' naming."
    )


def validate_embedding_array(name: str, embeddings: np.ndarray) -> None:
    """Check that an embedding file contains aligned sequence arrays."""
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 1:
        raise ValueError(f"{name} embeddings must be a one-dimensional NumPy array.")
    if not all(
        isinstance(sequence, np.ndarray) and sequence.ndim == 2
        for sequence in embeddings
    ):
        raise ValueError(f"{name} embeddings must contain two-dimensional arrays.")


def write_embedding_parquet(
    embeddings: np.ndarray, output_path: Path, column_name: str
) -> None:
    """Write sequence embeddings as one nested parquet column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    value_type = pa.float32()
    for sequence in embeddings:
        value_type = pa.from_numpy_dtype(sequence.dtype)
        break
    feature_dim = embeddings[0].shape[1] if len(embeddings) else 0
    sequence_type = pa.list_(pa.list_(value_type))
    table = pa.table(
        {
            column_name: pa.array(
                [sequence.tolist() for sequence in embeddings], type=sequence_type
            )
        }
    ).replace_schema_metadata(
        {
            "embedding_column": column_name,
            "embedding_feature_dim": str(feature_dim),
        }
    )
    pq.write_table(table, output_path)


def write_embedding_file(
    embeddings: np.ndarray,
    output_path: Path,
    column_name: str,
    embedding_format: EmbeddingFormat,
) -> None:
    """Write embeddings in the requested format."""
    if embedding_format == "pickle":
        pd.to_pickle(embeddings, output_path)
        return
    write_embedding_parquet(embeddings, output_path, column_name)


def validate_unique_keys(
    data: pd.DataFrame, key_columns: list[str], table_name: str
) -> None:
    duplicates = data.loc[data.duplicated(key_columns, keep=False), key_columns]
    if duplicates.empty:
        return
    examples = duplicates.drop_duplicates().head(5).to_dict(orient="records")
    raise ValueError(
        f"{table_name} has duplicate keys for {key_columns}. "
        f"Example duplicate keys: {examples}"
    )


def fit_ohe_columns(
    train_frame: pd.DataFrame,
    categorical_columns: list[str],
) -> list[str]:
    if not categorical_columns:
        return []
    train_ohe = pd.get_dummies(
        train_frame[categorical_columns],
        columns=categorical_columns,
        prefix=categorical_columns,
        prefix_sep="=",
        dummy_na=False,
        dtype="int8",
    )
    return train_ohe.columns.tolist()


def apply_ohe_to_frame(
    split_frame: pd.DataFrame,
    categorical_columns: list[str],
    ohe_columns: list[str],
) -> pd.DataFrame:
    if not categorical_columns:
        return split_frame
    ohe = pd.get_dummies(
        split_frame[categorical_columns],
        columns=categorical_columns,
        prefix=categorical_columns,
        prefix_sep="=",
        dummy_na=False,
        dtype="int8",
    )
    ohe = ohe.reindex(columns=ohe_columns, fill_value=0)
    remaining = split_frame.drop(columns=categorical_columns)
    return pd.concat([remaining, ohe], axis=1)


def apply_tabular_normalisation(
    split_frame: pd.DataFrame,
    normalisation_stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    split_frame = split_frame.copy()
    for column, (min_value, max_value) in normalisation_stats.items():
        range_value = max_value - min_value
        if pd.isna(range_value) or range_value == 0:
            values = split_frame[column]
            split_frame[column] = np.where(values.notna(), 0.0, np.nan).astype(
                "float32"
            )
            continue
        split_frame[column] = ((split_frame[column] - min_value) / range_value).astype(
            "float32"
        )
    return split_frame


def fit_embedding_normalisation_stats(
    split_keys: pd.DataFrame,
    embeddings: np.ndarray,
    sequence_min_max: Callable[
        [list[np.ndarray]], tuple[np.ndarray | None, np.ndarray | None]
    ],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    row_indices = split_keys["_source_row"].to_numpy(dtype=int)
    return sequence_min_max(list(embeddings[row_indices]))


def parse_time_values(raw_value: object, column_name: str) -> list[pd.Timestamp]:
    if pd.isna(raw_value):
        return []
    values = [value.strip() for value in str(raw_value).split("|") if value.strip()]
    if not values:
        return []
    timestamps = pd.to_datetime(
        values,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    if pd.isna(timestamps).any():
        timestamps = pd.to_datetime(values, errors="coerce")
    if pd.isna(timestamps).any():
        raise ValueError(f"{column_name} contains unparseable datetime values.")
    return list(timestamps)
