from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy.interpolate import interp1d


class ECGGuardSkip(ValueError):
    """Mark an ECG that cannot be processed safely"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "MIMIC-IV"
DEFAULT_ECG_ROOT = (
    DEFAULT_DATA_ROOT
    / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "ecg_ecgfm"

ECGFM_REPO_ID = "wanglab/ecg-fm"
ECGFM_FILENAME = "mimic_iv_ecg_physionet_pretrained.pt"
ECGFM_REVISION = "abf3d8c789cd5782ae171511b43dda73c0e8f6b5"

ECGFM_SAMPLE_RATE_HZ = 500
ECGFM_SEGMENT_SECONDS = 5.0
TEST_RECORD_LIMIT = 100
PROGRESS_EVERY_RECORDS = 50
ECG_FM_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4",
                     "V5", "V6"]
ECG_KEY_COLUMNS = [
    "subject_id", "study_id", "ecg_time", "path", "sample_rate_hz", "n_samples",
    "preprocessed_sample_rate_hz", "preprocessed_n_samples", "segment_seconds",
    "segment_samples", "n_segments",
]
ECG_EMBEDDING_PICKLE_EXCLUDE_COLUMNS = {
    "file_name", "sample_rate_hz", "n_samples", "preprocessed_sample_rate_hz",
    "preprocessed_n_samples", "segment_seconds", "segment_samples", "n_segments",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ECG-FM embeddings for every ECG record listed in the "
            "MIMIC-IV ECG matched subset record_list.csv."
        )
    )
    parser.add_argument(
        "--ecg-root",
        type=Path,
        default=DEFAULT_ECG_ROOT,
        help=(
            "Input MIMIC-IV ECG matched subset folder containing record_list.csv "
            "and waveform files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where extracted embedding artifacts should be written.",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        action="store_true",
        default=False,
        help=f"Process only the first {TEST_RECORD_LIMIT} records from record_list.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of 5-second ECG segments per ECG-FM forward pass.",
    )
    parser.add_argument(
        "--records-per-chunk",
        "-r",
        type=int,
        default=50000,
        help="Number of ECG records to preprocess/infer before offloading chunk outputs.",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        default=False,
        help="Keep temporary chunk output files after final outputs are built.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume a partial run: keep existing chunk outputs and skip any chunk whose "
            "_embeddings.pkl and _keys.csv already exist, recomputing only the missing "
            "chunks before the final concatenation. Requires the same --records-per-chunk "
            "(and same record_list.csv) as the interrupted run, since chunk boundaries are "
            "positional. Implies --keep-chunks for the reused chunks during the run."
        ),
    )
    return parser.parse_args()


def parse_header(record_base: Path) -> dict[str, Any]:
    """Parse a MIMIC-IV-ECG header file and return sampling metadata."""
    header_path = record_base.with_suffix(".hea")
    with header_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    record_fields = lines[0].split()
    n_leads = int(record_fields[1])
    sample_rate = int(record_fields[2])
    n_samples = int(record_fields[3])

    gains: list[float] = []
    baselines: list[float] = []
    lead_names: list[str] = []

    for line in lines[1 : 1 + n_leads]:
        parts = line.split()
        gain_field = parts[2].split("/")[0]
        if "(" in gain_field:
            gain_str, baseline_str = gain_field.split("(")
            baseline_str = baseline_str.rstrip(")")
        else:
            gain_str = gain_field
            baseline_str = "0"
        gain = float(gain_str)
        gains.append(gain if gain != 0 else 1.0)
        baselines.append(float(baseline_str))
        lead_names.append(parts[-1])

    return {
        "n_leads": n_leads,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
        "gains": np.asarray(gains, dtype=np.float32),
        "baselines": np.asarray(baselines, dtype=np.float32),
        "lead_names": lead_names,
    }


def load_ecg_waveform(record_base: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ECG record and convert digital values to mV."""
    header = parse_header(record_base)
    dat_path = record_base.with_suffix(".dat")
    raw = np.fromfile(dat_path, dtype="<i2")
    expected_size = header["n_samples"] * header["n_leads"]
    if raw.size != expected_size:
        raise ValueError(
            f"Unexpected waveform size for {record_base}: {raw.size} != {expected_size}"
        )

    waveform = raw.reshape(header["n_samples"], header["n_leads"]).astype(np.float32)
    waveform_mv = (waveform - header["baselines"]) / header["gains"]
    # Reject invalid waveform values so the record is recorded as failed
    if not np.isfinite(waveform_mv).all():
        raise ECGGuardSkip(
            "nan", f"Non-finite (NaN/Inf) waveform values for {record_base}"
        )
    return waveform_mv, header


def preprocess_for_ecgfm(
    waveform_mv: np.ndarray,
    lead_names: list[str],
    sample_rate: int,
    target_sample_rate: int,
    segment_seconds: float,
    lead_order: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare ECG-FM paper-style 5-second segment inputs.

    Returns segments as [segments, leads, time] plus preprocessing metadata.
    """
    lead_order = lead_order or ECG_FM_LEAD_ORDER
    waveform_df = pd.DataFrame(waveform_mv, columns=lead_names)
    missing_leads = [lead for lead in lead_order if lead not in waveform_df.columns]
    if missing_leads:
        raise ValueError(f"Missing ECG leads for ECG-FM preprocessing: {missing_leads}")

    waveform = waveform_df.loc[:, lead_order].to_numpy(dtype=np.float32).T

    if sample_rate != target_sample_rate:
        target_n_samples = (
            int(round((waveform.shape[1] - 1) * (target_sample_rate / sample_rate))) + 1
        )
        if target_n_samples <= 0:
            raise ValueError(
                f"Invalid target sample count after resampling: {target_n_samples}"
            )
        source_time = np.arange(waveform.shape[1], dtype=np.float32) / float(sample_rate)
        target_time = np.linspace(0.0, source_time[-1], target_n_samples)
        waveform = interp1d(source_time, waveform, kind="linear", axis=1)(
            target_time
        ).astype(np.float32)

    lead_std = waveform.std(axis=1, keepdims=True)
    constant_mask = lead_std < 1e-6
    waveform = waveform - waveform.mean(axis=1, keepdims=True)
    waveform = waveform / np.where(constant_mask, 1.0, lead_std)
    waveform = np.where(constant_mask, 0.0, waveform)

    segment_samples = int(segment_seconds * target_sample_rate)
    if segment_samples <= 0:
        raise ValueError(f"Invalid segment size: {segment_samples}")

    n_segments = waveform.shape[1] // segment_samples
    if n_segments == 0:
        raise ValueError(
            f"ECG has {waveform.shape[1]} samples after preprocessing, "
            f"which is shorter than one {segment_samples}-sample segment."
        )

    trimmed = waveform[:, : n_segments * segment_samples]
    segments = trimmed.reshape(len(lead_order), n_segments, segment_samples)
    segments = np.moveaxis(segments, 1, 0).astype(np.float32)
    metadata = {
        "preprocessed_sample_rate_hz": int(target_sample_rate),
        "preprocessed_n_samples": int(waveform.shape[1]),
        "segment_seconds": float(segment_seconds),
        "segment_samples": int(segment_samples),
        "n_segments": int(n_segments),
    }
    return segments, metadata


def prepare_record_keys(records: pd.DataFrame) -> pd.DataFrame:
    """Prepare ECG record metadata for embedding and downstream joins."""
    records = records.copy().reset_index(drop=True)
    if "path" not in records.columns:
        raise KeyError("record_list.csv must contain a 'path' column.")

    if "ecg_time" in records.columns:
        records["ecg_time"] = pd.to_datetime(records["ecg_time"], errors="coerce")

    return records


def encoder_out_to_embedding(encoder_out: torch.Tensor) -> torch.Tensor:
    """Pool ECG-FM encoder outputs across the token dimension."""
    if encoder_out.ndim == 2:
        return encoder_out
    non_zero = (encoder_out != 0).sum(dim=1).clamp(min=1)
    return encoder_out.sum(dim=1) / non_zero


def load_ecgfm_model(checkpoint_path: Path, device: str):
    """Load ECG-FM from a fairseq-signals checkpoint."""
    from fairseq_signals.models import build_model_from_checkpoint

    model = build_model_from_checkpoint(checkpoint_path=str(checkpoint_path))
    model.eval()
    model.to(device)
    return model


def failed_record_row(row: pd.Series, error: Exception) -> dict[str, Any]:
    """Build a compact failed-record row from record_list.csv keys."""
    failed = {
        column: row[column]
        for column in ["subject_id", "study_id", "ecg_time", "path"]
        if column in row.index
    }
    failed.update(
        {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "guard_reason": error.reason if isinstance(error, ECGGuardSkip) else None,
        }
    )
    return failed


def infer_segment_embeddings(
    model: torch.nn.Module,
    segment_inputs: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Run ECG-FM over fixed-length 5-second segment batches."""
    segment_embedding_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start_idx in range(0, len(segment_inputs), batch_size):
            batch_np = segment_inputs[start_idx : start_idx + batch_size]
            batch = torch.from_numpy(batch_np).float().to(device)
            # Use the unmasked transformer output for stable embeddings
            model_output = model(source=batch, mask=False, features_only=True)
            encoder_out = model_output["x"]
            batch_embeddings = encoder_out_to_embedding(encoder_out)
            segment_embedding_chunks.append(
                batch_embeddings.detach().cpu().numpy().astype(np.float32)
            )
            del batch, model_output, encoder_out, batch_embeddings

    if not segment_embedding_chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(segment_embedding_chunks, axis=0)


def pool_segment_embeddings_by_record(
    segment_embeddings: np.ndarray,
    segment_record_indices: np.ndarray,
    record_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Mean-pool only segment embeddings that came from the same ECG record."""
    if len(segment_embeddings) != len(segment_record_indices):
        raise RuntimeError(
            "Aggregation invariant failed: segment embeddings and record-index "
            f"mapping lengths differ ({len(segment_embeddings)} != {len(segment_record_indices)})."
        )

    pooled_embeddings: list[np.ndarray] = []
    for record_idx, row in enumerate(record_rows):
        mask = segment_record_indices == record_idx
        n_segments = int(mask.sum())
        if n_segments == 0:
            raise RuntimeError(
                f"Aggregation invariant failed: record {record_idx} has no segments."
            )
        if n_segments != int(row["n_segments"]):
            raise RuntimeError(
                "Aggregation invariant failed: row metadata n_segments does not "
                f"match segment mapping for record {record_idx} ({row['n_segments']} != {n_segments})."
            )
        pooled_embeddings.append(segment_embeddings[mask].mean(axis=0).astype(np.float32))

    if len(pooled_embeddings) != len(record_rows):
        raise RuntimeError(
            "Aggregation invariant failed: pooled embedding count does not match "
            f"successful record count ({len(pooled_embeddings)} != {len(record_rows)})."
        )
    if not pooled_embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(pooled_embeddings, axis=0)


def process_record_chunk(
    model: torch.nn.Module,
    chunk_records: pd.DataFrame,
    ecg_root: Path,
    target_sample_rate: int,
    segment_seconds: float,
    batch_size: int,
    device: str,
    chunk_id: int,
    total_records: int,
    processed_before_chunk: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess, infer, and pool one chunk of ECG records."""
    segment_chunks: list[np.ndarray] = []
    segment_record_indices: list[int] = []
    record_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for local_pos, (_, row) in enumerate(chunk_records.iterrows(), start=1):
        global_pos = processed_before_chunk + local_pos
        try:
            record_base = ecg_root / row["path"]
            waveform_mv, header = load_ecg_waveform(record_base)
            segments, preprocess_metadata = preprocess_for_ecgfm(
                waveform_mv=waveform_mv,
                lead_names=header["lead_names"],
                sample_rate=header["sample_rate"],
                target_sample_rate=target_sample_rate,
                segment_seconds=segment_seconds,
            )

            record_idx = len(record_rows)
            segment_chunks.append(segments)
            segment_record_indices.extend([record_idx] * len(segments))

            row_metadata = row.to_dict()
            row_metadata.update(
                {
                    "sample_rate_hz": int(header["sample_rate"]),
                    "n_samples": int(header["n_samples"]),
                }
            )
            row_metadata.update(preprocess_metadata)
            record_rows.append(row_metadata)
        except Exception as exc:
            failed = failed_record_row(row, exc)
            failed["chunk_id"] = int(chunk_id)
            failed_rows.append(failed)
            if isinstance(exc, ECGGuardSkip):
                # Break the progress line so the guard notice is readable.
                sys.stdout.write(
                    f"\n  [guard:{exc.reason}] skipped {row.get('path', '?')}: {exc}\n"
                )
                sys.stdout.flush()

        if global_pos % PROGRESS_EVERY_RECORDS == 0 or global_pos == total_records:
            pct = global_pos / total_records * 100 if total_records else 100.0
            sys.stdout.write(
                f"\r  preprocessed {global_pos}/{total_records} records ({pct:.2f}%)"
            )
            sys.stdout.flush()

    if not record_rows:
        return pd.DataFrame(), pd.DataFrame(failed_rows)

    segment_inputs = np.concatenate(segment_chunks, axis=0).astype(np.float32)
    segment_record_indices_np = np.asarray(segment_record_indices, dtype=np.int64)
    segment_embeddings = infer_segment_embeddings(
        model=model,
        segment_inputs=segment_inputs,
        batch_size=batch_size,
        device=device,
    )
    record_embeddings = pool_segment_embeddings_by_record(
        segment_embeddings=segment_embeddings,
        segment_record_indices=segment_record_indices_np,
        record_rows=record_rows,
    )

    embedding_table = pd.DataFrame(record_rows)
    embedding_features = pd.DataFrame(
        record_embeddings,
        columns=[f"ecg_ft{i}" for i in range(record_embeddings.shape[1])],
        index=embedding_table.index,
    )
    embedding_table = pd.concat([embedding_table, embedding_features], axis=1)

    del segment_chunks, segment_inputs, segment_record_indices_np, segment_embeddings
    del record_embeddings, embedding_features
    if device == "cuda":
        torch.cuda.empty_cache()

    return embedding_table, pd.DataFrame(failed_rows)


def write_chunk_outputs(
    chunk_df: pd.DataFrame,
    chunk_dir: Path,
    chunk_id: int,
) -> tuple[Path, Path] | tuple[None, None]:
    """Write temporary chunk master/key files."""
    if chunk_df.empty:
        return None, None

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_master_path = chunk_dir / f"chunk_{chunk_id:05d}_embeddings.pkl"
    chunk_key_path = chunk_dir / f"chunk_{chunk_id:05d}_keys.csv"
    embedding_columns = [
        column
        for column in chunk_df.columns
        if column not in ECG_EMBEDDING_PICKLE_EXCLUDE_COLUMNS
    ]
    key_columns = [column for column in ECG_KEY_COLUMNS if column in chunk_df.columns]
    chunk_df.loc[:, embedding_columns].to_pickle(chunk_master_path)
    chunk_df.loc[:, key_columns].to_csv(chunk_key_path, index=False)
    return chunk_master_path, chunk_key_path


def concatenate_chunk_outputs(
    chunk_master_paths: list[Path],
    chunk_key_paths: list[Path],
    embedding_table_path: Path,
    key_table_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build final master and key outputs from temporary chunk files."""
    if chunk_master_paths:
        embedding_table = pd.concat(
            [pd.read_pickle(path) for path in chunk_master_paths],
            ignore_index=True,
        )
    else:
        embedding_table = pd.DataFrame()

    if chunk_key_paths:
        key_table = pd.concat(
            [pd.read_csv(path) for path in chunk_key_paths],
            ignore_index=True,
        )
        key_columns = key_table.columns.tolist()
        key_table.to_csv(key_table_csv_path, index=False)
    else:
        key_columns = ECG_KEY_COLUMNS
        key_table = pd.DataFrame(columns=key_columns)
        key_table.to_csv(key_table_csv_path, index=False)

    embedding_table.to_pickle(embedding_table_path)

    return embedding_table, key_table, key_columns


def extract_embeddings_chunked(
    args: argparse.Namespace,
    model: torch.nn.Module,
    records: pd.DataFrame,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    """Extract ECG-FM embeddings with chunk offload and final aggregation."""
    output_suffix = f"_test{TEST_RECORD_LIMIT}" if args.test_mode else ""
    embedding_table_path = args.output_dir / f"ecgfm_embeddings{output_suffix}.pkl"
    key_table_csv_path = args.output_dir / f"ecgfm_embedding_keys{output_suffix}.csv"
    metadata_path = args.output_dir / f"ecgfm_embedding_metadata{output_suffix}.json"
    failed_records_path = args.output_dir / f"ecgfm_failed_records{output_suffix}.csv"
    chunk_dir = args.output_dir / f"ecgfm_chunks{output_suffix}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if chunk_dir.exists() and not args.resume:
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def cleanup_chunks_after_error() -> None:
        # In resume mode, preserve completed chunks so the run can be resumed again
        if not args.keep_chunks and not args.resume and chunk_dir.exists():
            shutil.rmtree(chunk_dir)

    total_records = len(records)
    n_chunks = int(np.ceil(total_records / args.records_per_chunk)) if total_records else 0
    print(
        f"Processing {total_records} ECG records across {n_chunks} chunks "
        f"(records_per_chunk={args.records_per_chunk}, "
        f"batch_size={args.batch_size})..."
    )

    chunk_master_paths: list[Path] = []
    chunk_key_paths: list[Path] = []
    failed_chunks: list[pd.DataFrame] = []
    processed_before_chunk = 0

    try:
        for chunk_id, chunk_start in enumerate(
            range(0, total_records, args.records_per_chunk), start=1
        ):
            chunk_end = min(chunk_start + args.records_per_chunk, total_records)
            chunk_records = records.iloc[chunk_start:chunk_end]

            if args.resume:
                existing_master = chunk_dir / f"chunk_{chunk_id:05d}_embeddings.pkl"
                existing_key = chunk_dir / f"chunk_{chunk_id:05d}_keys.csv"
                if existing_master.exists() and existing_key.exists():
                    existing_key_df = pd.read_csv(existing_key)
                    existing_rows = int(len(existing_key_df))
                    if existing_rows > len(chunk_records):
                        raise ValueError(
                            f"Resume mismatch: {existing_key.name} has {existing_rows} rows "
                            f"but chunk {chunk_id} spans only {len(chunk_records)} records. "
                            f"This means --records-per-chunk ({args.records_per_chunk}) or "
                            f"record_list.csv differs from the interrupted run. Re-run with "
                            f"the original --records-per-chunk, or drop --resume to start over."
                        )

                    # Record missing rows so resume reports every input record
                    if (
                        existing_rows < len(chunk_records)
                        and "path" in existing_key_df.columns
                        and "path" in chunk_records.columns
                    ):
                        processed_paths = set(existing_key_df["path"].dropna().astype(str))
                        missing = chunk_records.loc[
                            ~chunk_records["path"].astype(str).isin(processed_paths)
                        ]
                        if not missing.empty:
                            failed_chunks.append(
                                pd.DataFrame(
                                    [
                                        failed_record_row(
                                            row,
                                            RuntimeError(
                                                "Missing from reused chunk outputs during --resume"
                                            ),
                                        )
                                        for _, row in missing.iterrows()
                                    ]
                                )
                            )

                    print(
                        f"\nChunk {chunk_id}/{n_chunks}: records {chunk_start + 1}-{chunk_end} "
                        f"-> reusing existing output ({existing_rows} ecgs)"
                    )
                    chunk_master_paths.append(existing_master)
                    chunk_key_paths.append(existing_key)
                    processed_before_chunk += len(chunk_records)
                    continue
                    if existing_rows > len(chunk_records):
                        raise ValueError(
                            f"Resume mismatch: {existing_key.name} has {existing_rows} rows "
                            f"but chunk {chunk_id} spans only {len(chunk_records)} records. "
                            f"This means --records-per-chunk ({args.records_per_chunk}) or "
                            f"record_list.csv differs from the interrupted run. Re-run with "
                            f"the original --records-per-chunk, or drop --resume to start over."
                        )
                    print(
                        f"\nChunk {chunk_id}/{n_chunks}: records {chunk_start + 1}-{chunk_end} "
                        f"-> reusing existing output ({existing_rows} ecgs)"
                    )
                    chunk_master_paths.append(existing_master)
                    chunk_key_paths.append(existing_key)
                    processed_before_chunk += len(chunk_records)
                    continue

            print(f"\nChunk {chunk_id}/{n_chunks}: records {chunk_start + 1}-{chunk_end}")
            chunk_df, failed_df = process_record_chunk(
                model=model,
                chunk_records=chunk_records,
                ecg_root=args.ecg_root,
                target_sample_rate=ECGFM_SAMPLE_RATE_HZ,
                segment_seconds=ECGFM_SEGMENT_SECONDS,
                batch_size=args.batch_size,
                device=device,
                chunk_id=chunk_id,
                total_records=total_records,
                processed_before_chunk=processed_before_chunk,
            )
            sys.stdout.write("\n")
            sys.stdout.flush()

            chunk_master_path, chunk_key_path = write_chunk_outputs(
                chunk_df=chunk_df,
                chunk_dir=chunk_dir,
                chunk_id=chunk_id,
            )
            if chunk_master_path is not None and chunk_key_path is not None:
                chunk_master_paths.append(chunk_master_path)
                chunk_key_paths.append(chunk_key_path)

            if not failed_df.empty:
                failed_chunks.append(failed_df)

            processed_before_chunk += len(chunk_records)
            del chunk_df, failed_df
            if device == "cuda":
                torch.cuda.empty_cache()

        embedding_table, key_table, key_columns = concatenate_chunk_outputs(
            chunk_master_paths=chunk_master_paths,
            chunk_key_paths=chunk_key_paths,
            embedding_table_path=embedding_table_path,
            key_table_csv_path=key_table_csv_path,
        )

        if failed_chunks:
            failed_records = pd.concat(failed_chunks, ignore_index=True)
            failed_records.to_csv(failed_records_path, index=False)
        else:
            failed_records = pd.DataFrame()
            if failed_records_path.exists():
                failed_records_path.unlink()

        guard_counts: dict[str, int] = {}
        if "guard_reason" in failed_records.columns:
            guard_counts = {
                str(reason): int(count)
                for reason, count in failed_records["guard_reason"]
                .dropna()
                .value_counts()
                .items()
            }
        if guard_counts:
            summary_str = ", ".join(f"{k}={v}" for k, v in sorted(guard_counts.items()))
            print(f"Guarded off {sum(guard_counts.values())} recordings ({summary_str}).")

        segment_counts = (
            key_table["n_segments"].astype(int).to_numpy()
            if "n_segments" in key_table.columns and not key_table.empty
            else np.asarray([], dtype=np.int64)
        )
        embedding_dim = 0
        if not embedding_table.empty:
            embedding_dim = sum(
                column.startswith("ecg_ft") and column[6:].isdigit()
                for column in embedding_table.columns
            )

        summary = {
            "ecgfm_model_kind": "pretrained",
            "checkpoint_path": str(checkpoint_path),
            "device": device,
            "test_mode": bool(args.test_mode),
            "test_record_limit": TEST_RECORD_LIMIT if args.test_mode else None,
            "n_ecgs": int(len(embedding_table)),
            "n_subjects": (
                int(embedding_table["subject_id"].nunique())
                if "subject_id" in embedding_table.columns and not embedding_table.empty
                else None
            ),
            "embedding_dim": embedding_dim,
            "n_failed_records": int(len(failed_records)),
        }
        metadata = {
            "ecg_root": str(args.ecg_root),
            "ecgfm_model_kind": "pretrained",
            "checkpoint_path": str(checkpoint_path),
            "target_sample_rate": ECGFM_SAMPLE_RATE_HZ,
            "input_mode": "ecgfm_paper_5s_segment_mean_pool",
            "ecgfm_forward_mode": "mask=False, features_only=True, encoder_out=x",
            "resample_method": "linear_interp1d",
            "standardization": "per_lead_z_score",
            "segment_seconds": ECGFM_SEGMENT_SECONDS,
            "segment_aggregation": "mean_by_record_idx",
            "batch_size": args.batch_size,
            "records_per_chunk": args.records_per_chunk,
            "n_chunks": n_chunks,
            "device": device,
            "test_mode": args.test_mode,
            "test_record_limit": TEST_RECORD_LIMIT if args.test_mode else None,
            "embedding_dim": embedding_dim,
            "embedding_column_prefix": "ecg_ft",
            "embedding_table_path": str(embedding_table_path),
            "key_table_csv_path": str(key_table_csv_path),
            "failed_records_path": str(failed_records_path) if not failed_records.empty else None,
            "key_columns": key_columns,
            "n_segments_total": int(segment_counts.sum()) if len(segment_counts) else 0,
            "segments_per_ecg_min": int(segment_counts.min()) if len(segment_counts) else None,
            "segments_per_ecg_median": (
                float(np.median(segment_counts)) if len(segment_counts) else None
            ),
            "segments_per_ecg_max": int(segment_counts.max()) if len(segment_counts) else None,
            "n_failed_records": int(len(failed_records)),
            "n_guarded_records": sum(guard_counts.values()),
            "guarded_records_by_reason": guard_counts,
            "summary": summary,
        }
        if args.keep_chunks:
            metadata["chunk_dir"] = str(chunk_dir)

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except Exception:
        cleanup_chunks_after_error()
        raise

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir)

    print(f"Saved master embedding table to: {embedding_table_path}")
    print(f"Saved embedding key CSV to: {key_table_csv_path}")
    if not failed_records.empty:
        print(f"Saved failed records CSV to: {failed_records_path}")
    print(f"Saved run metadata to: {metadata_path}")
    return metadata


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.records_per_chunk <= 0:
        raise ValueError("--records-per-chunk must be a positive integer.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Downloading or loading the pinned pretrained ECG-FM checkpoint...")
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=ECGFM_REPO_ID,
            filename=ECGFM_FILENAME,
            revision=ECGFM_REVISION,
        )
    )

    print("Reading ECG record metadata...")

    record_list_path = args.ecg_root / "record_list.csv"
    records = pd.read_csv(record_list_path, low_memory=False)
    records = prepare_record_keys(records)
    records_for_run = (
        records.head(TEST_RECORD_LIMIT).copy() if args.test_mode else records.copy()
    )
    print(
        f"Record list rows to process: {len(records_for_run)} "
        f"of {len(records)} available rows."
    )
    if "subject_id" in records_for_run.columns:
        print(f"Unique patients: {records_for_run['subject_id'].nunique()}")

    model = load_ecgfm_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    extract_embeddings_chunked(
        args=args,
        model=model,
        records=records_for_run,
        checkpoint_path=checkpoint_path,
        device=device,
    )


if __name__ == "__main__":
    main()
