"""Extract and encode Brief Hospital Course (BHC) notes.

The script reads BHC text from ``mimic-iv-bhc.csv`` and identifiers and timestamps from
``discharge.csv``. It writes one embedding per note, a matching key table and run
metadata. Notes are encoded without cohort or time filters so they can be selected later for
different cohorts and time windows.

Usage::

    # encode one embedding per note and admission
    python -m src.preprocessing.extract_notes

    # process a small test run
    python -m src.preprocessing.extract_notes --test-mode
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Let this file run directly or as a module by making the repository root available
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "notes_bhc"

# Default models for turning note text into vectors and sentences
DEFAULT_TRANSFORMER_MODEL = "emilyalsentzer/Bio_Discharge_Summary_BERT"
DEFAULT_SPACY_MODEL = "en_core_sci_md"
# Maximum number of tokens per sentence
SENTENCE_MAX_LENGTH = 128
# Sentences per model call. Larger groups can be slower on the available hardware.
DEFAULT_BATCH_SIZE = 32
# Notes handled together when sentences are split
SPACY_PIPE_BATCH_SIZE = 256
EMBEDDING_DIM = 768
EMBEDDING_COLUMN_PREFIX = "notes_ft"
SENTENCE_POOLING = (
    "attention-mask-weighted mean over last_hidden_state dim=1 -> (n_sentences, 768)"
)
# Stored in the run record to describe how sentences were grouped
SENTENCE_BATCHING = "cross-note, length-bucketed"
NOTE_POOLING = (
    "mean over sentence embeddings -> one (768,) vector per note_id "
    "(= one per subject_id x hadm_id: exactly one discharge note per admission)"
)

# Note data sources. The BHC table supplies the text. ``discharge.csv`` supplies only
# identifiers and times, not the full discharge summary. The misspelt directory name is kept
# because it is part of the source path.
MIMIC_IV_DIR = WORKSPACE_ROOT / "data" / "MIMIC-IV"
DEFAULT_DISCHARGE_PATH = MIMIC_IV_DIR / "mimic-iv-note-2.2" / "note" / "discharge.csv"
DEFAULT_BHC_PATH = (
    MIMIC_IV_DIR
    / "mimic-iv-ext-bhc-labeled-clinical-notes-dataset-for-hospital-course-summarization-1.2.0"
    / "mimic-iv-bhc.csv"
)


# ``target`` is the extracted Brief Hospital Course section
NOTES_SUBJECT_COLUMN = "subject_id"
NOTES_TEXT_COLUMN = "target"
# Columns used to identify each note. ``note_id`` identifies the note, ``hadm_id`` identifies the
# admission and ``note_time`` is used for time windows
NOTE_ID_COLUMN = "note_id"
NOTES_HADM_COLUMN = "hadm_id"
NOTE_TIME_COLUMN = "note_time"
NOTE_STORETIME_COLUMN = "note_storetime"
BHC_SOURCE_COLUMNS = [NOTE_ID_COLUMN, NOTES_TEXT_COLUMN, "input_tokens", "target_tokens"]
DISCHARGE_ID_COLUMNS = [
    NOTE_ID_COLUMN,
    NOTES_SUBJECT_COLUMN,
    NOTES_HADM_COLUMN,
    "charttime",
    "storetime",
]
# ``is_empty_note`` is omitted because empty notes are written to a separate file rather than
# stored as zero vectors. ``n_input_tokens`` counts tokens sent to the model after shortening and
# excluding padding.
NOTE_TABLES_KEY_COLUMNS = [
    NOTE_ID_COLUMN,
    NOTES_SUBJECT_COLUMN,
    NOTES_HADM_COLUMN,
    NOTE_TIME_COLUMN,
    NOTE_STORETIME_COLUMN,
    "n_sentences",
    "n_input_tokens",
    "target_tokens",
    "input_tokens",
]
NOTES_KEY_COLUMNS = [
    NOTES_SUBJECT_COLUMN,
    "n_sentences",
    "n_input_tokens",
    "is_empty_note",
    "num_summaries",
    "num_input_tokens",
    "num_target_tokens",
]
# Keep row order fixed because ``--resume`` uses row positions to find each group
# ``note_seq`` cannot break ties because there is one discharge note per ``hadm_id``
NOTE_GRAIN_SORT_COLUMNS = [
    NOTES_SUBJECT_COLUMN,
    NOTE_TIME_COLUMN,
    NOTE_STORETIME_COLUMN,
    NOTES_HADM_COLUMN,
]

# ``___`` marks hidden parts of MIMIC notes. The cleaner replaces it with spaces and then joins
# repeated whitespace
DEIDENTIFICATION_PATTERN = re.compile(r"___")
WHITESPACE_PATTERN = re.compile(r"\s+")
TEST_RECORD_LIMIT = 100
PROGRESS_EVERY_SUBJECTS = 500


class SentenceEncoder(Protocol):
    """Convert a list of sentences into one vector per sentence.

    An encoder can also report the number of non-padding tokens used for each sentence. Encoders
    that do not report this information are supported, but their token counts are left blank.
    """

    def __call__(self, sentences: list[str]) -> np.ndarray: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode notes into one 768-dimensional Bio+Discharge Summary BERT vector per note/admission."
        )
    )
    parser.add_argument(
        "--bhc-path",
        type=Path,
        default=DEFAULT_BHC_PATH,
        help="Path to mimic-iv-bhc.csv, which supplies the BHC text.",
    )
    parser.add_argument(
        "--discharge-path",
        type=Path,
        default=DEFAULT_DISCHARGE_PATH,
        help=(
            "Path to discharge.csv. Read only for note_id, subject_id, hadm_id, charttime and "
            "storetime, never for its text column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the embedding, key and run metadata files.",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        action="store_true",
        default=False,
        help=f"Process only the first {TEST_RECORD_LIMIT} notes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of sentences per transformer call. Sentences from a chunk are grouped and "
            "sorted by length before processing."
        ),
    )
    parser.add_argument(
        "--split-processes",
        type=int,
        default=1,
        help=(
            "Worker processes for spaCy sentence splitting. Use fewer workers than available "
            "CPU cores. Sentence boundaries are unchanged."
        ),
    )
    parser.add_argument(
        "--notes-per-chunk",
        "-r",
        dest="notes_per_chunk",
        type=int,
        default=5000,
        help="Number of notes to encode before saving chunk outputs.",
    )
    parser.add_argument(
        "--keep-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep chunk files after the final outputs are built so an interrupted run can be "
            "resumed. Pass --no-keep-chunks to delete them."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume a partial run by reusing complete chunks and processing only missing chunks. "
            "Use the same --notes-per-chunk value and source tables as the interrupted run. "
            "Reused chunks are kept even with --no-keep-chunks."
        ),
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Check that the command-line values are valid."""
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}.")
    if args.notes_per_chunk < 1:
        raise ValueError(f"--notes-per-chunk must be >= 1, got {args.notes_per_chunk}.")
    if args.split_processes < 1:
        raise ValueError(f"--split-processes must be >= 1, got {args.split_processes}.")


def resolve_device() -> str:
    """Use CUDA when available, otherwise use the CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def clean_note_text(text: str | float | None) -> str:
    """Replace ``___`` placeholders with spaces and collapse repeated whitespace.

    Missing text becomes an empty string. Leading whitespace is retained to match the existing
    cleaner.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    cleaned = DEIDENTIFICATION_PATTERN.sub(" ", str(text))
    return WHITESPACE_PATTERN.sub(" ", cleaned)


def require_columns(path: Path, required: list[str]) -> None:
    """Check that a CSV has the columns needed by the script."""
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s) {missing}.")


def read_notes(
    notes_path: Path,
    test_mode: bool = False,
    test_record_limit: int = TEST_RECORD_LIMIT,
) -> pd.DataFrame:
    """Read one pooled note row per subject from a CSV."""
    required = [NOTES_SUBJECT_COLUMN, NOTES_TEXT_COLUMN]
    require_columns(notes_path, required)
    notes = pd.read_csv(notes_path)
    duplicated = int(notes[NOTES_SUBJECT_COLUMN].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{notes_path} has {duplicated} duplicated {NOTES_SUBJECT_COLUMN} value(s)."
        )
    if test_mode:
        notes = notes.head(test_record_limit).reset_index(drop=True)
    notes[NOTES_TEXT_COLUMN] = notes[NOTES_TEXT_COLUMN].map(clean_note_text)
    return notes


def read_notes_from_note_tables(
    bhc_path: Path,
    discharge_path: Path,
    test_mode: bool = False,
    test_record_limit: int = TEST_RECORD_LIMIT,
) -> pd.DataFrame:
    """Read BHC notes with their subject, admission and time information.

    The BHC file supplies the text. The discharge file supplies the identifiers and timestamps.

    All source rows are read without cohort or time filters.
    """
    require_columns(bhc_path, BHC_SOURCE_COLUMNS)
    require_columns(discharge_path, DISCHARGE_ID_COLUMNS)

    bhc = pd.read_csv(bhc_path, usecols=BHC_SOURCE_COLUMNS)
    duplicated = int(bhc[NOTE_ID_COLUMN].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"{bhc_path} has {duplicated} duplicated {NOTE_ID_COLUMN} value(s). The BHC source "
            "is expected to carry one row per note."
        )
    identifiers = pd.read_csv(
        discharge_path,
        usecols=DISCHARGE_ID_COLUMNS,
        parse_dates=["charttime", "storetime"],
    )
    notes = bhc.merge(identifiers, on=NOTE_ID_COLUMN, how="left", validate="one_to_one")

    # Make sure every BHC note has matching subject and admission information. A note without
    # this information cannot be used in later time windows.
    unmatched = int(notes[NOTES_SUBJECT_COLUMN].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} note_id(s) in {bhc_path} have no row in {discharge_path}. Every BHC "
            "note must resolve to a subject_id and hadm_id."
        )
    missing_hadm = int(notes[NOTES_HADM_COLUMN].isna().sum())
    if missing_hadm:
        raise ValueError(
            f"{missing_hadm} note(s) resolved to a null {NOTES_HADM_COLUMN}. The encounter id is "
            "required for downstream windowing."
        )

    notes = notes.rename(
        columns={"charttime": NOTE_TIME_COLUMN, "storetime": NOTE_STORETIME_COLUMN}
    )
    notes[NOTES_SUBJECT_COLUMN] = notes[NOTES_SUBJECT_COLUMN].astype("int64")
    notes[NOTES_HADM_COLUMN] = notes[NOTES_HADM_COLUMN].astype("int64")
    notes = notes.sort_values(NOTE_GRAIN_SORT_COLUMNS, kind="mergesort").reset_index(drop=True)
    if test_mode:
        notes = notes.head(test_record_limit).reset_index(drop=True)
    notes[NOTES_TEXT_COLUMN] = notes[NOTES_TEXT_COLUMN].map(clean_note_text)
    return notes


def load_sentence_splitter(
    spacy_model: str = DEFAULT_SPACY_MODEL,
    n_process: int = 1,
) -> Callable[[str], list[str]]:
    """Load spaCy and return a function that splits text into sentences.

    spaCy is loaded only when needed. The returned function can also split several notes at once.
    """
    import spacy

    nlp = spacy.load(
        spacy_model,
        exclude=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "parser", "ner"],
    )
    nlp.add_pipe("sentencizer")

    def split(text: str) -> list[str]:
        return split_sentences(text, nlp)

    def pipe(texts: list[str]) -> list[list[str]]:
        return split_sentences_batched(texts, nlp, n_process=n_process)

    split.pipe = pipe  # type: ignore[attr-defined]
    return split


def split_sentences(text: str, nlp: Any) -> list[str]:
    """Split cleaned note text into non-empty sentences."""
    if not text or not text.strip():
        return []
    return [sentence.text for sentence in nlp(text).sents if sentence.text.strip()]


def split_sentences_batched(texts: list[str], nlp: Any, n_process: int = 1) -> list[list[str]]:
    """Split several notes at once.

    Blank notes return empty lists in their original positions.
    """
    results: list[list[str]] = [[] for _ in texts]
    positions = [index for index, text in enumerate(texts) if text and text.strip()]
    if not positions:
        return results
    docs = nlp.pipe(
        [texts[index] for index in positions],
        batch_size=SPACY_PIPE_BATCH_SIZE,
        n_process=n_process,
    )
    for index, doc in zip(positions, docs, strict=True):
        results[index] = [sentence.text for sentence in doc.sents if sentence.text.strip()]
    return results


def split_notes(texts: list[str], splitter: Callable[[str], list[str]]) -> list[list[str]]:
    """Split every note, using the faster grouped method when available."""
    pipe = getattr(splitter, "pipe", None)
    if pipe is not None:
        sentences_per_note = pipe(texts)
        if len(sentences_per_note) != len(texts):
            raise ValueError(
                f"Splitter.pipe returned {len(sentences_per_note)} result(s) for {len(texts)} "
                "note(s). The two must be positionally aligned."
            )
        return sentences_per_note
    return [splitter(text) for text in texts]


class ClinicalBertSentenceEncoder:
    """Turn sentences into vectors with a Hugging Face model.

    Each call shortens long sentences when needed and ignores added padding when averaging the
    token information. It returns one 768-value vector for each sentence.

    The latest token count is stored in ``last_n_input_tokens``. Counts for individual sentences
    are stored in ``last_input_token_counts``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_TRANSFORMER_MODEL,
        device: str = "cpu",
        max_length: int = SENTENCE_MAX_LENGTH,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use the safer model file format because the cached file cannot be read here
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(device)
        self.model.eval()
        self.last_n_input_tokens = 0
        self.last_input_token_counts: list[int] = []

    def __call__(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            self.last_n_input_tokens = 0
            self.last_input_token_counts = []
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_attention_mask=True,
        ).to(self.device)
        mask = inputs["attention_mask"]
        per_sentence_tokens = mask.sum(dim=1)
        self.last_input_token_counts = [int(count) for count in per_sentence_tokens.tolist()]
        self.last_n_input_tokens = int(sum(self.last_input_token_counts))
        with self._torch.no_grad():
            outputs = self.model(**inputs)
            # Avoid division by zero if an encoder returns an empty mask
            weights = mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            summed = (outputs.last_hidden_state * weights).sum(dim=1)
            sentence_embeddings = (
                (summed / per_sentence_tokens.unsqueeze(-1).clamp(min=1)).cpu().numpy()
            )
        return sentence_embeddings.astype(np.float32)


def encode_sentences_batched(
    sentences: list[str],
    encoder: SentenceEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, int]:
    """Encode sentences in groups and return their vectors and token count."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if not sentences:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), 0

    batches: list[np.ndarray] = []
    n_input_tokens = 0
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        batch_embeddings = np.asarray(encoder(batch), dtype=np.float32)
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(batch):
            raise ValueError(
                f"Encoder returned shape {batch_embeddings.shape} for {len(batch)} sentence(s). "
                f"expected ({len(batch)}, {EMBEDDING_DIM})."
            )
        batches.append(batch_embeddings)
        token_counts = getattr(encoder, "last_input_token_counts", None)
        if token_counts is not None and len(token_counts) == len(batch):
            n_input_tokens += int(np.asarray(token_counts, dtype=np.int64).sum())
        else:
            n_input_tokens += int(getattr(encoder, "last_n_input_tokens", 0))
    return np.vstack(batches).astype(np.float32), n_input_tokens


def encode_note(
    text: str,
    splitter: Callable[[str], list[str]],
    encoder: SentenceEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, int, int]:
    """Encode one note and combine its sentence vectors."""
    sentences = splitter(clean_note_text(text))
    n_sentences = len(sentences)
    if n_sentences == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32), 0, 0
    sentence_embeddings, n_input_tokens = encode_sentences_batched(
        sentences, encoder, batch_size=batch_size
    )
    return pool_sentence_embeddings(sentence_embeddings), n_sentences, n_input_tokens


def encode_sentences_bucketed(
    sentences_per_note: list[list[str]],
    encoder: SentenceEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[np.ndarray], list[int | None]]:
    """Encode all notes in length order and restore their original order.

    Returns the sentence vectors for each note and its input token count. The count is ``None``
    when the encoder does not provide it.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    owners = [
        (note_index, sentence)
        for note_index, sentences in enumerate(sentences_per_note)
        for sentence in sentences
    ]
    n_sentences = len(owners)
    embeddings = np.zeros((n_sentences, EMBEDDING_DIM), dtype=np.float32)
    token_counts = np.zeros(n_sentences, dtype=np.int64)
    tokens_reported = True

    # Process longer sentences first and keep the original order when lengths match
    order = sorted(range(n_sentences), key=lambda index: (-len(owners[index][1]), index))
    for start in range(0, n_sentences, batch_size):
        positions = order[start : start + batch_size]
        batch = [owners[position][1] for position in positions]
        batch_embeddings = np.asarray(encoder(batch), dtype=np.float32)
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(batch):
            raise ValueError(
                f"Encoder returned shape {batch_embeddings.shape} for {len(batch)} sentence(s). "
                f"expected ({len(batch)}, {EMBEDDING_DIM})."
            )
        embeddings[positions] = batch_embeddings
        batch_token_counts = getattr(encoder, "last_input_token_counts", None)
        if batch_token_counts is None or len(batch_token_counts) != len(batch):
            tokens_reported = False
        else:
            token_counts[positions] = np.asarray(batch_token_counts, dtype=np.int64)

    per_note_embeddings: list[np.ndarray] = []
    per_note_tokens: list[int | None] = []
    cursor = 0
    for sentences in sentences_per_note:
        stop = cursor + len(sentences)
        per_note_embeddings.append(embeddings[cursor:stop])
        per_note_tokens.append(int(token_counts[cursor:stop].sum()) if tokens_reported else None)
        cursor = stop
    return per_note_embeddings, per_note_tokens


def pool_sentence_embeddings(
    sentence_embeddings: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    """Average a note's sentence vectors into one vector.

    Empty input returns a zero vector. The expected vector width is checked first.
    """
    embeddings = np.asarray(sentence_embeddings, dtype=np.float32)
    if embeddings.size == 0:
        return np.zeros(embedding_dim, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected sentence embeddings of shape (n_sentences, {embedding_dim}), got "
            f"{embeddings.shape}."
        )
    return embeddings.mean(axis=0).astype(np.float32)


def embedding_column_names(embedding_dim: int = EMBEDDING_DIM) -> list[str]:
    """Return the feature column names in order."""
    return [f"{EMBEDDING_COLUMN_PREFIX}{index}" for index in range(embedding_dim)]


def build_chunk_frame(
    rows: list[dict[str, Any]],
    embeddings: list[np.ndarray],
    key_columns: list[str],
) -> pd.DataFrame:
    """Combine one chunk's identifying columns and vectors into a table."""
    expected = list(key_columns)
    keys = pd.DataFrame(rows)
    if keys.empty:
        return pd.DataFrame(columns=[*expected, *embedding_column_names()])
    features = pd.DataFrame(
        np.vstack(embeddings).astype(np.float32),
        columns=embedding_column_names(),
    )
    present_key_columns = [column for column in expected if column in keys.columns]
    return pd.concat(
        [keys.loc[:, present_key_columns].reset_index(drop=True), features], axis=1
    )


def encode_note_chunk(
    chunk_notes: pd.DataFrame,
    splitter: Callable[[str], list[str]],
    encoder: SentenceEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_every_subjects: int = PROGRESS_EVERY_SUBJECTS,
    processed_before_chunk: int = 0,
    total_subjects: int | None = None,
    note_grain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode one group of notes and return its tables.

    Notes are split before they are encoded. A note with no sentences is returned separately.
    """
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    empty_rows: list[dict[str, Any]] = []

    sentences_per_note = split_notes(chunk_notes[NOTES_TEXT_COLUMN].tolist(), splitter)
    sentence_embeddings, token_counts = encode_sentences_bucketed(
        sentences_per_note, encoder, batch_size=batch_size
    )

    for local_pos, (_, note_row) in enumerate(chunk_notes.iterrows(), start=1):
        n_sentences = len(sentences_per_note[local_pos - 1])
        global_pos = processed_before_chunk + local_pos
        if progress_every_subjects and global_pos % progress_every_subjects == 0:
            total = total_subjects if total_subjects is not None else "?"
            print(f"  encoded {global_pos}/{total} notes", flush=True)
        if note_grain and n_sentences == 0:
            empty_rows.append(
                {
                    NOTE_ID_COLUMN: note_row[NOTE_ID_COLUMN],
                    NOTES_SUBJECT_COLUMN: note_row[NOTES_SUBJECT_COLUMN],
                    NOTES_HADM_COLUMN: note_row[NOTES_HADM_COLUMN],
                    NOTE_TIME_COLUMN: note_row[NOTE_TIME_COLUMN],
                    "reason": "no_sentences_after_cleaning",
                }
            )
            continue
        if note_grain:
            rows.append(
                {
                    NOTE_ID_COLUMN: note_row[NOTE_ID_COLUMN],
                    NOTES_SUBJECT_COLUMN: note_row[NOTES_SUBJECT_COLUMN],
                    NOTES_HADM_COLUMN: note_row[NOTES_HADM_COLUMN],
                    NOTE_TIME_COLUMN: note_row[NOTE_TIME_COLUMN],
                    NOTE_STORETIME_COLUMN: note_row[NOTE_STORETIME_COLUMN],
                    "n_sentences": int(n_sentences),
                    "n_input_tokens": token_counts[local_pos - 1],
                    "target_tokens": note_row.get("target_tokens"),
                    "input_tokens": note_row.get("input_tokens"),
                }
            )
        else:
            rows.append(
                {
                    NOTES_SUBJECT_COLUMN: note_row[NOTES_SUBJECT_COLUMN],
                    "n_sentences": int(n_sentences),
                    "n_input_tokens": token_counts[local_pos - 1],
                    "is_empty_note": bool(n_sentences == 0),
                    "num_summaries": note_row.get("num_summaries"),
                    "num_input_tokens": note_row.get("num_input_tokens"),
                    "num_target_tokens": note_row.get("num_target_tokens"),
                }
            )
        embeddings.append(
            pool_sentence_embeddings(sentence_embeddings[local_pos - 1])
            if n_sentences > 0
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )

    key_columns = NOTE_TABLES_KEY_COLUMNS if note_grain else NOTES_KEY_COLUMNS
    chunk_df = build_chunk_frame(rows, embeddings, key_columns=key_columns)
    return chunk_df, pd.DataFrame(empty_rows)


def write_chunk_outputs(
    chunk_df: pd.DataFrame,
    chunk_dir: Path,
    chunk_id: int,
    key_columns: list[str],
    empty_df: pd.DataFrame | None = None,
) -> tuple[Path, Path] | tuple[None, None]:
    """Write one group's temporary vector, key and empty-note files."""
    if empty_df is not None and not empty_df.empty:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(chunk_dir / f"chunk_{chunk_id:05d}_empty.csv", index=False)
    if chunk_df.empty:
        return None, None

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_embeddings_path = chunk_dir / f"chunk_{chunk_id:05d}_embeddings.pkl"
    chunk_key_path = chunk_dir / f"chunk_{chunk_id:05d}_keys.csv"
    present_key_columns = [column for column in key_columns if column in chunk_df.columns]
    chunk_df.to_pickle(chunk_embeddings_path)
    chunk_df.loc[:, present_key_columns].to_csv(chunk_key_path, index=False)
    return chunk_embeddings_path, chunk_key_path


def concatenate_chunk_outputs(
    chunk_embeddings_paths: list[Path],
    chunk_key_paths: list[Path],
    embedding_table_path: Path,
    key_table_csv_path: Path,
    key_columns: list[str],
    note_grain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the final vector and key tables from the temporary files."""
    expected = list(key_columns)
    if chunk_embeddings_paths:
        embedding_table = pd.concat(
            [pd.read_pickle(path) for path in chunk_embeddings_paths],
            ignore_index=True,
        )
    else:
        embedding_table = pd.DataFrame(columns=[*expected, *embedding_column_names()])

    if chunk_key_paths:
        key_table = pd.concat(
            [pd.read_csv(path) for path in chunk_key_paths],
            ignore_index=True,
        )
        key_columns = key_table.columns.tolist()
    else:
        key_columns = list(expected)
        key_table = pd.DataFrame(columns=key_columns)

    validate_row_alignment(embedding_table, key_table, note_grain=note_grain)
    embedding_table.to_pickle(embedding_table_path)
    key_table.to_csv(key_table_csv_path, index=False)
    return embedding_table, key_table, key_columns


def validate_row_alignment(
    embedding_table: pd.DataFrame, key_table: pd.DataFrame, note_grain: bool = False
) -> None:
    """Check that the key table and vector table have the same rows in the same order."""
    if len(embedding_table) != len(key_table):
        raise ValueError(
            f"Embedding table has {len(embedding_table)} row(s) but the key table has "
            f"{len(key_table)}. The two must be row-aligned."
        )
    if embedding_table.empty:
        return
    identity = NOTE_ID_COLUMN if note_grain else NOTES_SUBJECT_COLUMN
    if not np.array_equal(
        embedding_table[identity].to_numpy(), key_table[identity].to_numpy()
    ):
        raise ValueError(
            f"Embedding and key tables disagree on {identity} order. The two must be "
            "row-aligned."
        )
    duplicated = int(key_table[identity].duplicated().sum())
    if duplicated:
        raise ValueError(
            f"Key table has {duplicated} duplicated {identity} value(s). Each note must "
            "appear exactly once."
        )
    if note_grain and key_table[NOTES_HADM_COLUMN].isna().any():
        raise ValueError(
            "Key table has null hadm_id value(s). Every note must carry its encounter id."
        )


def chunk_bounds(total_rows: int, rows_per_chunk: int) -> list[tuple[int, int, int]]:
    """Return the start and end positions for each group of rows."""
    if total_rows <= 0:
        return []
    n_chunks = int(np.ceil(total_rows / rows_per_chunk))
    return [
        (chunk_id, chunk_id * rows_per_chunk,
         min((chunk_id + 1) * rows_per_chunk, total_rows))
        for chunk_id in range(n_chunks)
    ]


def encode_notes_chunked(
    notes: pd.DataFrame,
    splitter: Callable[[str], list[str]],
    encoder: SentenceEncoder,
    output_paths: dict[str, Path],
    chunk_dir: Path,
    subjects_per_chunk: int = 5000,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
    keep_chunks: bool = True,
    progress_every_subjects: int = PROGRESS_EVERY_SUBJECTS,
    note_grain: bool = False,
    empty_records_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, pd.DataFrame]:
    """Encode all notes in groups and combine the results.

    The same group size and source rows are needed to resume an interrupted run.

    Groups are kept by default so completed work does not need to be repeated.
    """
    if chunk_dir.exists() and not resume:
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total_subjects = len(notes)
    bounds = chunk_bounds(total_subjects, subjects_per_chunk)
    print(
        f"Encoding {total_subjects} notes across {len(bounds)} chunk(s) of up to "
        f"{subjects_per_chunk} notes.",
        flush=True,
    )

    chunk_embeddings_paths: list[Path] = []
    chunk_key_paths: list[Path] = []
    n_reused_chunks = 0
    key_columns = NOTE_TABLES_KEY_COLUMNS if note_grain else NOTES_KEY_COLUMNS
    try:
        for chunk_id, start, stop in bounds:
            chunk_embeddings_path = chunk_dir / f"chunk_{chunk_id:05d}_embeddings.pkl"
            chunk_key_path = chunk_dir / f"chunk_{chunk_id:05d}_keys.csv"
            if resume and chunk_embeddings_path.exists() and chunk_key_path.exists():
                if note_grain and NOTE_ID_COLUMN not in pd.read_csv(chunk_key_path, nrows=0).columns:
                    raise ValueError(
                        f"Cannot resume from {chunk_key_path}: it is not a note-level chunk. "
                        "Use a different output directory or run without --resume."
                    )
                print(
                    f"Chunk {chunk_id} (rows {start}:{stop}) already complete, reusing.",
                    flush=True,
                )
                chunk_embeddings_paths.append(chunk_embeddings_path)
                chunk_key_paths.append(chunk_key_path)
                n_reused_chunks += 1
                continue

            chunk_df, chunk_empty = encode_note_chunk(
                notes.iloc[start:stop],
                splitter,
                encoder,
                batch_size=batch_size,
                progress_every_subjects=progress_every_subjects,
                processed_before_chunk=start,
                total_subjects=total_subjects,
                note_grain=note_grain,
            )
            written_embeddings, written_keys = write_chunk_outputs(
                chunk_df, chunk_dir, chunk_id, key_columns=key_columns, empty_df=chunk_empty
            )
            if written_embeddings is not None and written_keys is not None:
                chunk_embeddings_paths.append(written_embeddings)
                chunk_key_paths.append(written_keys)
            del chunk_df

        embedding_table, key_table, key_columns = concatenate_chunk_outputs(
            chunk_embeddings_paths,
            chunk_key_paths,
            output_paths["embeddings"],
            output_paths["keys"],
            key_columns=key_columns,
            note_grain=note_grain,
        )
        # Collect empty-note files before removing the chunk directory
        empty_shards = sorted(chunk_dir.glob("chunk_*_empty.csv"))
        empty_records = (
            pd.concat([pd.read_csv(path) for path in empty_shards], ignore_index=True)
            if empty_shards
            else pd.DataFrame()
        )
        if empty_records_path is not None:
            if empty_records.empty:
                if empty_records_path.exists():
                    empty_records_path.unlink()
            else:
                empty_records.to_csv(empty_records_path, index=False)
    except BaseException:
        # Preserve completed chunks when a resumable run fails
        if not keep_chunks and not resume and chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        raise

    if not keep_chunks and chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    return embedding_table, key_table, key_columns, n_reused_chunks, empty_records


def output_paths_for(output_dir: Path, test_mode: bool, test_record_limit: int) -> dict[str, Path]:
    """Return the output file paths, including the test limit when needed."""
    suffix = f"_test{test_record_limit}" if test_mode else ""
    return {
        "embeddings": output_dir / f"notes_embeddings{suffix}.pkl",
        "keys": output_dir / f"notes_embedding_keys{suffix}.csv",
        "metadata": output_dir / f"notes_embedding_metadata{suffix}.json",
        "chunks": output_dir / f"notes_chunks{suffix}",
        "empty_records": output_dir / f"notes_empty_records{suffix}.csv",
    }


def build_metadata(
    args: argparse.Namespace,
    key_table: pd.DataFrame,
    key_columns: list[str],
    output_paths: dict[str, Path],
    device: str,
    n_chunks: int,
    n_reused_chunks: int,
    note_grain: bool = False,
    n_empty_notes: int | None = None,
) -> dict[str, Any]:
    """Create the information recorded about the run."""
    n_rows = int(len(key_table))
    n_subjects = (
        int(key_table[NOTES_SUBJECT_COLUMN].nunique())
        if n_rows and NOTES_SUBJECT_COLUMN in key_table.columns
        else n_rows
    )
    sentence_counts = (
        key_table["n_sentences"] if "n_sentences" in key_table.columns else pd.Series(dtype=float)
    )
    if n_empty_notes is None:
        n_empty_notes = int(key_table["is_empty_note"].sum()) if "is_empty_note" in key_table else 0
    summary = {
        "transformer_model": DEFAULT_TRANSFORMER_MODEL,
        "device": device,
        "test_mode": bool(args.test_mode),
        "test_record_limit": TEST_RECORD_LIMIT if args.test_mode else None,
        "n_subjects": n_subjects,
        "n_notes": n_rows,
        "embedding_dim": EMBEDDING_DIM,
        "n_empty_notes": int(n_empty_notes),
    }
    metadata = {
        "grain": "note" if note_grain else "subject",
        "notes_source": "note-tables" if note_grain else "pooled",
        # The script reads only the MIMIC source tables, so cohort changes do not affect the
        # saved vectors
        "cohort_filter_applied": False,
        "transformer_model": DEFAULT_TRANSFORMER_MODEL,
        "spacy_model": DEFAULT_SPACY_MODEL,
        "sentence_max_length": int(SENTENCE_MAX_LENGTH),
        "sentence_pooling": SENTENCE_POOLING,
        "sentence_batching": SENTENCE_BATCHING,
        "pooling_description": (
            "mean over sentence embeddings -> one (768,) vector per note_id"
            if note_grain
            else "mean over sentence embeddings"
        ),
        "subject_pooling": "mean over sentence embeddings -> one (768,) vector per subject_id",
        "batch_size": int(args.batch_size),
        "split_processes": int(args.split_processes),
        "notes_per_chunk": int(args.notes_per_chunk),
        "n_chunks": int(n_chunks),
        "n_reused_chunks": int(n_reused_chunks),
        "resume": bool(args.resume),
        "device": device,
        "test_mode": bool(args.test_mode),
        "test_record_limit": TEST_RECORD_LIMIT if args.test_mode else None,
        "n_subjects": n_subjects,
        "n_notes": n_rows,
        "n_hadm_ids": (
            int(key_table[NOTES_HADM_COLUMN].nunique())
            if n_rows and NOTES_HADM_COLUMN in key_table.columns
            else None
        ),
        "n_empty_notes": summary["n_empty_notes"],
        "n_sentences_total": int(sentence_counts.sum()) if n_rows else 0,
        "sentences_per_row_min": int(sentence_counts.min()) if n_rows else None,
        "sentences_per_row_median": float(sentence_counts.median()) if n_rows else None,
        "sentences_per_row_max": int(sentence_counts.max()) if n_rows else None,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_column_prefix": EMBEDDING_COLUMN_PREFIX,
        "embedding_table_path": str(output_paths["embeddings"]),
        "key_table_csv_path": str(output_paths["keys"]),
        "key_columns": list(key_columns),
        "summary": summary,
    }
    if note_grain:
        metadata["bhc_path"] = str(args.bhc_path)
        metadata["discharge_path"] = str(args.discharge_path)
    else:
        metadata["notes_path"] = str(args.notes_path)
    return metadata


def main(argv: list[str] | None = None) -> None:
    start = datetime.now()
    args = parse_args(argv)
    device = resolve_device()
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = output_paths_for(args.output_dir, args.test_mode, TEST_RECORD_LIMIT)

    notes = read_notes_from_note_tables(
        args.bhc_path,
        args.discharge_path,
        test_mode=args.test_mode,
        test_record_limit=TEST_RECORD_LIMIT,
    )
    print(
        f"Loaded {len(notes)} note row(s) covering "
        f"{notes[NOTES_SUBJECT_COLUMN].nunique()} subject(s) and "
        f"{notes[NOTES_HADM_COLUMN].nunique()} admission(s) from: "
        f"{args.bhc_path} + {args.discharge_path}"
    )

    print(f"Loading spaCy sentence splitter: {DEFAULT_SPACY_MODEL}")
    splitter = load_sentence_splitter(n_process=args.split_processes)
    print(f"Loading transformer encoder: {DEFAULT_TRANSFORMER_MODEL}")
    encoder = ClinicalBertSentenceEncoder(device=device)

    embedding_table, key_table, key_columns, n_reused_chunks, empty_records = (
        encode_notes_chunked(
            notes,
            splitter,
            encoder,
            output_paths=output_paths,
            chunk_dir=output_paths["chunks"],
            subjects_per_chunk=args.notes_per_chunk,
            batch_size=args.batch_size,
            resume=args.resume,
            keep_chunks=args.keep_chunks,
            note_grain=True,
            empty_records_path=output_paths["empty_records"],
        )
    )
    print(f"Shape of embedding table: {embedding_table.shape}")
    print(f"Shape of key table: {key_table.shape}")
    if not empty_records.empty:
        print(f"Excluded {len(empty_records)} empty note(s).")

    run_metadata = build_metadata(
        args,
        key_table=key_table,
        key_columns=key_columns,
        output_paths=output_paths,
        device=device,
        n_chunks=len(chunk_bounds(len(notes), args.notes_per_chunk)),
        n_reused_chunks=n_reused_chunks,
        note_grain=True,
        n_empty_notes=int(len(empty_records)),
    )
    with output_paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    print(f"Saved master embedding table to: {output_paths['embeddings']}")
    print(f"Saved embedding key CSV to: {output_paths['keys']}")
    if not empty_records.empty:
        print(f"Saved empty records CSV to: {output_paths['empty_records']}")
    print(f"Saved run metadata to: {output_paths['metadata']}")

    end = datetime.now()
    elapsed = timedelta(seconds=round((end - start).total_seconds()))
    print("End of feature extraction for BHC notes.")
    print(f"Total time taken: {elapsed}")


if __name__ == "__main__":
    main()
