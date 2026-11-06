import os
import pandas as pd
import torchxrayvision as xrv
import cv2
from pathlib import Path
import skimage, skimage.io
import argparse
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm 
from typing import Any
import json

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "MIMIC-IV"
DEFAULT_CXR_ROOT = DEFAULT_DATA_ROOT / "mimic-cxr-jpg"

DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "cxr"
TEST_RECORD_LIMIT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DenseNet (torchxrayvision/densenet121-res224-chex) embeddings for MIMIC-CXR-JPG as FuseMoE Han et al. 2024"
        )
    )
    parser.add_argument(
        "--cxr-root",
        type=Path,
        default=DEFAULT_CXR_ROOT,
        help="Input MIMIC-CXR dataset folder containing both metadata and images.",
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
        help=f"Process only the first {TEST_RECORD_LIMIT} records from mimic-cxr-2.0.0-metadata.csv.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Use auto to select CUDA when available.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="densenet121-res224-chex",
        help="Name of torchxrayvision DenseNet checkpoint to use.",
    )
    return parser.parse_args()

def resolve_device(device_arg: str) -> str:
    """Return the requested torch device string."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg

def pack_image_path(row: pd.Series, source_dir: str) -> str:
    """Generate path to the image from a row in the meta data file"""
    curr_subject_id = int(row['subject_id'])
    curr_study_id = int(row['study_id'])
    curr_dicom_id = row['dicom_id']

    subfolder_path = "p" + str(curr_subject_id)[0:2]
    patient_path = "p" + str(curr_subject_id)
    study_path = "s" + str(curr_study_id)
    curr_img_path = os.path.join(
        source_dir, 'files', subfolder_path, patient_path, study_path, 
        curr_dicom_id + ".jpg")
    path_store =  os.path.join('files', subfolder_path, patient_path, study_path, 
        curr_dicom_id + ".jpg")
    return curr_img_path, path_store

def generate_process_embedding(img: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Generate image embedding using the densenet model and process it as in FuseMoE Han et al. 2024"""
    embedding = model.features(img)
    # print(f"DenseNet original embedding shape: {embedding.shape}")
    embedding = F.relu(embedding, inplace=True)
    embedding = F.adaptive_avg_pool2d(embedding, (1, 1))
    embedding = embedding.cpu().detach().numpy().reshape(-1)
    return embedding

def split_embeddings_to_features(meta_data: pd.DataFrame) -> pd.DataFrame:
    """Split the embedding column to individual feature columns"""
    num_features = meta_data['embedding'].iloc[0].shape[0]
    print(f"Number of CXR features extracted by DenseNet: {num_features}")
    feature_names = [f"cxr_ft{i}" for i in range(num_features)]
    embedding_split = pd.DataFrame(meta_data['embedding'].to_list(), columns = feature_names)
    meta_data = pd.concat([meta_data, embedding_split], axis = 1)
    return meta_data

def split_embeddings_to_features_batching(meta_data: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """Split the embedding column to individual feature columns in batches"""
    num_features = meta_data['embedding'].iloc[0].shape[0]
    print(f"Number of CXR features extracted by DenseNet: {num_features}")
    feature_names = [f"cxr_ft{i}" for i in range(num_features)]
    embedding_splits = list()
    num_chunks = len(meta_data) // chunk_size + 1
    print(f"The dataframe is split into {num_chunks} chunks of {chunk_size} rows.")
    for i in tqdm(range(num_chunks), total=num_chunks):
        chunk = meta_data[i*chunk_size:(i+1)*chunk_size]
        embedding_split = pd.DataFrame(chunk['embedding'].to_list(), columns = feature_names)
        embedding_splits.append(embedding_split)
    embedding_splits_df = pd.concat(embedding_splits, axis = 0, ignore_index = True)
    meta_data = pd.concat([meta_data, embedding_splits_df], axis = 1)
    return meta_data


def metadata_json_creation(
    args: argparse.Namespace,
    meta_data: pd.DataFrame,
    embedding_pkl_path: str,
    key_csv_path: str,
    key_columns: list,
    device: str,
    ) -> dict[str, Any]:
    """Create meta_data dictionary for embedding extraction."""
    summary = {
        "cxr_root": str(args.cxr_root),
        "torchxrayvision_model": str(args.model_checkpoint),
        "device": device,
        "test_mode": bool(args.test_mode),
        "test_record_limit": TEST_RECORD_LIMIT if args.test_mode else None,
        "n_cxrs": int(len(meta_data)),
        "n_subjects": (
            int(meta_data["subject_id"].nunique())
            if "subject_id" in meta_data.columns and not meta_data.empty
            else None
        ),
        "embedding_dim": int(meta_data['embedding'].iloc[0].shape[0]),
        "embedding_column_prefix": "cxr_ft",
        "embedding_table_path": str(embedding_pkl_path),
        "key_table_csv_path": str(key_csv_path),
        "key_columns": key_columns,
        "processed": True,
        }
    return summary

# def create_datetime_str(date, time):
#     """Create datetime string from StudyDate and StudyTime"""
#     date = str(date)
#     time = str(round(time))
#     if len(time) != 6: time = "0" + time
#     datetime = (date[0:4] + "-" + date[4:6] + "-" + date[6:8] + " " + 
#     time[0:2] + ":" + time[2:4] + ":" + time[4:6])
#     return datetime

def create_update_datatime(meta_data):
    study_date = meta_data["StudyDate"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    study_time = meta_data["StudyTime"].astype(str).str.split(".", regex=False).str[0].str.zfill(6)

    meta_data["StudyDate"] = pd.to_datetime(study_date, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    meta_data["StudyTime"] = pd.to_datetime(study_time, format="%H%M%S", errors="coerce").dt.time
    meta_data["cxr_time"] = pd.to_datetime(meta_data["StudyDate"] + " " + meta_data["StudyTime"].astype(str), errors="coerce")
    return meta_data

def main() -> None:
    start = datetime.now()
    args = parse_args()
    device = resolve_device(args.device)

    # Set names for the files for embeddings, keys and run metadata
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.test_mode:
        embedding_pkl_path = os.path.join(args.output_dir, f"cxr_embeddings_test{TEST_RECORD_LIMIT}.pkl")
        key_csv_path = os.path.join(args.output_dir, f"cxr_embedding_keys_test{TEST_RECORD_LIMIT}.csv")
        run_metadata_json_path = args.output_dir / f"cxr_embedding_metadata_test{TEST_RECORD_LIMIT}.json"

    else:
        embedding_pkl_path = os.path.join(args.output_dir, "cxr_embeddings.pkl")
        key_csv_path = os.path.join(args.output_dir, "cxr_embedding_keys.csv")
        run_metadata_json_path = args.output_dir / f"cxr_embedding_metadata.json"

    # Load the meta data of CXR (of the selected subset if in test model)
    meta_path = os.path.join(args.cxr_root, "mimic-cxr-2.0.0-metadata.csv")
    meta_data = pd.read_csv(meta_path)
    print(f"Shape of meta data file: {meta_data.shape}")
    print(f"Total number of CXR images: {meta_data.shape[0]}")
    columns_meta = list(meta_data.columns) + ['path']

    if args.test_mode:
        meta_data = meta_data.head(TEST_RECORD_LIMIT)

    print(f"Number of CXR images selected for processing: {meta_data.shape[0]}")

    # Load selected denset weight
    print(f"DenseNet model weights: {args.model_checkpoint}")
    model = xrv.models.DenseNet(weights = args.model_checkpoint)
    model.eval()
    model = model.to(device)

    # Extract densenet features and predictions
    meta_data['path'] = None
    meta_data['embedding'] = None
    meta_data['predictions'] = None

    for index, row in tqdm(meta_data.iterrows(), total=meta_data.shape[0]):
        curr_img_path, path_store = pack_image_path(row, args.cxr_root)
        meta_data.at[index, 'path'] = path_store

        if os.path.exists(curr_img_path):
            img = skimage.io.imread(curr_img_path)

            img = xrv.datasets.normalize(img, 255)
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            img = img[None, :, :]

            with torch.no_grad():
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)

                embedding = generate_process_embedding(
                    img = img,
                    model = model)
                
                meta_data.at[index, 'embedding'] = embedding # append to list of dense features for all images

                # Record model predictions of the probability of diseases and conditions selected by torchxrayvision
                """
                List of torchxrayvision outcomes:
                Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema,
                Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration,
                Lung Lesion, Lung Opacity, Mass, Nodule, Pleural_Thickening, Pneumonia,
                Pneumothorax
                """

                preds = model(img).cpu()
                predictions = preds[0].detach().numpy()
                meta_data.at[index, 'predictions'] = predictions

    # Split embeddings into feature columns and combine the study date and time.
    print("Standardise and merge the date and time columns.")
    meta_data = create_update_datatime(meta_data)
    print(f"Split the embedding column to {int(meta_data['embedding'].iloc[0].shape[0])} cxr_ft columns.")
    if len(meta_data) > 10000:
        meta_data = split_embeddings_to_features_batching(meta_data, chunk_size=2000)
    else:
        meta_data = split_embeddings_to_features(meta_data)
    columns_meta = columns_meta + ["cxr_time"]

    # Store the embeddings to .pkl, the keys to .csv and the run metadata to .json
    print(f"Shape of embeddings.pkl: {meta_data.shape}")
    meta_data.to_pickle(embedding_pkl_path)
    print(f"Save CXR embeddings to: {embedding_pkl_path}")

    keys = meta_data[columns_meta]
    print(f"Shape of keys.csv: {keys.shape}")
    keys.to_csv(key_csv_path, index=False)
    print(f"Save keys to: {key_csv_path}")

    run_metadata_json = metadata_json_creation(
        args=args,
        meta_data=meta_data,
        embedding_pkl_path=embedding_pkl_path,
        key_csv_path=key_csv_path,
        key_columns=columns_meta,
        device=device,
    )
    print("Run metadata:")
    print(run_metadata_json)
    with run_metadata_json_path.open("w", encoding="utf-8") as handle:
        json.dump(run_metadata_json, handle, indent=2) # , default=convert
    print(f"Save run metadata to: {run_metadata_json_path}")
    

    end = datetime.now()

    print("End of feature extractions for CXR.")
    print(f"Total time taken: {end - start}")


if __name__ == "__main__":
    main()
        
        
        
    


