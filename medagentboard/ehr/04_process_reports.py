import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Set

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

METHOD_NAME = "ColaCare"
EMBEDDING_LOG_DIR = "logs"
DATASET_DIR = "my_datasets"

PatientID = Any # Can be int or str depending on the dataset
DataDict = Dict[str, List[Any]]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process EHR and Text data to generate embeddings.")
    parser.add_argument("--model", "-m", type=str, nargs="+", required=True, help="EHR model names for feature extraction.")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["tjh", "mimic-iv", "esrd", "obstetrics"], help="Dataset name.")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "los", "sptb"], help="Task name.")
    parser.add_argument("--language_model", "-lm", type=str, default="GatorTron", help="Language model to encode reports.")
    parser.add_argument("--modality", "-md", type=str, default="ehr", choices=["ehr", "mm"], help="Modality.")
    return parser.parse_args()

def load_ehr_model_outputs(model_names: List[str], dataset: str, task: str) -> Dict[str, list]:
    """Loads and aggregates outputs from multiple EHR models."""
    print("Loading EHR model outputs...")
    results = {"pids": None, "labels": None}
    ehr_embeddings, ehr_preds = [], []

    for model_name in model_names:
        outputs_path = os.path.join(EMBEDDING_LOG_DIR, dataset, task, model_name, "outputs.pkl")
        if not os.path.exists(outputs_path):
            print(f"Warning: EHR model outputs not found for '{model_name}'. Skipping.")
            continue

        ehr_outputs = pd.read_pickle(outputs_path)

        if results["pids"] is None:
            results["pids"] = ehr_outputs["pids"]
            results["labels"] = ehr_outputs["labels"]

        if "embeddings" in ehr_outputs:
            ehr_embeddings.append(ehr_outputs["embeddings"])
        if "preds" in ehr_outputs:
            ehr_preds.append(ehr_outputs["preds"])

    if ehr_embeddings:
        # (num_models, num_patients, dim) -> (num_patients, num_models, dim)
        results["ehr_embeddings"] = np.array(ehr_embeddings).transpose(1, 0, 2).tolist()
    if ehr_preds:
        # (num_models, num_patients) -> (num_patients, num_models)
        results["ehr_preds"] = np.array(ehr_preds).transpose(1, 0).tolist()

    print(f"Loaded EHR data for {len(results.get('pids', []))} patients.")
    return results

def load_pid_splits(dataset: str) -> Tuple[Set[PatientID], Set[PatientID], Set[PatientID]]:
    """Loads patient IDs for train, validation, and test splits."""
    split_folder = "split" if dataset != "obstetrics" else "solo"
    base_path = os.path.join(DATASET_DIR, "ehr", dataset, f"processed/{split_folder}")

    train_pids = set(item["id"] for item in pd.read_pickle(os.path.join(base_path, "fusion_train_data.pkl")))
    val_pids = set(item["id"] for item in pd.read_pickle(os.path.join(base_path, "fusion_val_data.pkl")))
    test_pids = set(item["id"] for item in pd.read_pickle(os.path.join(base_path, "test_data.pkl")))

    print(f"Loaded PIDs: {len(train_pids)} train, {len(val_pids)} validation, {len(test_pids)} test.")
    return train_pids, val_pids, test_pids

def _process_final_report(final_report: Any) -> str:
    """Helper to convert final_report object into a single string."""
    if isinstance(final_report, dict):
        return " ".join(map(str, final_report.values()))
    if isinstance(final_report, list):
        return " ".join(map(str, final_report))
    return str(final_report)

def generate_text_embeddings_and_scores(pids: List[PatientID], args: argparse.Namespace) -> Dict[PatientID, Dict[str, Any]]:
    """Generates text embeddings and extracts scores from ColaCare reports."""
    print("Generating text embeddings and scores...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.language_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)

    pid_to_text_data = {}

    for pid in pids:
        report_path = os.path.join(EMBEDDING_LOG_DIR, args.dataset, args.task, METHOD_NAME,
                                   f"{args.modality}/results", f"ehr_{pid}-result.json")
        if not os.path.exists(report_path):
            print(f"Warning: Report not found for PID {pid}. Skipping.")
            continue

        with open(report_path, 'r') as f:
            report_data = json.load(f)

        final_report_text = _process_final_report(report_data["case_history"]["final_decision"]["explanation"])

        inputs = tokenizer(
            final_report_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token embedding
        text_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy().tolist()
        ehr_scores = [item["score"] for item in report_data["case_history"]["doctor_scores"]]

        pid_to_text_data[pid] = {"text_embedding": text_embedding, "ehr_scores": ehr_scores}

    print(f"Processed text data for {len(pid_to_text_data)} patients.")
    return pid_to_text_data

def _create_dataset_dict(pids: List[PatientID] = None) -> DataDict:
    """Creates a standard dictionary structure for storing dataset features."""
    return {
        "pids": pids or [],
        "labels": [],
        "ehr_preds": [],
        "ehr_embeddings": [],
        "text_embeddings": [],
        "ehr_scores": []
    }

def save_results(data: Dict[str, DataDict], base_save_path: str):
    """Saves the processed data dictionaries to pickle files."""
    print("Saving results...")
    os.makedirs(base_save_path, exist_ok=True)

    for name, content in data.items():
        save_path = os.path.join(base_save_path, f"{name}_embeddings.pkl")
        pd.to_pickle(content, save_path)
        print(f"Saved {len(content['pids'])} patient embeddings to {save_path}")

def main():
    """Main function to orchestrate the data processing pipeline."""
    args = parse_args()

    # 1. Load and aggregate EHR model outputs
    ehr_data = load_ehr_model_outputs(args.model, args.dataset, args.task)
    if not ehr_data.get("pids"):
        print("Error: No patient data loaded from EHR models. Exiting.")
        return

    # 2. Load PID splits
    train_pids, val_pids, test_pids = load_pid_splits(args.dataset)

    # 3. Generate text embeddings and scores for all patients
    pid_to_text_data = generate_text_embeddings_and_scores(ehr_data["pids"], args)

    # 4. Initialize data structures for each split
    all_pids_list = ehr_data["pids"]
    split_datasets = {
        "train": _create_dataset_dict(list(train_pids)),
        "val": _create_dataset_dict(list(val_pids)),
        "test": _create_dataset_dict(list(test_pids)),
        "all": _create_dataset_dict(all_pids_list)
    }

    # 5. Distribute data into respective splits
    print("Assembling and splitting final datasets...")
    pid_to_split = {pid: "train" for pid in train_pids}
    pid_to_split.update({pid: "val" for pid in val_pids})
    pid_to_split.update({pid: "test" for pid in test_pids})

    for i, pid in enumerate(all_pids_list):
        if pid not in pid_to_text_data:
            continue # Skip patients for whom text processing failed

        text_data = pid_to_text_data[pid]

        # Append to the "all" dataset
        split_datasets["all"]["labels"].append(ehr_data["labels"][i])
        split_datasets["all"]["ehr_preds"].append(ehr_data["ehr_preds"][i])
        split_datasets["all"]["ehr_embeddings"].append(ehr_data["ehr_embeddings"][i])
        split_datasets["all"]["text_embeddings"].append(text_data["text_embedding"])
        split_datasets["all"]["ehr_scores"].append(text_data["ehr_scores"])

        # Append to the correct split (train/val/test)
        split_name = pid_to_split.get(pid)
        if split_name:
            target_dict = split_datasets[split_name]
            target_dict["labels"].append(ehr_data["labels"][i])
            target_dict["ehr_preds"].append(ehr_data["ehr_preds"][i])
            target_dict["ehr_embeddings"].append(ehr_data["ehr_embeddings"][i])
            target_dict["text_embeddings"].append(text_data["text_embedding"])
            target_dict["ehr_scores"].append(text_data["ehr_scores"])

    # 6. Save all results
    base_save_path = os.path.join(EMBEDDING_LOG_DIR, args.dataset, args.task, METHOD_NAME, args.modality)
    save_results(split_datasets, base_save_path)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()