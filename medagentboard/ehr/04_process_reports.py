import os
import json
import argparse
import numpy as np

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, nargs="+", required=True, help="Model name")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd"])
    parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los"])
    parser.add_argument("--language_model", "-lm", type=str, default="GatorTron", help="Language model to encode reports")

    return parser.parse_args()


def main():
    args = parse_args()
    results = {
        "pids": None,
        "labels": None,
        "ehr_embeddings": None,
        "text_embeddings": [],
        "ehr_scores": []
    }

    # Ehr model outputs
    ehr_embeddings = []
    for model in args.model:
        ehr_outputs_path = os.path.join("logs", args.dataset, args.task, model, "outputs.pkl")
        if not os.path.exists(ehr_outputs_path):
            print(f"Ehr model outputs not found for {model}")
            continue
        ehr_outputs = pd.read_pickle(ehr_outputs_path)
        if results["pids"] is None:
            results["pids"] = ehr_outputs["pids"]
        if results["labels"] is None:
            results["labels"] = ehr_outputs["labels"]
        if "embeddings" in ehr_outputs:
            ehr_embeddings.append(ehr_outputs["embeddings"])
    if len(ehr_embeddings) > 0:
        ehr_embeddings = np.array(ehr_embeddings).transpose(1, 0, 2) # (num_patients, num_models, embedding_dim)
        results["ehr_embeddings"] = ehr_embeddings.tolist()

    train_pids = [item["id"] for item in pd.read_pickle(os.path.join("my_datasets", "ehr", args.dataset, "processed/split/fusion_train_data.pkl"))]
    val_pids = [item["id"] for item in pd.read_pickle(os.path.join("my_datasets", "ehr", args.dataset, "processed/split/fusion_val_data.pkl"))]
    test_pids = [item["id"] for item in pd.read_pickle(os.path.join("my_datasets", "ehr", args.dataset, "processed/split/test_data.pkl"))]

    train_embeddings = {
        "pids": train_pids,
        "labels": [],
        "ehr_embeddings": [],
        "text_embeddings": [],
        "ehr_scores": []
    }
    val_embeddings = {
        "pids": val_pids,
        "labels": [],
        "ehr_embeddings": [],
        "text_embeddings": [],
        "ehr_scores": []
    }
    test_embeddings = {
        "pids": test_pids,
        "labels": [],
        "ehr_embeddings": [],
        "text_embeddings": [],
        "ehr_scores": []
    }

    # ColaCare reports embeddings
    method = "ColaCare"
    model = AutoModel.from_pretrained(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    for i, pid in enumerate(results["pids"]):
        reports_path = os.path.join("logs", args.dataset, args.task, method, "ehr/results", f"ehr_{str(pid)}-result.json")
        if not os.path.exists(reports_path):
            print(f"ColaCare reports not found for {pid}")
            continue
        reports = json.load(open(reports_path))
        final_report = reports["case_history"]["final_decision"]["explanation"]
        if isinstance(final_report, dict):
            final_report = " ".join(list(map(str, final_report.values())))
        if isinstance(final_report, list):
            final_report = " ".join(final_report)

        input_tokens = tokenizer(final_report,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=512,
            padding=True)
        with torch.no_grad():
            outputs = model(**input_tokens)
        text_embedding = outputs.last_hidden_state[0, 0, :].numpy().tolist() # 1024
        results["text_embeddings"].append(text_embedding)
        results["ehr_scores"].append([item["score"] for item in reports["case_history"]["doctor_scores"]])

        if pid in train_pids:
            train_embeddings["labels"].append(results["labels"][i])
            train_embeddings["text_embeddings"].append(text_embedding)
            train_embeddings["ehr_scores"].append([item["score"] for item in reports["case_history"]["doctor_scores"]])
            train_embeddings["ehr_embeddings"].append(ehr_embeddings[i])
        elif pid in val_pids:
            val_embeddings["labels"].append(results["labels"][i])
            val_embeddings["text_embeddings"].append(text_embedding)
            val_embeddings["ehr_scores"].append([item["score"] for item in reports["case_history"]["doctor_scores"]])
            val_embeddings["ehr_embeddings"].append(ehr_embeddings[i])
        elif pid in test_pids:
            test_embeddings["labels"].append(results["labels"][i])
            test_embeddings["text_embeddings"].append(text_embedding)
            test_embeddings["ehr_scores"].append([item["score"] for item in reports["case_history"]["doctor_scores"]])
            test_embeddings["ehr_embeddings"].append(ehr_embeddings[i])

    # Save results
    pd.to_pickle(results, os.path.join("logs", args.dataset, args.task, method, "all_embeddings.pkl"))
    pd.to_pickle(train_embeddings, os.path.join("logs", args.dataset, args.task, method, "train_embeddings.pkl"))
    pd.to_pickle(val_embeddings, os.path.join("logs", args.dataset, args.task, method, "val_embeddings.pkl"))
    pd.to_pickle(test_embeddings, os.path.join("logs", args.dataset, args.task, method, "test_embeddings.pkl"))


if __name__ == "__main__":
    main()