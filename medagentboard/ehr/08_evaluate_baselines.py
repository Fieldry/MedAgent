import os
import json
import argparse
from typing import Dict

import pandas as pd

from pyehr.utils.bootstrap import run_bootstrap


EMBEDDING_LOG_DIR = "logs"
DATASET_DIR = "my_datasets"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process EHR and Text data to generate embeddings.")
    parser.add_argument("--dataset", "-d", type=str, default="esrd", choices=["cdsl", "mimic-iv", "esrd", "obstetrics"], help="Dataset name.")
    parser.add_argument("--task", "-t", type=str, default="mortality", choices=["mortality", "readmission", "los", "sptb"], help="Task name.")
    parser.add_argument("--framework", "-f", type=str, nargs="+", default=["MedAgent", "ReConcile", "ZeroShotLLM", "FewShotLLM"], help="Framework name.")
    return parser.parse_args()


def load_dataset(dataset: str, task: str) -> Dict[str, list]:
    """Loads and aggregates outputs from multiple EHR models."""
    if dataset == "obstetrics":
        split = "solo"
    else:
        split = "split"
    base_path = os.path.join(DATASET_DIR, "ehr", dataset, f"processed/{split}")
    test_data = pd.read_pickle(os.path.join(base_path, "test_data.pkl"))
    test_pids = [item["id"] for item in test_data]
    test_labels = [item[f"y_{task}"][-1] for item in test_data]

    print(f"Loaded PIDs: {len(test_pids)} test.")
    return test_pids, test_labels


def main():
    """Main function to orchestrate the data processing pipeline."""
    args = parse_args()

    print(f"Evaluating {args.dataset} {args.task} with {args.framework}.")

    # 1. Load and aggregate EHR model outputs
    test_pids, test_labels = load_dataset(args.dataset, args.task)

    all_perf_df = pd.DataFrame()
    for framework in args.framework:
        report_dir = os.path.join(EMBEDDING_LOG_DIR, args.dataset, args.task, framework, "ehr_deepseek-v3-official")
        preds = []
        for pid in test_pids:
            report_path = os.path.join(report_dir, "results", f"ehr_{pid}-result.json")
            if not os.path.exists(report_path):
                print(f"Warning: Report not found for PID {pid}. Skipping.")
                continue

            with open(report_path, 'r') as f:
                report_data = json.load(f)
                if framework == "MedAgent":
                    preds.append(report_data["case_history"]["final_decision"]["answer"])
                elif framework == "ReConcile":
                    preds.append(report_data["case_history"]["discussion_history"][-1]["final_prediction"])
                elif framework in ["ZeroShotLLM", "FewShotLLM"]:
                    preds.append(report_data["predicted_value"])
                else:
                    raise ValueError(f"Unsupported framework: {framework}")

        perf_boot = run_bootstrap(preds, test_labels, {"task": args.task, "los_info": None})
        for key, value in perf_boot.items():
            if args.task in ["mortality", "readmission", "sptb"]:
                perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'
            else:
                perf_boot[key] = f'{value["mean"]:.2f}±{value["std"]:.2f}'
        perf_boot = dict({
            "model": framework,
            "dataset": args.dataset,
            "task": args.task,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        all_perf_df = pd.concat([all_perf_df, perf_df], ignore_index=True)
    save_dir = os.path.join(EMBEDDING_LOG_DIR, args.dataset, args.task, "baseline")
    os.makedirs(save_dir, exist_ok=True)
    all_perf_df.to_csv(os.path.join(save_dir, "08_baseline_performance.csv"), index=False)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()