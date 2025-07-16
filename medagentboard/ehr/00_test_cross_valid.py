import os
import pickle
import argparse

import numpy as np
import pandas as pd
import lightning as L
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from lightning.pytorch.loggers import CSVLogger
import torch

from pyehr.datasets.utils.datamodule import EhrDataModule
from pyehr.pipelines.dl import DlPipeline
from pyehr.utils.bootstrap import run_bootstrap
from pyehr.utils.calibration import find_optimal_threshold


def find_best_model_path(log_dir: str):
    versions = []
    for file in os.listdir(log_dir):
        if file.endswith('.ckpt'):
            version = file.split('.')[0]
            if 'v' in version:
                versions.append(version.split('v')[-1])
    versions.sort(key=lambda x: int(x), reverse=True)
    latest_version = versions[0]
    return os.path.join(log_dir, f"best-v{latest_version}.ckpt")


def run_dl_experiment(config):
    # data
    version = f"{config['split']}/fold_{config['fold']}" if "split" in config else f"fold_{config['fold']}"
    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/{version}'
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"])

    # los infomation
    los_info = pd.read_pickle(os.path.join(dataset_path, "los_info.pkl")) if config["task"] == "los" else None
    config["los_info"] = los_info

    # logger
    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}/{config["model"]}', version=version)

    # seed for reproducibility
    L.seed_everything(42)

    # train/val/test
    pipeline = DlPipeline(config)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1
    trainer = L.Trainer(accelerator=accelerator, devices=devices, max_epochs=1, logger=logger)

    # Load best model checkpoint
    best_model_path = find_best_model_path(os.path.join("logs", f"{config['dataset']}/{config['task']}/{config['model']}/fold_{config['fold']}/checkpoints"))
    print("best_model_path:", best_model_path)
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    val_loader = dm.val_dataloader()

    # Calibration
    if config["task"] in ["mortality", "readmission", "sptb"]:
        print("Fitting DL classification calibration (temperature scaling)...")
        pipeline.fit_calibration_temperature(val_loader) # pipeline 实例上调用

    # Optimize decision threshold
    if config["task"] in ["mortality", "readmission", "sptb"]:
        print("Optimizing decision threshold for DL model...")
        calibrated_probs_val, true_labels_val = pipeline.get_calibrated_probs_and_labels_for_threshold_tuning(val_loader)

        if calibrated_probs_val.numel() > 0 and true_labels_val.numel() > 0:
            calibrated_probs_val_np = calibrated_probs_val.cpu().numpy()
            true_labels_val_np = true_labels_val.cpu().numpy()

            best_threshold, _ = find_optimal_threshold(true_labels_val_np, calibrated_probs_val_np, metric='f1')
            print(f"DL Optimal threshold: {best_threshold:.4f}")
            pipeline.optimal_decision_threshold = best_threshold
        else:
            print("Warning: Not enough data for DL threshold optimization. Using default 0.5.")
            pipeline.optimal_decision_threshold = getattr(pipeline, 'optimal_decision_threshold', 0.5)

    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return config, perf, outs, pipeline.optimal_decision_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, nargs="+", required=True, help="Model name (2 or 3 models recommended)")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los", "sptb"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    base_dir = 'logs'
    model_names = args.model
    dataset = args.dataset
    task = args.task

    config = {
        "dataset": args.dataset,
        "task": args.task,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "output_dim": args.output_dim,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }

    # Set the input dimensions based on the dataset
    if args.dataset == "mimic-iv":
        config["demo_dim"] = 2
        config["lab_dim"] = 42
    elif args.dataset == "esrd":
        config["demo_dim"] = 0
        config["lab_dim"] = 17
    elif args.dataset == "obstetrics":
        config["demo_dim"] = 0
        config["lab_dim"] = 32
        config["split"] = "solo"
    else:
        raise ValueError("Unsupported dataset. Choose either 'tjh' or 'mimic-iv' or 'esrd' or 'obstetrics'.")

    # Load data
    all_labels = []
    model_preds = {model: [] for model in model_names}
    model_correctness = {model: [] for model in model_names}
    total_samples = 0

    print("Loading and aligning prediction data...")
    for model in model_names:
        config["model"] = model

        for i in range(10):
            config["fold"] = i

            # Print the configuration
            print("Configuration:")
            for key, value in config.items():
                print(f"{key}: {value}")

            # Run the experiment
            try:
                config, perf, outs, threshold = run_dl_experiment(config)
            except Exception as e:
                print(f"Error occurred while running the experiment for model {model}.")
                import traceback
                traceback.print_exc()
                continue

            all_labels.append(outs['labels'])
            model_preds[model].append(outs['preds'])
            model_correctness[model].append(outs['preds'] >= threshold)

    for model_name in model_names:
        model_preds[model_name] = np.concatenate(model_preds[model_name])
        model_correctness[model_name] = np.concatenate(model_correctness[model_name])

    all_labels = np.concatenate(all_labels)
    total_samples = len(all_labels)
    print(f"Data loaded. Total samples across 10 folds: {total_samples}")

    # Compute bootstrap metrics
    perf_all_df = pd.DataFrame()
    for model_name in model_names:
        perf_boot = run_bootstrap(model_preds[model_name], all_labels, {"task": task, "los_info": None})

        for key, value in perf_boot.items():
            if task in ["mortality", "readmission", "sptb"]:
                perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'
            else:
                perf_boot[key] = f'{value["mean"]:.2f}±{value["std"]:.2f}'

        perf_boot = dict({
            "model": model_name,
            "dataset": dataset,
            "task": task,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        perf_all_df = pd.concat([perf_all_df, perf_df], ignore_index=True)

    print(perf_all_df)

    # Plot venn diagram
    plt.figure(figsize=(12, 8))

    if len(model_names) == 2:
        m1_correct = model_correctness[model_names[0]]
        m2_correct = model_correctness[model_names[1]]

        # 计算交集大小
        # (A and not B), (B and not A), (A and B)
        venn2(
            subsets=(np.sum(m1_correct & ~m2_correct),
                    np.sum(m2_correct & ~m1_correct),
                    np.sum(m1_correct & m2_correct)),
            set_labels=model_names
        )
        title = f'Prediction Agreement between {model_names[0]} and {model_names[1]}'

    elif len(model_names) == 3:
        m1_correct = model_correctness[model_names[0]]
        m2_correct = model_correctness[model_names[1]]
        m3_correct = model_correctness[model_names[2]]

        # 计算维恩图的7个区域的大小
        # 顺序: (A), (B), (A&B), (C), (A&C), (B&C), (A&B&C)
        subsets = (
            np.sum(m1_correct & ~m2_correct & ~m3_correct), # A only
            np.sum(~m1_correct & m2_correct & ~m3_correct), # B only
            np.sum(m1_correct & m2_correct & ~m3_correct),  # A and B
            np.sum(~m1_correct & ~m2_correct & m3_correct), # C only
            np.sum(m1_correct & ~m2_correct & m3_correct),  # A and C
            np.sum(~m1_correct & m2_correct & m3_correct),  # B and C
            np.sum(m1_correct & m2_correct & m3_correct)   # A, B and C
        )

        venn3(subsets=subsets, set_labels=model_names)
        title = f'Prediction Agreement among {", ".join(model_names)}'

    all_correct_count = np.sum(m1_correct & m2_correct & m3_correct)
    all_incorrect_count = np.sum(~m1_correct & ~m2_correct & (~m3_correct if len(model_names) == 3 else True))


    plt.title(title, fontsize=16)
    plt.text(
        x=0.01,
        y=0.01,
        s='\n'.join([f'{model}: {auroc}' for model, auroc in zip(model_names, perf_all_df["auroc"].values)]),
        fontsize=12,
        ha='left',
        va='bottom',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    )
    plt.text(
        x=0.99,
        y=0.01,
        s=f'Total Samples: {total_samples}' \
            f'\nAll model correct: {all_correct_count}({all_correct_count / total_samples * 100:.2f}%)' \
            f'\nAll models incorrect: {all_incorrect_count}({all_incorrect_count / total_samples * 100:.2f}%)',
        fontsize=12,
        ha='right',
        va='bottom',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    )
    plt.show()