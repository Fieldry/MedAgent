import os
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

from pyehr.utils.bootstrap import run_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, nargs="+", required=True, help="Model name (2 or 3 models recommended)")
parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd", "obstetrics"])
parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los", "sptb"])
args = parser.parse_args()

base_dir = 'logs'
model_names = args.model
dataset = args.dataset
task = args.task

# Load data

all_labels = []
model_preds = {model: [] for model in model_names}
model_correctness = {model: [] for model in model_names}
total_samples = 0

print("Loading and aligning prediction data...")
for i in range(10):
    first_model_file_path = os.path.join(base_dir, dataset, task, model_names[0], f'fold_{i}', 'outputs.pkl')
    with open(first_model_file_path, 'rb') as f:
        data = pickle.load(f)
        labels = np.array(data['labels'])
        all_labels.append(labels)

        print(f"Fold {i} loaded {len(labels)} samples")

    for model_name in model_names:
        file_path = os.path.join(base_dir, dataset, task, model_name, f'fold_{i}', 'outputs.pkl')

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        preds = np.array(data['preds'])
        model_preds[model_name].append(preds)

        correct_mask = ((preds >= 0.5) == labels)
        model_correctness[model_name].append(correct_mask)

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