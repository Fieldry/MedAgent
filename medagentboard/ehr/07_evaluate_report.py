import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", type=str, default="ColaCare", help="Method name", choices=["ColaCare"])
    parser.add_argument("--dataset", "-d", type=str, nargs="+", default=["mimic-iv", "mimic-iv"])
    parser.add_argument("--task", "-t", type=str, nargs="+", default=["mortality", "readmission"])
    parser.add_argument("--modality", "-md", type=str, help="Modality", default="ehr", choices=["ehr", "mm"])
    parser.add_argument("--llm", "-lm", type=str, help="LLM name", nargs="+", default=["deepseek-v3-official", "deepseek-r1-official", "o4-mini", "claude4", "qwen3"])

    return parser.parse_args()


def plot_facet_grid(df, args):
    """
    为每个实验创建一个子图，在子图内部按类别（Accuracy等）比较模型。
    """
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        x="Category",
        y="Score",
        hue="Model",
        col="Experiment",
        kind="bar",
        height=8,
        aspect=0.9,
        legend_out=True
    )

    g.figure.subplots_adjust(top=0.88, right=0.85)
    g.figure.suptitle(f'Evaluation Scores for {args.method}\'s Final Report', y=0.99, fontsize=20)

    sns.move_legend(
        g, "upper right",
        bbox_to_anchor=(0.98, 0.95),
        title='Model',
        frameon=True
    )

    g.set_axis_labels("Score Category", "Score")
    g.set_titles("Experiment: {col_name}\n\nEvaluator: DeepSeek-V3")
    g.despine(left=True)

    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() < 0.5:
                continue
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    plt.savefig(os.path.join(args.save_path, "facet_grid_adjusted.png"), dpi=300)


def main():
    args = parse_args()

    if len(args.dataset) != len(args.task):
        print("Error: The number of datasets and tasks must be the same.")
        print(f"You provided {len(args.dataset)} datasets and {len(args.task)} tasks.")
        return

    for dataset, task in zip(args.dataset, args.task):
        all_scores_data = []
        for model in ["deepseek-v3-official", "deepseek-r1-official", "o4-mini", "claude4", "qwen3"]:
            report_dir = os.path.join("logs", dataset, task, args.method, f"{args.modality}_{model}/results")

            # 检查目录是否存在
            if not os.path.exists(report_dir):
                print(f"Warning: Directory not found, skipping: {report_dir}")
                continue

            for file in os.listdir(report_dir):
                if not file.endswith(".json") or "worag" in file:
                    continue

                report_path = os.path.join(report_dir, file)
                try:
                    with open(report_path) as f:
                        reports = json.load(f)

                    # 提取分数
                    eval_results = reports["case_history"]["report_trustworthiness_evaluation"]
                    accuracy_score = eval_results["accuracy"]["score"]
                    explainability_score = eval_results["explainability"]["score"]
                    safety_score = eval_results["safety"]["score"]

                    # 为每个分数类别创建一条记录
                    experiment_label = f"{dataset}-{task}"
                    if "deepseek" in model:
                        model = model.replace("deepseek", "ds").replace("-official", "")
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Accuracy", "Score": accuracy_score, "Model": model})
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Explainability", "Score": explainability_score, "Model": model})
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Safety", "Score": safety_score, "Model": model})

                except (FileNotFoundError, KeyError, TypeError) as e:
                    print(f"Warning: Could not process file {report_path}. Error: {e}")
                    continue

        df = pd.DataFrame(all_scores_data)

        save_path = os.path.join("logs", dataset, task, args.method)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.save_path = save_path

        plot_facet_grid(df, args)


if __name__ == "__main__":
    main()