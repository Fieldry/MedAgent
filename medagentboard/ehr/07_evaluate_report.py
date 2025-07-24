import os
import json
import argparse
import numpy as np

# 导入绘图库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", "-m", type=str, required=True, help="Method name", choices=["ColaCare"])
    # 允许传入多个数据集和任务
    parser.add_argument("--dataset", "-d", type=str, nargs="+", required=True, default=["cdsl", "mimic-iv", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, nargs="+", required=True, default=["mortality", "mortality", "readmission", "mortality", "sptb"])
    parser.add_argument("--modality", "-md", type=str, help="Modality", default="ehr", choices=["ehr", "mm"])
    parser.add_argument("--llm", "-lm", type=str, help="LLM name", nargs="+", default=["deepseek-v3-official", "o4-mini", "claude4", "qwen3"])

    return parser.parse_args()


def plot_facet_grid(df, args):
    """
    为每个实验创建一个子图，在子图内部按类别（Accuracy等）比较模型。
    """
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        x="Category",      # X轴是评估维度
        y="Score",         # Y轴是分数
        hue="Model",       # 用颜色区分不同模型
        col="Experiment",  # 按“实验”分列，创建子图
        kind="bar",        # 图表类型为条形图
        height=5,
        aspect=0.8,
        legend_out=True
    )

    # 调整细节
    g.figure.suptitle(f'Evaluation Scores for {args.method} ({args.modality.upper()})', y=1.03, fontsize=16)
    g.set_axis_labels("Score Category", "Score")
    g.set_titles("Experiment: {col_name}")
    g.despine(left=True) # 移除左侧轴线

    # 在每个条形图上显示数值
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "facet_grid.png"), dpi=300)


def plot_radar_charts(df, args):
    """
    为每个实验（Dataset-Task）创建一个雷达图，比较不同模型的表现。
    """
    sns.set_theme(style="whitegrid")
    experiments = df['Experiment'].unique()
    models = df['Model'].unique()
    categories = ['Accuracy', 'Explainability', 'Safety']

    # 创建一个大的图形窗口，准备放多个子图
    fig, axes = plt.subplots(figsize=(7 * len(experiments), 7), nrows=1, ncols=len(experiments),
                             subplot_kw=dict(polar=True))
    if len(experiments) == 1: # 如果只有一个实验，axes不是数组
        axes = [axes]

    # 准备角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1] # 闭合雷达图

    # 为每个实验（子图）绘图
    for ax, experiment in zip(axes, experiments):
        ax.set_title(experiment, size=14, color='blue', y=1.1)

        # 为每个模型画一条线
        for model in models:
            # 提取数据并闭合
            values = df[(df['Experiment']==experiment) & (df['Model']==model)].set_index('Category').loc[categories, 'Score'].tolist()
            values += values[:1]

            # 绘图
            ax.plot(angles, values, marker='o', linestyle='solid', label=model)
            ax.fill(angles, values, alpha=0.1)

        # 设置雷达图细节
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 6) # 根据你的分数范围调整

    # 添加总标题和图例
    fig.suptitle(f'Model Performance Radar for {args.method} ({args.modality.upper()})', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局防止标题重叠
    plt.savefig(os.path.join(args.save_path, "radar_charts.png"), dpi=300)


def plot_heatmaps(df, args):
    """
    为每个评估维度（Accuracy, Explainability, Safety）创建一个热力图。
    """
    sns.set_theme(style="white")
    categories = df['Category'].unique()

    fig, axes = plt.subplots(1, len(categories), figsize=(18, 6), sharey=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        # 数据透视：行为模型，列为实验，值为分数
        pivot_df = df[df['Category'] == category].pivot_table(
            index='Model',
            columns='Experiment',
            values='Score'
        )

        sns.heatmap(
            pivot_df,
            annot=True,     # 在格子上显示数值
            fmt=".2f",      # 格式化为两位小数
            linewidths=.5,
            cmap="viridis", # 选择一个好看的色板
            ax=ax
        )
        ax.set_title(category, fontsize=14)
        ax.set_xlabel('') # 隐藏子图的x轴标签
        ax.set_ylabel('Model') # 只在最左侧显示y轴标签

    fig.suptitle(f'Score Heatmaps for {args.method} ({args.modality.upper()})', fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.save_path, "heatmaps.png"), dpi=300)


def plot_tradeoffs(df, args):
    """
    使用散点图展示 Accuracy 和 Explainability 之间的关系，
    用气泡大小表示 Safety。
    """
    # 需要先将数据从长格式转换为宽格式
    wide_df = df.pivot_table(index=['Experiment', 'Model'], columns='Category', values='Score').reset_index()

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=wide_df,
        x="Accuracy",
        y="Explainability",
        hue="Model",          # 用颜色区分模型
        size="Safety",        # 用气泡大小表示安全性得分
        sizes=(50, 500),      # 控制气泡大小范围
        style="Experiment",   # 用不同形状的点区分实验
        height=7,
        aspect=1.2,
    )

    g.figure.suptitle(f'Performance Trade-offs for {args.method} ({args.modality.upper()})', y=1.03, fontsize=16)
    g.set(xlim=(0, 6), ylim=(0, 6)) # 设置坐标轴范围
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, "tradeoffs.png"), dpi=300)


def main():
    args = parse_args()

    all_scores_data = []

    # 确保数据集和任务的数量匹配
    if len(args.dataset) != len(args.task):
        print("Error: The number of datasets and tasks must be the same.")
        print(f"You provided {len(args.dataset)} datasets and {len(args.task)} tasks.")
        return

    # ColaCare reports embeddings
    for dataset, task in zip(args.dataset, args.task):
        for model in ["deepseek-v3-official", "o4-mini", "claude4", "qwen3"]:
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
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Accuracy", "Score": accuracy_score, "Model": model})
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Explainability", "Score": explainability_score, "Model": model})
                    all_scores_data.append({"Experiment": experiment_label, "Category": "Safety", "Score": safety_score, "Model": model})

                except (FileNotFoundError, KeyError, TypeError) as e:
                    print(f"Warning: Could not process file {report_path}. Error: {e}")
                    continue

        df = pd.DataFrame(all_scores_data)
        for model in ["deepseek-v3-official", "o4-mini", "claude4", "qwen3"]:
            tmp_df = df[df["Model"] == model]
            print(len(tmp_df))

        save_path = os.path.join("logs", dataset, task, args.method)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.save_path = save_path

        plot_facet_grid(df, args)
        plot_heatmaps(df, args)
        plot_tradeoffs(df, args)
        plot_radar_charts(df, args)


if __name__ == "__main__":
    main()