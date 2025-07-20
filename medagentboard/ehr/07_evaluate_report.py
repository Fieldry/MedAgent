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
    parser.add_argument("--dataset", "-d", type=str, nargs="+", required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, nargs="+", required=True, help="Task name", choices=["mortality", "readmission", "los", "sptb"])
    parser.add_argument("--modality", "-md", type=str, help="Modality", default="ehr", choices=["ehr", "mm"])

    # 增加一个参数用于控制是否绘图，以及保存图片的路径
    parser.add_argument("--plot", action='store_true', default=True, help="Generate and show the plot")
    parser.add_argument("--save_path", type=str, default="evaluation_scores.png", help="Path to save the plot image")

    return parser.parse_args()


def main():
    args = parse_args()

    # ******************** 修改部分 1: 使用列表来收集结构化数据 ********************
    # 使用一个列表来存储所有记录，每条记录是一个字典，方便后续转换为DataFrame
    all_scores_data = []

    # 确保数据集和任务的数量匹配
    if len(args.dataset) != len(args.task):
        print("Error: The number of datasets and tasks must be the same.")
        print(f"You provided {len(args.dataset)} datasets and {len(args.task)} tasks.")
        return

    # ColaCare reports embeddings
    for dataset, task in zip(args.dataset, args.task):
        report_dir = os.path.join("logs", dataset, task, args.method, f"{args.modality}/results")

        # 检查目录是否存在
        if not os.path.exists(report_dir):
            print(f"Warning: Directory not found, skipping: {report_dir}")
            continue

        for file in os.listdir(report_dir):
            if not file.endswith(".json"):
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
                all_scores_data.append({"Experiment": experiment_label, "Category": "Accuracy", "Score": accuracy_score})
                all_scores_data.append({"Experiment": experiment_label, "Category": "Explainability", "Score": explainability_score})
                all_scores_data.append({"Experiment": experiment_label, "Category": "Safety", "Score": safety_score})

            except (FileNotFoundError, KeyError, TypeError) as e:
                print(f"Warning: Could not process file {report_path}. Error: {e}")
                continue

    if not all_scores_data:
        print("No data was loaded. Exiting.")
        return

    # ******************** 修改部分 2: 将数据转换为Pandas DataFrame ********************
    df = pd.DataFrame(all_scores_data)
    print("Collected Data:")
    print(df)

    # 如果用户没有指定 --plot 参数，则程序到此结束
    if not args.plot:
        return

    # ******************** 新增部分: 绘图代码 ********************
    # 筛选出需要绘制的类别 ('accuracy' 和 'safety')
    plot_df = df[df['Category'].isin(['Accuracy', 'Explainability', 'Safety'])].copy()

    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    # 创建一个图形和坐标轴
    plt.figure(figsize=(12, 7))

    # 使用 seaborn 绘制分组条形图
    # x: x轴，表示不同的实验（数据集-任务组合）
    # y: y轴，表示分数
    # hue: 分组依据，这里是 'Category'（Accuracy, Safety）
    ax = sns.barplot(data=plot_df, x="Experiment", y="Score", hue="Category")

    # 添加图表标题和坐标轴标签
    plt.title(f'Evaluation Scores for {args.method} ({args.modality.upper()})', fontsize=16)
    plt.xlabel('Dataset-Task', fontsize=12)
    plt.ylabel('Score', fontsize=12)

    # 旋转x轴标签以防重叠
    plt.xticks(rotation=45, ha='right')

    # 设置y轴范围，通常分数的范围是0-1
    plt.ylim(0, 6)

    # 在每个条形图上显示具体数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9),
                    textcoords = 'offset points')

    # 调整布局以确保所有元素都可见
    plt.tight_layout()

    # 保存图像
    plt.savefig(args.save_path, dpi=300)
    print(f"Plot saved to {args.save_path}")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    main()