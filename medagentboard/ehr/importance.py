import os
import argparse

import pandas as pd
import shap
import torch
import numpy as np
import lightning as L

from pyehr.datasets.utils.datamodule import EhrDataModule
from pyehr.pipelines import DlPipeline, MlPipeline
from pyehr.utils.bootstrap import run_bootstrap

shap.initjs()
torch.set_grad_enabled(True)


def get_feature_importance(config, x_bg, x_shap, device="cuda:0"):
    """计算特征重要性"""
    # 加载模型
    if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost", "LR"]:
        pipeline = MlPipeline.load_from_checkpoint(config["checkpoint_path"], config=config)
    else:
        pipeline = DlPipeline.load_from_checkpoint(config["checkpoint_path"], map_location=device, config=config)

    def predict(x):
        output = pipeline.predict(x, device)
        return output

    # 使用K-means选择背景数据
    k = 2 if config["dataset"] == 'esrd' else 4
    x_bg_kmeans = shap.kmeans(x_bg, k=k).data

    # 计算SHAP值
    explainer = shap.KernelExplainer(predict, x_bg_kmeans)
    shap_values = explainer.shap_values(x_shap).squeeze()
    return shap_values


def run_importance_analysis(config):
    """运行特征重要性分析"""
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # 加载数据
    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/split'
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"])

    # 获取背景数据
    x_bg, y_bg, lens_bg, pid_bg = next(iter(dm.train_dataloader()))
    x_bg = x_bg[:, -1, :].detach().cpu().numpy()

    # 计算SHAP值
    shap_values = None
    for batch in dm.test_dataloader():
        x_shap = batch[0]
        x_shap = x_shap[:, -1, :].detach().cpu().numpy()
        shap_v = get_feature_importance(config, x_bg, x_shap, device)
        shap_values = shap_v if shap_values is None else np.concatenate((shap_values, shap_v), axis=0)

    # 保存结果
    save_dir = os.path.join(config["output_root"], f"{config['dataset']}/{config['task']}/{config['model']}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存SHAP值
    pd.to_pickle(shap_values, os.path.join(save_dir, "shap_values.pkl"))

    # 处理特征名称和原始数据
    feature_names_path = os.path.join(dataset_path, 'labtest_features.pkl')
    test_raw_x_path = os.path.join(dataset_path, 'test_raw_x.pkl')

    feature_names = pd.read_pickle(feature_names_path)
    test_raw_x = pd.read_pickle(test_raw_x_path)

    # 处理特征
    if config["dataset"] in ['mimic-iv', 'mimic-iii']:
        features = shap_values[:, config["demo_dim"] + 47:]
        test_raw_x = np.array([x[-1] for x in test_raw_x])[:, config["demo_dim"] + 47:]
    else:
        features = shap_values[:, config["demo_dim"]:]
        test_raw_x = np.array([x[-1] for x in test_raw_x])[:, config["demo_dim"]:]

    # 提取重要特征
    all_features = []
    for feature_weight_item, raw_item in zip(features, test_raw_x):
        last_feat_dict = {key: {'value': value, 'attention': abs(attn)}
                         for key, value, attn in zip(feature_names, raw_item, feature_weight_item)}
        last_feat_dict_sort = dict(sorted(last_feat_dict.items(),
                                        key=lambda x: abs(x[1]['attention']),
                                        reverse=True))
        selected_features = [item for item in last_feat_dict_sort.items()][:3]
        all_features.append(selected_features)

    # 保存处理后的特征
    pd.to_pickle(all_features, os.path.join(save_dir, "important_features.pkl"))

    return config, shap_values, all_features


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Calculate feature importance for EHR models')

    # 基本配置
    parser.add_argument('--model', '-m', type=str, nargs='+', required=True, help='Model name')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name',
                       choices=['tjh', 'mimic-iv', 'mimic-iii', 'esrd'])
    parser.add_argument('--task', '-t', type=str, required=True, help='Task name',
                       choices=['mortality', 'readmission', 'los'])

    # 模型和训练超参数
    parser.add_argument('--hidden_dim', '-hd', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--batch_size', '-bs', type=int, default=2048, help='Batch size')
    parser.add_argument('--output_dim', '-od', type=int, default=1, help='Output dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # 其他配置
    parser.add_argument('--output_root', type=str, default='logs', help='Root directory for saving outputs')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 设置配置字典
    config = {
        'dataset': args.dataset,
        'task': args.task,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'output_dim': args.output_dim,
        'seed': args.seed,
        'output_root': args.output_root,
        'checkpoint_path': args.checkpoint_path,
    }

    # 设置输入维度
    if args.dataset == 'tjh':
        config['demo_dim'] = 2
        config['lab_dim'] = 73
    elif args.dataset in ['mimic-iv', 'mimic-iii']:
        config['demo_dim'] = 2
        config['lab_dim'] = 42
    elif args.dataset == 'esrd':
        config['demo_dim'] = 2
        config['lab_dim'] = 47
    else:
        raise ValueError("Unsupported dataset")

    # 设置随机种子
    L.seed_everything(config['seed'])

    for model in args.model:
        # Add the model name to the configuration
        config['model'] = model

        # Print the configuration
        print("Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        # Analyze the feature importance
        try:
            config, shap_values, important_features = run_importance_analysis(config)
            print(f"Feature importance analysis completed. Results saved to {config['output_root']}")
        except Exception as e:
            print(f"Error occurred while running feature importance analysis:")
            print(e)
            continue
