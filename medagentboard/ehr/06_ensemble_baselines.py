import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# 假設 pyehr 工具存在於您的環境中
from pyehr.utils.metrics import get_binary_metrics, check_metric_is_better
from pyehr.utils.bootstrap import run_bootstrap

# --- 1. 資料集和資料模組定義 ---

class MyDataset(Dataset):
    """從預存的 embeddings 中載入資料的自訂資料集"""
    def __init__(self, config, mode):
        # 根據模式（train/val/test）載入對應的資料
        file_path = f'logs/{config["dataset"]}/{config["task"]}/ColaCare/ehr_deepseek-v3-official/{mode}_embeddings.pkl'
        data = pd.read_pickle(file_path)
        self.data = data["ehr_preds"]
        self.y = data["labels"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 返回三個模型的 logits 和對應的標籤
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.y[idx]


class MyDataModule(L.LightningDataModule):
    """用於訓練、驗證和測試的 LightningDataModule"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]

    def setup(self, stage=None):
        # 在此處設定資料集
        self.train_dataset = MyDataset(self.config, mode="train")
        self.val_dataset = MyDataset(self.config, mode="val")
        self.test_dataset = MyDataset(self.config, mode="test")

    def train_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, multiprocessing_context=multiprocessing_context)

    def val_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, multiprocessing_context=multiprocessing_context)

    def test_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, multiprocessing_context=multiprocessing_context)


# --- 2. 模型定義 ---

class TemperatureEnsemble(nn.Module):
    """帶有溫度縮放的加權集成模型"""
    def __init__(self, n_models=3):
        super().__init__()
        # 每個模型一個溫度參數
        self.temperatures = nn.Parameter(torch.ones(n_models))
        # 模型的權重
        self.weights = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, *logits_list):
        # 溫度縮放
        scaled_logits = [logits / temp for logits, temp in zip(logits_list, self.temperatures)]
        # 計算 softmax 權重
        weights = F.softmax(self.weights, dim=0)
        # 加權求和
        ensembled_logits = sum(w * l for w, l in zip(weights, scaled_logits))
        return torch.sigmoid(ensembled_logits)


# --- 3. Lightning 訓練流程 ---

class Pipeline(L.LightningModule):
    """用於訓練和評估集成模型的 LightningModule"""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = TemperatureEnsemble(n_models=len(config["models"]))
        self.loss_fn = nn.BCELoss()
        self.main_metric = config["main_metric"]
        self.task_type = config["task"]

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        logits1, logits2, logits3, _ = batch
        return self.model(logits1, logits2, logits3)

    def _shared_step(self, batch):
        *logits, y = batch
        y_hat = self.model(*logits)
        y = y.to(y_hat.dtype)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_outputs.append({'y_pred': y_hat, 'y_true': y})

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).cpu()
        metrics = get_binary_metrics(y_pred, y_true)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch)
        self.log("test_loss", loss)
        self.test_step_outputs.append({'y_pred': y_hat, 'y_true': y})

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).cpu()
        self.test_outputs = {'preds': y_pred, 'labels': y_true}
        metrics = get_binary_metrics(y_pred, y_true)
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.config["learning_rate"])


# --- 4. 實驗和評估函數 ---

def run_temperature_ensemble_experiment(config):
    """訓練和測試 TemperatureEnsemble 模型"""
    L.seed_everything(config["seed"])

    dm = MyDataModule(config)
    pipeline = Pipeline(config)

    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}/temperature_ensemble')

    # 根據任務確定監控的指標和模式
    mode = "max"

    early_stopping_callback = EarlyStopping(monitor="val_" + config["main_metric"], patience=config["patience"], mode=mode)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="val_" + config["main_metric"], mode=mode)

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(pipeline, datamodule=dm)

    # 載入最佳模型進行測試
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    best_pipeline = Pipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(best_pipeline, datamodule=dm)

    return best_pipeline.test_outputs


def evaluate_baseline_ensembles(config):
    """評估 average 和 weighted_average 集成方法"""
    pred_path = 'logs'
    labels = None
    model_preds = []
    model_auprcs = []

    # 收集單一模型的預測和性能
    individual_model_perfs = []

    model_output_path = f'logs/{config["dataset"]}/{config["task"]}/ColaCare/ehr_deepseek-v3-official/test_embeddings.pkl'
    if not os.path.exists(model_output_path):
        raise FileNotFoundError(f"Output file not found for model {model_name} at {model_output_path}")
    outputs = pd.read_pickle(model_output_path)
    preds = np.array(outputs['ehr_preds'])
    labels = np.array(outputs['labels'])

    for i, model_name in enumerate(config["models"]):
        model_pred = preds[:, i]
        model_preds.append(model_pred)
        metrics = run_bootstrap(model_pred, labels, {'task': config["task"], 'los_info': None})
        model_auprcs.append(metrics['auprc']['mean'])

        individual_model_perfs.append({
            'method': model_name,
            **calculate_formatted_metrics(model_pred, labels, config)
        })

    # 計算 average 和 weighted_average 集成
    ensembled_perfs = []
    model_preds = np.array(model_preds)

    # Average
    avg_preds = np.mean(model_preds, axis=0)
    ensembled_perfs.append({
        'method': 'average',
        **calculate_formatted_metrics(avg_preds, labels, config)
    })

    # Weighted Average
    weights = np.array(model_auprcs) / np.sum(model_auprcs)
    weighted_avg_preds = np.average(model_preds, axis=0, weights=weights)
    ensembled_perfs.append({
        'method': 'weighted_average',
        **calculate_formatted_metrics(weighted_avg_preds, labels, config)
    })

    return individual_model_perfs + ensembled_perfs


def calculate_formatted_metrics(preds, labels, config):
    """使用 bootstrap 計算並格式化性能指標"""
    metrics = run_bootstrap(preds, labels, {'task': config['task'], 'los_info': None})
    return {
        'dataset': config['dataset'],
        'task': config['task'],
        'auroc': f"{metrics['auroc']['mean']*100:.2f}±{metrics['auroc']['std']*100:.2f}",
        'auprc': f"{metrics['auprc']['mean']*100:.2f}±{metrics['auprc']['std']*100:.2f}",
        'minpse': f"{metrics['minpse']['mean']*100:.2f}±{metrics['minpse']['std']*100:.2f}"
    }

# --- 5. 主執行函數 ---

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="Train and evaluate ensemble models for EHR data.")

    # 基本配置
    parser.add_argument("--models", "-m", type=str, nargs="+", default=["AdaCare", "ConCare", "RETAIN"], help="List of models to ensemble.")
    parser.add_argument("--dataset", "-d", type=str, default="esrd", help="Dataset name.")
    parser.add_argument("--task", "-t", type=str, default="mortality", help="Task name.")

    # 訓練超參數
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main():
    """主函數：運行實驗並產生性能報告"""
    config = vars(parse_args())
    config["main_metric"] = "auroc"

    # 1. 運行 Temperature Scaling Ensemble 實驗
    print("--- Running Temperature Ensemble Experiment ---")
    ts_outputs = run_temperature_ensemble_experiment(config)
    ts_perf = {
        'method': 'temperature_ensemble',
        **calculate_formatted_metrics(ts_outputs['preds'], ts_outputs['labels'], config)
    }

    # 2. 評估基線 Ensemble 方法（Average & Weighted Average）
    print("\n--- Evaluating Baseline Ensemble Methods ---")
    baseline_perfs = evaluate_baseline_ensembles(config)

    # 3. 合併所有結果並儲存為 CSV
    all_performances = baseline_perfs + [ts_perf]
    performance_df = pd.DataFrame(all_performances)

    # 重新排列欄位順序
    performance_df = performance_df[['method', 'dataset', 'task', 'auroc', 'auprc', 'minpse']]

    # 儲存到指定的 CSV 檔案中
    output_filename = f'logs/{config["dataset"]}/{config["task"]}/06_ensemble_performance.csv'
    performance_df.to_csv(output_filename, index=False)

    print(f"\n✅ Performance evaluation saved to {output_filename}")
    print(performance_df)


if __name__ == '__main__':
    main()