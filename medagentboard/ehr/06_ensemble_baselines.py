import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pyehr.utils.metrics import get_binary_metrics, check_metric_is_better
from pyehr.utils.bootstrap import run_bootstrap


def ensemble_predictions(logits, method='average', weights=None):
    if method == 'average':
        return np.mean(logits, axis=0)
    elif method == 'weighted_average':
        return np.average(logits, axis=0, weights=weights)
    else:
        raise ValueError("Invalid method specified")


def ensemble_baselines_wo_params(config):
    datasets = [config["dataset"]]
    models = config["models"]
    tasks = [config["task"]]
    pred_path = 'logs'
    perf_dict = {'method': [], 'dataset': [], 'task': [], 'auroc': [], 'auprc': [], 'minpse': []}
    for dataset, task in zip(datasets, tasks):
        labels = None
        preds = []
        weighted = []
        for model in models:
            pred = pd.read_pickle(os.path.join(pred_path, dataset, task, model, 'outputs.pkl'))['preds']
            if labels is None:
                labels = pd.read_pickle(os.path.join(pred_path, dataset, task, model, 'outputs.pkl'))['labels']
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred)
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels)
                preds.append(pred)
                metrics = run_bootstrap(pred, labels, {'task': task, 'los_info': None})
                weighted.append(metrics['auprc']['mean'])
            perf_dict['method'].append(model)
            perf_dict['dataset'].append(dataset)
            perf_dict['task'].append(task)
            perf_dict['auprc'].append(
                f"{metrics['auprc']['mean']*100:.2f}±{metrics['auprc']['std']*100:.2f}")
            perf_dict['auroc'].append(
                f"{metrics['auroc']['mean']*100:.2f}±{metrics['auroc']['std']*100:.2f}")
            perf_dict['minpse'].append(
                f"{metrics['minpse']['mean']*100:.2f}±{metrics['minpse']['std']*100:.2f}")
        preds = np.array(preds)
        weighted = np.array(weighted)
        weighted = weighted / np.sum(weighted)
        for method in ['average', 'weighted_average']:
            ensemble = ensemble_predictions(
                preds, method=method, weights=weighted)
            metrics = run_bootstrap(ensemble, labels, {'task': task, 'los_info': None})
            perf_dict['method'].append(method)
            perf_dict['dataset'].append(dataset)
            perf_dict['task'].append(task)
            perf_dict['auprc'].append(
                f"{metrics['auprc']['mean']*100:.2f}±{metrics['auprc']['std']*100:.2f}")
            perf_dict['auroc'].append(
                f"{metrics['auroc']['mean']*100:.2f}±{metrics['auroc']['std']*100:.2f}")
            perf_dict['minpse'].append(
                f"{metrics['minpse']['mean']*100:.2f}±{metrics['minpse']['std']*100:.2f}")
    return pd.DataFrame(perf_dict)


class MyDataset(Dataset):
    def __init__(self, config, mode):
        self.data = pd.read_pickle(f'logs/{config["dataset"]}/{config["task"]}/ColaCare/{mode}_embeddings.pkl')["ehr_preds"]
        self.y = pd.read_pickle(f'logs/{config["dataset"]}/{config["task"]}/ColaCare/{mode}_embeddings.pkl')["labels"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.y[idx]


class MyDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.train_dataset = MyDataset(config, mode="train")
        self.val_dataset = MyDataset(config, mode="val")
        self.test_dataset = MyDataset(config, mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class TemperatureEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperatures = nn.Parameter(torch.ones(3))
        self.weights = nn.Parameter(torch.ones(3)/3)

    def forward(self, logits1, logits2, logits3):
        scaled_logits = [
            logits1 / self.temperatures[0],
            logits2 / self.temperatures[1],
            logits3 / self.temperatures[2]
        ]
        weights = F.softmax(self.weights, dim=0)
        return F.sigmoid(sum(w * l for w, l in zip(weights, scaled_logits)))


class Pipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.main_metric = config["main_metric"]
        self.task = config["task"]
        self.output_dim = 1
        self.model = TemperatureEnsemble()
        self.loss_fn = nn.BCELoss()

        self.cur_best_performance = {}  # val set
        self.test_performance = {}  # test set

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_outputs = {}

    def forward(self, batch):
        data1, data2, data3, _ = batch
        y_hat = self.model(data1, data2, data3).to(data1.device)
        return y_hat

    def _get_loss(self, batch):
        data1, data2, data3, y = batch
        y_hat = self(batch)
        y = y.to(y_hat.dtype)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ehr, text, y, pid = batch
        loss, y_hat = self._get_loss(batch)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred']
                           for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true']
                           for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack(
            [x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)

        metrics = get_binary_metrics(y_pred, y_true)
        for k, v in metrics.items():
            self.log(k, v)

        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items():
                self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        data1, data2, data3, y = batch
        loss, y_hat = self._get_loss(batch)
        self.log("test_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'test_loss': loss}
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred']
                           for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true']
                           for x in self.test_step_outputs]).detach().cpu()
        loss = torch.stack([x['test_loss']
                           for x in self.test_step_outputs]).mean().detach().cpu()
        self.log("test_loss_epoch", loss)

        test_performance = get_binary_metrics(y_pred, y_true)
        for k, v in test_performance.items():
            self.log("test_"+k, v)

        self.test_outputs = {'y_pred': y_pred,
                             'y_true': y_true, 'test_loss': loss}
        self.test_step_outputs.clear()

        self.test_performance = test_performance
        return test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_experiment(config):
    dm = MyDataModule(config)

    # logger
    logger = CSVLogger(save_dir="logs", flush_logs_every_n_steps=1, name=f'{config["dataset"]}/{config["task"]}/ensemble')

    # main metric
    main_metric = "auroc" if config["task"] in ["mortality", "readmission", "sptb"] else "mae"
    mode = "max" if config["task"] in ["mortality", "readmission", "sptb"] else "min"
    config["main_metric"] = main_metric

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor=main_metric, patience=config["patience"], mode=mode)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor=main_metric, mode=mode)

    L.seed_everything(42)  # seed for reproducibility

    # train/val/test
    pipeline = Pipeline(config)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=config["epochs"], logger=logger, callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = Pipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


def main(config):
    performance_table = {'method': [], 'dataset': [], 'task': [], 'auroc': [], 'auprc': [], 'minpse': []}
    _, outs = run_experiment(config)
    metrics = run_bootstrap(outs['y_pred'], outs['y_true'], {'task': config['task'], 'los_info': None})
    metrics = {k: f"{v['mean']*100:.2f}±{v['std']*100:.2f}" for k, v in metrics.items()}
    performance_table['dataset'].append(config['dataset'])
    performance_table['task'].append(config['task'])
    performance_table['method'].append('temperature_ensemble')
    for k, v in metrics.items():
        performance_table[k].append(v) if k in performance_table else None

    performance_table = pd.concat([pd.DataFrame(performance_table), ensemble_baselines_wo_params(config)], ignore_index=True)
    performance_table.to_csv(f'{config["dataset"]}_{config["task"]}_ensemble_metrics.csv', index=False)


if __name__ == '__main__':
    configs = [{
        'dataset': 'esrd',
        'task': 'mortality',
        'models': ['AdaCare', 'ConCare', 'RETAIN'],
        'model': 'AdaCare',
        'batch_size': 256,
        'learning_rate': 1e-3,
        'main_metric': 'auprc',
        'epochs': 30,
        'patience': 10,
        'fold': 1,
        'seed': 0,
        "demo_dim": 0,
        "lab_dim": 17,
        "hidden_dim": 128,
        "output_dim": 1,
    }, {
        "model": "AdaCare",
        'models': ['AdaCare', 'ConCare', 'RETAIN'],
        "dataset": "mimic-iv",
        "task": "mortality",
        "main_metric": "auprc",
        "patience": 10,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 128,
        "output_dim": 1,
        'fold': 1,
        'seed': 0,
    }, {
        "model": "ConCare",
        'models': ['AdaCare', 'ConCare', 'RETAIN'],
        "dataset": "mimic-iv",
        "task": "readmission",
        "main_metric": "auprc",
        "patience": 10,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "demo_dim": 2,
        "lab_dim": 59,
        "hidden_dim": 128,
        "output_dim": 1,
        'fold': 1,
        'seed': 0,
    }, {
        "model": "RETAIN",
        'models': ['AdaCare', 'ConCare', 'RETAIN'],
        "dataset": "obstetrics",
        "task": "sptb",
        "main_metric": "auprc",
        "patience": 10,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "demo_dim": 2,
        "lab_dim": 59,
        "hidden_dim": 128,
        "output_dim": 1,
        'fold': 1,
        'seed': 0,
    },]
    for config in configs:
        main(config)
