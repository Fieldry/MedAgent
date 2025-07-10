import os
import argparse

import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pyehr.utils.metrics import get_binary_metrics, check_metric_is_better
from pyehr.utils.bootstrap import run_bootstrap

class MyDataset(Dataset):
    def __init__(self, data_path, mode="train"):
        super().__init__()
        data = pd.read_pickle(os.path.join(data_path, f"{mode}_embeddings.pkl"))
        self.text_embeddings = data["text_embeddings"]
        length = len(self.text_embeddings)
        self.ehr_scores = data["ehr_scores"][:length]
        self.ehr_embeddings = data["ehr_embeddings"][:length]
        self.y = data["labels"][:length]
        self.pids = data["pids"][:length]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        ehr_score = self.ehr_scores[index]
        ehr_embedding = self.ehr_embeddings[index]
        merged_ehr_embedding = torch.tensor([score * embedding for score, embedding in zip(ehr_score, ehr_embedding)])
        merged_ehr_embedding = merged_ehr_embedding.sum(dim=0) / sum(ehr_score)
        text_embedding = torch.tensor(self.text_embeddings[index])
        y = self.y[index]
        pid = self.pids[index]
        return merged_ehr_embedding, text_embedding, y, pid


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_path):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MyDataset(data_path, mode="train")
        self.val_dataset = MyDataset(data_path, mode='val')
        self.test_dataset = MyDataset(data_path, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class FusionModel(nn.Module):
    def __init__(self, ehr_embed_dim, text_embed_dim, merge_embed_dim, output_dim):
        super().__init__()
        self.ehr_embed_dim = ehr_embed_dim
        self.text_embed_dim = text_embed_dim
        self.merge_embed_dim = merge_embed_dim
        self.output_dim = output_dim

        self.ehr_embed = nn.Linear(ehr_embed_dim, merge_embed_dim)
        self.text_embed = nn.Linear(text_embed_dim, merge_embed_dim)
        self.merge_embed = nn.Sequential(
            nn.Linear(merge_embed_dim * 2, merge_embed_dim),
            nn.GELU(),
            nn.Linear(merge_embed_dim, merge_embed_dim),
            nn.GELU(),
        )

    def forward(self, ehr, text):
        ehr_embed = self.ehr_embed(ehr.to(torch.float32))
        text_embed = self.text_embed(text.to(torch.float32))
        merge_embed = self.merge_embed(torch.cat([ehr_embed, text_embed], dim=-1))
        return merge_embed


class Pipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.main_metric = config["main_metric"]
        self.output_dim = config["output_dim"]
        self.task = config["task"]
        self.model = FusionModel(ehr_embed_dim=config["ehr_embed_dim"], text_embed_dim=config["text_embed_dim"], merge_embed_dim=config["merge_embed_dim"], output_dim=self.output_dim)
        self.head = nn.Sequential(
            nn.Linear(config["merge_embed_dim"], self.output_dim),
            nn.Dropout(0.0),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

        self.cur_best_performance = {}  # val set
        self.test_performance = {}  # test set
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_outputs = {}

    def forward(self, batch):
        merged_ehr_embedding, text_embedding, _, _ = batch
        y_hat = self.model(merged_ehr_embedding, text_embedding).to(merged_ehr_embedding.device)
        y_hat = self.head(y_hat).squeeze(-1)
        return y_hat

    def _get_loss(self, batch):
        y = batch[2]
        y_hat = self(batch)
        y = y.to(y_hat.dtype)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[2]
        loss, y_hat = self._get_loss(batch)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)

        metrics = get_binary_metrics(y_pred, y_true)
        for k, v in metrics.items():
            self.log(k, v)

        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        y = batch[2]
        loss, y_hat = self._get_loss(batch)
        self.log("test_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'test_loss': loss}
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).detach().cpu()
        loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().detach().cpu()
        self.log("test_loss_epoch", loss)

        test_performance = get_binary_metrics(y_pred, y_true)
        for k, v in test_performance.items(): self.log("test_"+k, v)

        self.test_outputs = {'y_pred': y_pred, 'y_true': y_true, 'test_loss': loss}
        self.test_step_outputs.clear()

        self.test_performance = test_performance
        return test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_experiment(config):
    # data
    data_path = os.path.join("logs", config["dataset"], config["task"], config["method"])
    dm = MyDataModule(batch_size=config["batch_size"], data_path=data_path)

    # logger
    logger = CSVLogger(save_dir="logs", name=f"{config['dataset']}/{config['task']}/{config['method']}/fusion/{'-'.join(config['ehr_model_names'])}_{config['lm_name']}", flush_logs_every_n_steps=1)

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor="auroc", patience=config["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="auroc", mode="max")

    L.seed_everything(42)  # seed for reproducibility

    # train/val/test
    pipeline = Pipeline(config)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = Pipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic configurations
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd"])
    parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los"])
    parser.add_argument("--method", "-m", type=str, default="ColaCare", help="Method name", choices=["ColaCare"])
    parser.add_argument("--ehr_model_names", "-em", type=str, nargs="+", required=True, help="EHR model names")
    parser.add_argument("--lm_name", "-lm", type=str, required=True, help="Language model name")

    # Model and training hyperparameters
    parser.add_argument('--ehr_embed_dim', '-ehd', type=int, default=128, help='EHR embedding dimension')
    parser.add_argument('--text_embed_dim', '-thd', type=int, default=1024, help='Text embedding dimension')
    parser.add_argument('--merge_embed_dim', '-mhd', type=int, default=1024, help='Merged embedding dimension')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--output_dim', '-od', type=int, default=1, help='Output dimension')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Seed')
    parser.add_argument('--main_metric', '-mm', type=str, default='auroc', help='Main metric', choices=['auroc', 'auprc'])

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up the configuration dictionary
    config = {
        'dataset': args.dataset,
        'task': args.task,
        'method': args.method,
        'ehr_model_names': args.ehr_model_names,
        'lm_name': args.lm_name,
        'ehr_embed_dim': args.ehr_embed_dim,
        'text_embed_dim': args.text_embed_dim,
        'merge_embed_dim': args.merge_embed_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'output_dim': args.output_dim,
        'seed': args.seed,
        'main_metric': args.main_metric,
    }

    perf, outs = run_experiment(config)
    print(perf)
    print(outs)