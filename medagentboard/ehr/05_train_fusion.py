import os
import argparse

import numpy as np
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
        self.ehr_preds = data["ehr_preds"][:length]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        ehr_embeddings = torch.tensor(self.ehr_embeddings[index], dtype=torch.float32)
        text_embedding = torch.tensor(self.text_embeddings[index], dtype=torch.float32)
        y = self.y[index]
        pid = self.pids[index]
        ehr_score = torch.tensor(self.ehr_scores[index], dtype=torch.float32)
        return ehr_embeddings, text_embedding, y, pid, ehr_score


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_path):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MyDataset(data_path, mode="train")
        self.val_dataset = MyDataset(data_path, mode="val")
        self.test_dataset = MyDataset(data_path, mode="test")

    def train_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, multiprocessing_context=multiprocessing_context)

    def val_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, multiprocessing_context=multiprocessing_context)

    def test_dataloader(self):
        multiprocessing_context='fork' if torch.backends.mps.is_available() else None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, multiprocessing_context=multiprocessing_context)


class AttentionFusionLayer(nn.Module):
    """
    使用注意力机制动态融合多个EHR模型的隐层表示.
    Query: text_embedding (LLM报告的嵌入)
    Keys/Values: ehr_embeddings_list (多个小模型的嵌入)
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, text_embedding, ehr_embeddings_list):
        """
        Args:
            text_embedding (torch.Tensor): LLM报告嵌入, shape (batch_size, embed_dim)
            ehr_embeddings_list (torch.Tensor): 多个EHR模型嵌入, shape (batch_size, num_models, embed_dim)
        Returns:
            torch.Tensor: 融合后的EHR嵌入, shape (batch_size, embed_dim)
        """
        # MultiheadAttention期望的输入是 (batch, seq_len, embed_dim)
        # Query需要扩展一个维度: (batch, 1, embed_dim)
        query = text_embedding.unsqueeze(1)
        # Keys和Values就是我们的EHR嵌入列表
        keys = ehr_embeddings_list
        values = ehr_embeddings_list

        # attn_output shape: (batch_size, 1, embed_dim)
        # attn_weights shape: (batch_size, 1, num_models) -> 我们可以用这个来做可视化分析！
        attn_output, attn_weights = self.attention(query, keys, values)

        # fused_ehr_embedding shape: (batch_size, embed_dim)
        fused_ehr_embedding = attn_output.squeeze(1)

        return fused_ehr_embedding


class AttentionFusionModel(nn.Module):
    def __init__(self, ehr_embed_dim, text_embed_dim, merge_embed_dim, output_dim, num_models=3):
        super().__init__()
        self.embed_dim = merge_embed_dim

        self.ehr_proj = nn.Linear(ehr_embed_dim, merge_embed_dim)

        self.attention_fusion = AttentionFusionLayer(embed_dim=self.embed_dim)

        self.merge_embed = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )

    def forward(self, ehr_list, text):
        # ehr_list shape: (batch, num_models, ehr_embed_dim)
        # text shape: (batch, embed_dim)

        ehr_list = self.ehr_proj(ehr_list)
        # ehr_list shape: (batch, num_models, merge_embed_dim)

        # 1. 动态融合EHR嵌入
        fused_ehr_embedding = self.attention_fusion(text, ehr_list)

        # 2. 拼接融合后的EHR嵌入和文本嵌入
        merge_input = torch.cat([fused_ehr_embedding, text], dim=-1)

        # 3. 通过MLP进行最终的特征提取
        merge_embed = self.merge_embed(merge_input)
        return merge_embed


class ConcatFusionModel(nn.Module):
    def __init__(self, ehr_embed_dim, text_embed_dim, merge_embed_dim, output_dim, score=False, num_models=3):
        super().__init__()
        self.embed_dim = merge_embed_dim
        self.score = score

        self.ehr_proj = nn.Linear(ehr_embed_dim, merge_embed_dim)

        self.merge_embed = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
        )

    def forward(self, ehr_list, text, ehr_scores=None):
        # ehr_list shape: (batch, num_models, ehr_embed_dim)
        # ehr_scores shape: (batch, num_models)
        if self.score and ehr_scores is not None:
            ehr_list = ehr_list * ehr_scores.unsqueeze(-1)
            ehr_list = self.ehr_proj(ehr_list).mean(dim=1) # (batch, merge_embed_dim)
        else:
            ehr_list = self.ehr_proj(ehr_list).mean(dim=1) # (batch, merge_embed_dim)

        merge_input = torch.cat([ehr_list, text], dim=-1)
        merge_embed = self.merge_embed(merge_input)
        return merge_embed


class Pipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # 保存配置，方便加载
        self.learning_rate = self.hparams.learning_rate
        self.main_metric = self.hparams.main_metric
        self.output_dim = self.hparams.output_dim
        self.fusion_method = self.hparams.fusion_method

        if self.fusion_method == "attention":
            self.model = AttentionFusionModel(
                ehr_embed_dim=self.hparams.ehr_embed_dim,
                text_embed_dim=self.hparams.text_embed_dim,
                merge_embed_dim=self.hparams.merge_embed_dim,
                output_dim=self.output_dim
            )
        elif self.fusion_method == "concat":
            self.model = ConcatFusionModel(
                ehr_embed_dim=self.hparams.ehr_embed_dim,
                text_embed_dim=self.hparams.text_embed_dim,
                merge_embed_dim=self.hparams.merge_embed_dim,
                output_dim=self.output_dim
            )
        elif self.fusion_method == "score":
            self.model = ConcatFusionModel(
                ehr_embed_dim=self.hparams.ehr_embed_dim,
                text_embed_dim=self.hparams.text_embed_dim,
                merge_embed_dim=self.hparams.merge_embed_dim,
                output_dim=self.output_dim,
                score=True
            )

        self.head = nn.Sequential(
            nn.Linear(self.hparams.merge_embed_dim, self.output_dim),
            nn.Dropout(0.0),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

        self.cur_best_performance = {}
        self.test_performance = {}
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_outputs = {}

    def forward(self, batch):
        ehr_embeddings_list, text_embedding, _, _, ehr_score = batch
        if self.fusion_method == "score":
            y_hat = self.model(ehr_embeddings_list, text_embedding, ehr_score)
        else:
            y_hat = self.model(ehr_embeddings_list, text_embedding)
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
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.hparams.task):
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
    data_path = os.path.join("logs", config["dataset"], config["task"], config["method"], f'{config["modality"]}_{config["lm_name"]}')
    dm = MyDataModule(batch_size=config["batch_size"], data_path=data_path)

    # logger
    log_name = f"{config['dataset']}/{config['task']}/{config['method']}/{config['modality']}_{config['lm_name']}/fusion/{'-'.join(config['ehr_model_names'])}_{config['lm_name']}"
    logger = CSVLogger(save_dir="logs", name=log_name, flush_logs_every_n_steps=1)

    # Callbacks
    early_stopping_callback = EarlyStopping(monitor="auroc", patience=config["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="auroc", mode="max")

    L.seed_everything(config['seed'])

    # train/val/test
    pipeline = Pipeline(config)
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model and test
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = Pipeline.load_from_checkpoint(best_model_path)
    trainer.test(pipeline, datamodule=dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return dm, perf, outs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["cdsl", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los", "sptb"])
    parser.add_argument("--method", "-m", type=str, default="ColaCare", help="Method name", choices=["ColaCare", "MedAgent", "ReConcile"])
    parser.add_argument("--ehr_model_names", "-em", type=str, nargs="+", required=True, help="EHR model names")
    parser.add_argument("--lm_name", "-lm", type=str, required=True, help="Large Language model name")
    parser.add_argument("--modality", "-md", type=str, help="Modality", default="ehr", choices=["ehr", "mm"])
    parser.add_argument("--use_rag", "-ur", action="store_true", help="Use RAG to generate reports.")

    parser.add_argument('--ehr_embed_dim', '-ehd', type=int, default=128, help='EHR embedding dimension')
    parser.add_argument('--text_embed_dim', '-thd', type=int, default=1024, help='Text embedding dimension')
    parser.add_argument('--merge_embed_dim', '-mhd', type=int, default=1024, help='The unified embedding dimension for all inputs and fusion model')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', '-p', type=int, default=30, help='Patience for early stopping')
    parser.add_argument('--output_dim', '-od', type=int, default=1, help='Output dimension')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Seed')
    parser.add_argument('--main_metric', '-mm', type=str, default='auroc', help='Main metric', choices=['auroc', 'auprc'])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    fusion_perf_df = pd.DataFrame()
    for fusion_method in ["score", "concat", "attention"]:
        config["fusion_method"] = fusion_method
        dm, perf, outs = run_experiment(config)

        perf_boot = run_bootstrap(outs["y_pred"], outs["y_true"], {"task": args.task, "los_info": None})
        for key, value in perf_boot.items():
            if args.task in ["mortality", "readmission", "sptb"]:
                perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'
            else:
                perf_boot[key] = f'{value["mean"]:.2f}±{value["std"]:.2f}'

        perf_boot = dict({
            "model": args.method + "_" + fusion_method,
            "dataset": args.dataset,
            "task": args.task,
            "llm": args.lm_name,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        fusion_perf_df = pd.concat([fusion_perf_df, perf_df], ignore_index=True)

    ehr_preds = dm.test_dataset.ehr_preds
    ehr_preds = np.array(ehr_preds).transpose(1, 0)

    assert ehr_preds.shape[0] == len(args.ehr_model_names)
    assert ehr_preds.shape[1] == len(dm.test_dataset.pids)

    ehr_perf_df = pd.DataFrame()
    for ehr_pred, ehr_model_name in zip(ehr_preds, args.ehr_model_names):
        perf_boot = run_bootstrap(ehr_pred, outs["y_true"], {"task": args.task, "los_info": None})
        for key, value in perf_boot.items():
            if args.task in ["mortality", "readmission", "sptb"]:
                perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'
            else:
                perf_boot[key] = f'{value["mean"]:.2f}±{value["std"]:.2f}'
        perf_boot = dict({
            "model": ehr_model_name,
            "dataset": args.dataset,
            "task": args.task,
            "llm": args.lm_name,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        ehr_perf_df = pd.concat([ehr_perf_df, perf_df], ignore_index=True)

    log_dir = os.path.join("logs", args.dataset, args.task, args.method, f'{args.modality}_{args.lm_name}', "fusion")
    os.makedirs(log_dir, exist_ok=True)
    all_perf_df = pd.concat([ehr_perf_df, fusion_perf_df], ignore_index=True)
    all_perf_df.to_csv(os.path.join(log_dir, f"05_fusion_performance_{'w' if args.use_rag else 'worag'}.csv"), index=False)
    print(f"Performance saved to {log_dir}/05_fusion_performance_{'w' if args.use_rag else 'worag'}.csv")