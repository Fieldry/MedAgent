import lightning as L
import torch
import torch.nn as nn

from pyehr import models
from pyehr.datasets.utils.utils import unpad_y
from pyehr.utils.loss import get_loss
from pyehr.utils.metrics import get_all_metrics, check_metric_is_better
from pyehr.models.utils import generate_mask


class DlPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.demo_dim = config["demo_dim"]
        self.lab_dim = config["lab_dim"]
        self.input_dim = self.demo_dim + self.lab_dim
        config["input_dim"] = self.input_dim
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.task = config["task"]
        self.los_info = config["los_info"]
        self.model_name = config["model"]
        self.main_metric = config["main_metric"]
        self.time_aware = config.get("time_aware", False)
        self.cur_best_performance = {}

        model_class = getattr(models, self.model_name)
        self.ehr_encoder = model_class(**config)
        if self.task in ["outcome", "mortality", "readmission", "sptb"]:
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif self.task == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif self.task == "multitask":
            self.head = models.heads.MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_performance = {}
        self.test_outputs = {}

    def forward(self, batch):
        x, y, lens, _ = batch
        if self.model_name == "ConCare":
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding, attn, decov_loss = self.ehr_encoder(x_lab, x_demo, mask)
            embedding, attn, decov_loss = embedding.to(x.device), attn.to(x.device), decov_loss.to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding, attn, decov_loss
        elif self.model_name in ["AdaCare", "RETAIN"]:
            mask = generate_mask(lens)
            embedding, attn = self.ehr_encoder(x, mask)
            embedding, attn = embedding.to(x.device), attn.to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding, attn
        elif self.model_name in ["GRASP", "Agent", "AICare"]:
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding = self.ehr_encoder(x_lab, x_demo, mask).to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding
        elif self.model_name in ["AnchCare", "TCN", "Transformer", "StageNet"]:
            mask = generate_mask(lens)
            embedding = self.ehr_encoder(x, mask).to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding
        elif self.model_name in ["GRU", "LSTM", "RNN", "MLP"]:
            embedding = self.ehr_encoder(x).to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding
        elif self.model_name in ["MCGRU"]:
            x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
            embedding = self.ehr_encoder(x_lab, x_demo).to(x.device)
            y_hat = self.head(embedding)
            return y_hat, y, embedding

    def _get_loss(self, batch):
        lens = batch[2]
        if self.model_name == "ConCare":
            y_hat, y, embedding, attn, decov_loss = self(batch)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
            loss += 10*decov_loss
        elif self.model_name in ["AdaCare", "RETAIN"]:
            y_hat, y, embedding, attn = self(batch)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
        else:
            y_hat, y, embedding = self(batch)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
            attn = None
        return loss, y, y_hat, embedding, attn

    def training_step(self, batch, _):
        loss = self._get_loss(batch)[0]
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        loss, y, y_hat, _, _ = self._get_loss(batch)
        self.log("val_loss", loss)
        outs = {'preds': y_hat, 'labels': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss
    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)
        metrics = get_all_metrics(preds, labels, self.task, self.los_info)
        for k, v in metrics.items():
            self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log(f"best_{k}", v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, _):
        pid = batch[-1]
        loss, y, y_hat, embedding, attn = self._get_loss(batch)
        outs = {'preds': y_hat, 'labels': y, 'pids': pid, 'embedding': embedding, 'attn': attn}
        self.test_step_outputs.append(outs)
        return loss
    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in self.test_step_outputs]).detach().cpu()
        pids = []
        for x in self.test_step_outputs:
            pids.extend(x['pids'])
        embeddings = torch.cat([x['embedding'] for x in self.test_step_outputs]).detach().cpu()
        attns = torch.cat([x['attn'] for x in self.test_step_outputs]).detach().cpu()
        if attns.size(1) > self.lab_dim:
            attns = attns[:, self.demo_dim:]
        self.test_performance = get_all_metrics(preds, labels, self.task, self.los_info)
        self.test_outputs = {'preds': preds.tolist(), 'labels': labels.tolist(), 'pids': pids, 'embeddings': embeddings.tolist(), 'attns': attns.tolist()}
        self.test_step_outputs.clear()
        return self.test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer