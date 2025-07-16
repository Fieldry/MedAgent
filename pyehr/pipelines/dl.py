import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyehr import models
from pyehr.datasets.utils.utils import unpad_y
from pyehr.utils.loss import get_loss
from pyehr.utils.metrics import get_all_metrics, check_metric_is_better
from pyehr.utils.calibration import SigmoidTemperatureScaling
from pyehr.models.utils import generate_mask


class DlPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.demo_dim = config["demo_dim"]
        self.lab_dim = config["lab_dim"]
        self.input_dim = self.demo_dim + self.lab_dim
        config["input_dim"] = self.demo_dim + self.lab_dim
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.task = config.get("task", "outcome") # 从config获取task，默认为outcome
        self.los_info = config.get("los_info", {}) # 兼容LOS任务
        self.model_name = config["model"]
        self.main_metric = config["main_metric"]
        self.time_aware = config.get("time_aware", False)
        config['apply_mha_decov_loss'] = True
        self.cur_best_performance = {}
        self.calibrated = False
        self.optimal_decision_threshold = 0.5

        if self.model_name == "StageNet":
            config["chunk_size"] = self.hidden_dim

        model_class = getattr(models, self.model_name)
        self.ehr_encoder = model_class(**config)

        # 根据任务类型选择不同的模型头部 (Head)
        if self.task in ["outcome", "mortality", "readmission", "sptb"]:
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif self.task == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif self.task == "multitask":
            self.head = models.heads.MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)
        else:
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_performance = {}
        self.test_outputs = {}

    def forward(self, x, lens):
        x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
        mask = generate_mask(lens)

        if self.model_name == "ConCare":
            embedding, attn, decov_loss = self.ehr_encoder(x_lab, x_demo, mask)
            embedding, attn, decov_loss = embedding.to(x.device), attn.to(x.device), decov_loss.to(x.device)
            y_hat = self.head(embedding)
            return y_hat, embedding, attn, decov_loss
        elif self.model_name in ["AdaCare", "RETAIN"]:
            embedding, attn = self.ehr_encoder(x, mask)
            embedding, attn = embedding.to(x.device), attn.to(x.device)
            y_hat = self.head(embedding)
            return y_hat, embedding, attn

    def _get_loss(self, x, y, lens):
        if self.model_name == "ConCare":
            y_hat, embedding, attn, decov_loss = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
            loss += 10 * decov_loss
        elif self.model_name in ["AdaCare", "RETAIN"]:
            y_hat, embedding, attn = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
        else:
            y_hat, embedding = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task, self.time_aware)
            attn = None
        return loss, y, y_hat, embedding, attn

    def training_step(self, batch, batch_idx):
        x, y, lens, _ = batch
        loss, _, _, _, _ = self._get_loss(x, y, lens)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lens, _ = batch
        loss, y, y_hat, _, _ = self._get_loss(x, y, lens)
        self.log("val_loss", loss)
        outs = {'preds': y_hat, 'labels': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['preds'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['labels'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)
        metrics = get_all_metrics(y_pred, y_true, self.task, self.los_info)
        for k, v in metrics.items():
            self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items():
                self.log("best_" + k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat, embedding, attn = self._get_loss(x, y, lens)
        outs = {'preds': y_hat, 'labels': y, 'pids': pid, 'embedding': embedding}
        if attn is not None:
            outs['attn'] = attn
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in self.test_step_outputs]).detach().cpu()

        pids = []
        for x in self.test_step_outputs:
            if isinstance(x['pids'], (list, tuple)):
                pids.extend(x['pids'])
            else:
                pids.append(x['pids'])

        embeddings = torch.cat([x['embedding'] for x in self.test_step_outputs]).detach().cpu()

        attns = None
        if 'attn' in self.test_step_outputs[0] and self.test_step_outputs[0]['attn'] is not None:
             attns = torch.cat([x['attn'] for x in self.test_step_outputs]).detach().cpu()
             if attns.size(1) > self.lab_dim:
                attns = attns[:, self.demo_dim:]

        self.test_performance = get_all_metrics(preds, labels, self.task, self.los_info)

        self.test_outputs = {
            'preds': preds,
            'labels': labels,
            'pids': pids,
            'embeddings': embeddings
        }
        if attns is not None:
            self.test_outputs['attns'] = attns

        self.test_step_outputs.clear()
        return self.test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def fit_calibration_temperature(self, calibration_dataloader: DataLoader):
        print("Fitting calibration temperature...")
        self.calibration_scaler = SigmoidTemperatureScaling()

        self.ehr_encoder.eval()
        if hasattr(self, 'head'):
            self.head.eval()

        all_logits_list = []
        all_labels_list = []
        device = self.device

        for batch in calibration_dataloader:
            x, y, lens, _ = batch
            x, y, lens = x.to(device), y.to(device), lens.to(device)

            logits = self._get_logits_for_calibration(x, lens)
            logits_unpadded, labels_unpadded = unpad_y(logits, y, lens)

            all_logits_list.append(logits_unpadded)
            all_labels_list.append(labels_unpadded)

        if not all_logits_list:
            print("Warning: No valid logits in calibration dataset. Calibration failed.")
            self.calibrated = False
            return self

        final_logits = torch.cat(all_logits_list, dim=0)
        final_labels = torch.cat(all_labels_list, dim=0)

        self.calibration_scaler = self.calibration_scaler.to(device)
        self.calibration_scaler.fit(final_logits, final_labels.float())

        self.calibrated = True
        print(f"Calibration complete. Learned temperature: {self.calibration_scaler.temperature.item():.4f}")
        return self

    def _get_logits_for_calibration(self, x, lens):
        x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
        mask = generate_mask(lens)

        if self.model_name == "ConCare":
            embedding, _, _ = self.ehr_encoder(x_lab, x_demo, mask)
        elif self.model_name in ["AdaCare", "RETAIN"]:
            embedding, _ = self.ehr_encoder(x, mask)
        else:
            raise ValueError(f"Model {self.model_name} not supported for direct logit extraction here.")

        embedding = embedding.to(self.device)

        if isinstance(self.head, nn.Sequential) and isinstance(self.head[-1], (nn.Sigmoid, nn.Softmax)):
            logits = embedding
            for layer_idx in range(len(self.head) - 1):
                logits = self.head[layer_idx](logits)
            return logits
        elif isinstance(self.head, nn.Sequential):
             return self.head(embedding)
        else:
            raise NotImplementedError("Logit extraction for this head structure needs specific handling.")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['model_calibrated_flag'] = self.calibrated
        checkpoint['optimal_decision_threshold'] = self.optimal_decision_threshold

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        loaded_calibrated_flag = checkpoint.get('model_calibrated_flag', False)
        if hasattr(self, 'calibration_scaler'):
            self.calibrated = loaded_calibrated_flag
            if self.calibrated:
                print(f"on_load_checkpoint: Classification model restored as CALIBRATED. Temperature: {self.calibration_scaler.temperature.item():.4f}")
            else:
                print(f"on_load_checkpoint: Classification model restored as UNCALIBRATED. Current temperature: {self.calibration_scaler.temperature.item():.4f}")
        else:
            self.calibrated = False
            print("on_load_checkpoint: calibration_scaler not found. Classification calibration state not restored.")

        self.optimal_decision_threshold = checkpoint.get('optimal_decision_threshold', 0.5)
        print(f"on_load_checkpoint: Loaded optimal decision threshold: {self.optimal_decision_threshold:.4f}")

    @torch.no_grad()
    def get_calibrated_probs_and_labels_for_threshold_tuning(self, dataloader: DataLoader):
        if not self.calibrated:
            print("Warning: Model is not calibrated. Probabilities will be based on original logits.")

        self.ehr_encoder.eval()
        if hasattr(self, 'head'):
            self.head.eval()

        all_probs_list = []
        all_labels_list = []
        device = self.device

        for batch in dataloader:
            x, y, lens, _ = batch
            x, y, lens = x.to(device), y.to(device), lens.to(device)

            logits_batch = self._get_logits_for_calibration(x, lens)

            if self.calibrated:
                scaled_logits_batch = self.calibration_scaler.to(logits_batch.device)(logits_batch)
            else:
                scaled_logits_batch = logits_batch

            probs_batch = torch.sigmoid(scaled_logits_batch)

            y_labels = y[..., 0].unsqueeze(-1)

            unpadded_probs, unpadded_labels = unpad_y(probs_batch, y_labels, lens)

            all_probs_list.append(unpadded_probs)
            all_labels_list.append(unpadded_labels)

        if not all_probs_list:
            print("Warning: Could not extract any valid probabilities or labels from the dataloader for threshold tuning.")
            return torch.empty(0), torch.empty(0)

        final_probs = torch.cat(all_probs_list, dim=0)
        final_labels = torch.cat(all_labels_list, dim=0)

        return final_probs.squeeze(), final_labels.squeeze().long()