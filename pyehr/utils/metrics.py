import numpy as np
from sklearn import metrics as sklearn_metrics

import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


def get_all_metrics(preds, labels, task, los_info):
    # convert preds and labels to tensor if they are ndarray type
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    if task in ["outcome", "mortality", "readmission", "sptb"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 2] if task == "readmission" else labels[:, 0]
        return get_binary_metrics(preds, labels)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        return get_regression_metrics(reverse_los(preds, los_info), reverse_los(labels, los_info))
    elif task == "multitask":
        return get_binary_metrics(preds[:, 0], labels[:, 0]) | get_regression_metrics(reverse_los(preds[:, 1], los_info), reverse_los(labels[:, 1], los_info))
    else:
        raise ValueError("Task not supported")


def reverse_los(y, los_info):
    return y * los_info["los_std"] + los_info["los_mean"]


def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels):
    # get binary metrics: auroc, auprc, minpse, f1, accuracy
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    accuracy = Accuracy(task="binary", threshold=0.5)
    f1 = BinaryF1Score()

    # convert labels type to int
    labels = labels.type(torch.int)

    # compute the metrics
    auroc(preds, labels)
    auprc(preds, labels)
    minpse_score = minpse(preds, labels)
    accuracy(preds, labels)
    f1(preds, labels)

    return {
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse_score,
        "accuracy": accuracy.compute().item(),
        "f1": f1.compute().item(),
    }


def get_regression_metrics(preds, labels):
    mae = MeanAbsoluteError()
    mse = MeanSquaredError(squared=True)
    rmse = MeanSquaredError(squared=False)
    r2 = R2Score()

    # compute the metrics
    mae(preds, labels)
    mse(preds, labels)
    rmse(preds, labels)
    r2(preds, labels)

    return {
        "mae": mae.compute().item(),
        "mse": mse.compute().item(),
        "rmse": rmse.compute().item(),
        "r2": r2.compute().item(),
    }


def check_metric_is_better(cur_best, main_metric, score, task):
    if task == "los":
        if cur_best == {}:
            return True
        if score < cur_best[main_metric]:
            return True
        return False
    elif task in ["mortality", "readmission", "sptb", "multitask"]:
        if cur_best == {}:
            return True
        if score > cur_best[main_metric]:
            return True
        return False
    else:
        raise ValueError("Task not supported")