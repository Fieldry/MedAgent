import torch
from torch import nn
import torch.nn.functional as F


def get_loss(preds, labels, task, time_aware=False):
    if task in ["outcome", "mortality", "readmission"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 2] if task == "readmission" else labels[:, 0]
        loss = F.binary_cross_entropy(preds, labels)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        loss = F.mse_loss(preds, labels)
    elif task == "multitask":
        loss = get_multitask_loss(preds[:, 0], preds[:, 1], labels[:, 0], labels[:, 1])

    # If use time aware loss:
    if task in ["outcome", "mortality", "readmission"] and time_aware:
        loss = get_time_aware_loss(preds, labels)

    return loss


def get_multitask_loss(outcome_preds, los_preds, outcome_labels, los_labels, task_num=2):
    alpha = nn.Parameter(torch.ones(task_num))
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    loss0 = bce(outcome_preds, outcome_labels)
    loss1 = mse(los_preds, los_labels)
    return loss0 * alpha[0] + loss1 * alpha[1]


def get_time_aware_loss(outcome_preds, outcome_labels, los_labels, decay_rate=0.1, reward_factor=0.1):
    bce = nn.BCELoss()
    los_weights = torch.exp(-decay_rate * los_labels)  # Exponential decay
    loss_unreduced = bce(outcome_preds, outcome_labels)

    reward_term = (los_labels * torch.abs(outcome_labels - outcome_preds)).mean()  # Reward term
    loss = (loss_unreduced * los_weights).mean() - reward_factor * reward_term  # Weighted loss
    return torch.clamp(loss, min=0)