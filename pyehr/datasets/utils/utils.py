import os
from typing import Literal

import torch
from torch.nn.utils.rnn import unpad_sequence


def unpad_y(preds, labels, lens, pred_type: Literal['n-1', 'n'] = 'n-1'):
    raw_device = preds.device
    device = torch.device("cpu")
    preds, labels, lens = preds.squeeze(dim=-1).to(device), labels.squeeze(dim=-1).to(device), lens.to(device)
    if preds.dim() == 2:  # [batch_size, seq_len]
        preds_unpad = unpad_sequence(preds, batch_first=True, lengths=lens)
        if pred_type == 'n-1':
            preds_unpad = [pred[-1] for pred in preds_unpad]
        preds = torch.vstack(preds_unpad)
    preds = preds.squeeze(dim=-1)
    if labels.dim() == 2: # [batch_size, seq_len]
        labels_unpad = unpad_sequence(labels, batch_first=True, lengths=lens)
        if pred_type == 'n-1':
            labels_unpad = [label[-1] for label in labels_unpad]
        labels = torch.vstack(labels_unpad)
    labels = labels.squeeze(dim=-1)
    assert preds.shape == labels.shape, f"preds.shape: {preds.shape}, labels.shape: {labels.shape}"
    return preds.to(raw_device), labels.to(raw_device)


def unpad_batch(x, y, lens):
    x = x.detach().cpu()
    y = y.detach().cpu()
    lens = lens.detach().cpu()
    x_unpad = unpad_sequence(x, batch_first=True, lengths=lens)
    x_unpad = [x[-1] for x in x_unpad]
    x_stack = torch.vstack(x_unpad).squeeze(dim=-1)
    y_unpad = unpad_sequence(y, batch_first=True, lengths=lens)
    y_unpad = [y[-1] for y in y_unpad]
    y_stack = torch.vstack(y_unpad).squeeze(dim=-1)
    return x_stack.numpy().squeeze(), y_stack.numpy().squeeze()