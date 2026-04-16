from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pathmnist import labels_to_long
from utils.metrics import classification_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    save_predictions: bool = True,
) -> Dict[str, object]:
    model.eval()

    y_true, y_pred, y_conf = [], [], []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels_cpu = labels_to_long(labels)
        labels_cpu = labels_cpu.squeeze()
        labels_cpu = labels_cpu.to(torch.int64)
        labels_cpu = labels_cpu.contiguous()
        if labels_cpu.ndim == 0:
            labels_cpu = labels_cpu.unsqueeze(0)
        labels = labels_cpu.to(device, non_blocking=True)

        logits = model(images)
        assert logits.ndim == 2, (
            f"CrossEntropy-compatible logits expected shape [B, C], got {tuple(logits.shape)}"
        )
        assert labels.ndim == 1, (
            f"Class-index labels expected shape [B], got {tuple(labels.shape)}"
        )
        assert labels.dtype == torch.int64, (
            f"Invalid label dtype after transfer: {labels.dtype} (expected torch.int64)"
        )
        assert labels.min().item() >= 0 and labels.max().item() < logits.shape[1], (
            f"Invalid labels after transfer: min={labels.min().item()}, "
            f"max={labels.max().item()}, num_classes={logits.shape[1]}"
        )
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())

    metrics = classification_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_conf),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if save_predictions:
        prediction_path = Path(save_path).with_name(
            Path(save_path).stem.replace("_metrics", "") + "_predictions.json"
        )
        with open(prediction_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_conf": y_conf,
                },
                f,
                indent=2,
            )
    return metrics
