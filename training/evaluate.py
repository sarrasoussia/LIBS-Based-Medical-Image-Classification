from __future__ import annotations

import json
import os
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
) -> Dict[str, object]:
    model.eval()

    y_true, y_pred, y_conf = [], [], []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels_to_long(labels).to(device, non_blocking=True)

        logits = model(images)
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
    return metrics
