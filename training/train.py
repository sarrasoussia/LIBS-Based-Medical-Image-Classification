from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pathmnist import labels_to_long


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    best_epoch: int
    best_val_loss: float


def _epoch_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels_to_long(labels).to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
) -> Tuple[nn.Module, TrainingHistory, str]:
    """Train model with early stopping and save best checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_acc=[],
        val_acc=[],
        best_epoch=0,
        best_val_loss=float("inf"),
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch_step(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = _epoch_step(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(
                f"[{model_name}] Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val loss {best_val_loss:.4f}."
            )
            break

    model.load_state_dict(best_state_dict)
    history.best_epoch = best_epoch
    history.best_val_loss = best_val_loss

    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(asdict(history), f, indent=2)

    return model, history, checkpoint_path
