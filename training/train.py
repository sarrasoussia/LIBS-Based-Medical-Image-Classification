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

from data.pathmnist import labels_to_long


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    train_grad_norm: List[float]
    best_epoch: int
    best_val_loss: float


def _format_loss(value: float) -> str:
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _epoch_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    expected_num_classes: int | None = None,
    debug_label_transfer: bool = False,
) -> Tuple[float, float, float | None]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    grad_norm_total = 0.0
    grad_norm_batches = 0
    printed_transfer_debug = False

    for images, labels in loader:
        # Keep label sanitation/checks on CPU first to avoid device-specific
        # integer reduction artifacts (notably on MPS with int64).
        labels_cpu = labels_to_long(labels)
        labels_cpu = labels_cpu.squeeze()
        labels_cpu = labels_cpu.to(torch.int64)
        labels_cpu = labels_cpu.contiguous()
        if labels_cpu.ndim == 0:
            labels_cpu = labels_cpu.unsqueeze(0)

        label_min = int(labels_cpu.min().item())
        label_max = int(labels_cpu.max().item())

        images = images.to(device, non_blocking=True)
        labels = labels_cpu.to(device)

        assert labels.dtype == torch.int64, (
            f"Invalid label dtype after transfer: {labels.dtype} (expected torch.int64)"
        )
        num_classes_for_assert = expected_num_classes if expected_num_classes is not None else None
        if num_classes_for_assert is not None:
            assert labels.min().item() >= 0 and labels.max().item() < num_classes_for_assert, (
                f"Invalid labels after transfer: min={labels.min().item()}, "
                f"max={labels.max().item()}, num_classes={num_classes_for_assert}"
            )

        if debug_label_transfer and not printed_transfer_debug:
            print(f"Before transfer: {labels_cpu[:10].tolist()}")
            print(f"After transfer: {labels[:10].detach().cpu().tolist()}")
            printed_transfer_debug = True

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            assert logits.ndim == 2, (
                f"CrossEntropyLoss expects logits with shape [B, C], got {tuple(logits.shape)}"
            )
            assert labels.ndim == 1, (
                f"CrossEntropyLoss expects class-index labels shape [B], got {tuple(labels.shape)}"
            )
            assert logits.shape[0] == labels.shape[0], (
                f"Batch mismatch: logits batch={logits.shape[0]}, labels batch={labels.shape[0]}"
            )
            num_classes = expected_num_classes or int(logits.shape[1])
            assert label_min >= 0 and label_max < num_classes, (
                f"Invalid labels: min={label_min}, max={label_max}, num_classes={num_classes}"
            )
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                squared_norm = 0.0
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    param_norm = parameter.grad.detach().data.norm(2).item()
                    squared_norm += param_norm * param_norm
                grad_norm_total += squared_norm ** 0.5
                grad_norm_batches += 1
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    avg_grad_norm = grad_norm_total / grad_norm_batches if grad_norm_batches > 0 else None
    return avg_loss, accuracy, avg_grad_norm


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
    auto_resume: bool = True,
    debug_sanity_checks: bool = True,
    fail_on_metric_inconsistency: bool = True,
) -> Tuple[nn.Module, TrainingHistory, str]:
    """Train model with early stopping and save best checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")
    last_checkpoint_path = os.path.join(save_dir, f"{model_name}_last.pt")
    history_path = os.path.join(save_dir, f"{model_name}_history.json")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state_dict = _state_dict_to_cpu(model)
    start_epoch = 1

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_acc=[],
        val_acc=[],
        train_grad_norm=[],
        best_epoch=0,
        best_val_loss=float("inf"),
    )

    if len(train_loader) == 0:
        raise RuntimeError(f"[{model_name}] Train loader is empty.")
    if len(val_loader) == 0:
        raise RuntimeError(f"[{model_name}] Validation loader is empty.")

    if debug_sanity_checks:
        print(
            f"[{model_name}] Loader sanity | train_batches={len(train_loader)} "
            f"val_batches={len(val_loader)}"
        )
        sample_images, sample_labels = next(iter(train_loader))
        sample_labels = labels_to_long(sample_labels)
        print(f"[{model_name}] Label sample (first 20): {sample_labels[:20].tolist()}")
        print(
            f"[{model_name}] Label range: {int(sample_labels.min().item())} "
            f"to {int(sample_labels.max().item())}"
        )

    # infer class count once for strict label assertions
    with torch.no_grad():
        sample_images, _ = next(iter(train_loader))
        sample_logits = model(sample_images.to(device, non_blocking=True))
        expected_num_classes = int(sample_logits.shape[1])

    if auto_resume and os.path.exists(last_checkpoint_path):
        try:
            payload = torch.load(last_checkpoint_path, map_location="cpu")
            if isinstance(payload, dict) and "model_state_dict" in payload:
                model.load_state_dict(payload["model_state_dict"])
                if "optimizer_state_dict" in payload:
                    optimizer.load_state_dict(payload["optimizer_state_dict"])

                start_epoch = int(payload.get("epoch", 0)) + 1
                best_val_loss = float(payload.get("best_val_loss", best_val_loss))
                best_epoch = int(payload.get("best_epoch", best_epoch))
                patience_counter = int(payload.get("patience_counter", patience_counter))

                loaded_history = payload.get("history")
                if isinstance(loaded_history, dict):
                    history = TrainingHistory(
                        train_loss=list(loaded_history.get("train_loss", [])),
                        val_loss=list(loaded_history.get("val_loss", [])),
                        train_acc=list(loaded_history.get("train_acc", [])),
                        val_acc=list(loaded_history.get("val_acc", [])),
                        train_grad_norm=list(loaded_history.get("train_grad_norm", [])),
                        best_epoch=int(loaded_history.get("best_epoch", 0)),
                        best_val_loss=float(loaded_history.get("best_val_loss", float("inf"))),
                    )

                if os.path.exists(checkpoint_path):
                    best_state_dict = torch.load(checkpoint_path, map_location="cpu")
                else:
                    best_state_dict = _state_dict_to_cpu(model)

                if start_epoch <= epochs:
                    print(
                        f"[{model_name}] Resuming from epoch {start_epoch}/{epochs} "
                        f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f})."
                    )
                else:
                    print(
                        f"[{model_name}] Last checkpoint already reached epoch {start_epoch - 1}. "
                        "Skipping new training epochs."
                    )
        except Exception as exc:
            print(f"[{model_name}] Could not resume from last checkpoint: {exc}. Starting fresh.")
            start_epoch = 1
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0
            best_state_dict = _state_dict_to_cpu(model)
            history = TrainingHistory(
                train_loss=[],
                val_loss=[],
                train_acc=[],
                val_acc=[],
                train_grad_norm=[],
                best_epoch=0,
                best_val_loss=float("inf"),
            )

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc, train_grad_norm = _epoch_step(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            expected_num_classes=expected_num_classes,
            debug_label_transfer=debug_sanity_checks,
        )
        val_loss, val_acc, _ = _epoch_step(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            expected_num_classes=expected_num_classes,
            debug_label_transfer=False,
        )

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.train_grad_norm.append(float(train_grad_norm) if train_grad_norm is not None else 0.0)

        print(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {_format_loss(train_loss)}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {_format_loss(val_loss)}, Val Acc: {val_acc:.4f}"
        )

        if debug_sanity_checks:
            # For multiclass CE with argmax misclassifications,
            # loss has a rough lower bound: (1-acc) * log(2).
            ce_lower_bound = (1.0 - val_acc) * 0.69314718056
            if val_loss + 1e-12 < ce_lower_bound * 0.8 and val_acc < 0.999:
                message = (
                    f"[{model_name}] suspicious val metrics: "
                    f"val_loss={val_loss:.6e}, val_acc={val_acc:.6f}, "
                    f"expected_loss_lower_bound≈{ce_lower_bound:.6e}."
                )
                if fail_on_metric_inconsistency:
                    raise RuntimeError(message)
                print(f"WARNING: {message}")

            if epoch == start_epoch:
                model.eval()
                with torch.no_grad():
                    sample_images, sample_labels = next(iter(val_loader))
                    sample_images = sample_images.to(device, non_blocking=True)
                    sample_labels_cpu = labels_to_long(sample_labels)
                    sample_logits = model(sample_images)
                    sample_preds = torch.argmax(sample_logits, dim=1)
                    print(
                        f"[{model_name}] Debug logits min/max: "
                        f"{sample_logits.min().item():.4f}/{sample_logits.max().item():.4f}"
                    )
                    print(
                        f"[{model_name}] Debug preds vs labels (first 10): "
                        f"{list(zip(sample_preds[:10].cpu().tolist(), sample_labels_cpu[:10].tolist()))}"
                    )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = _state_dict_to_cpu(model)
            torch.save(best_state_dict, checkpoint_path)
        else:
            patience_counter += 1

        # Save resumable training state every epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": _state_dict_to_cpu(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": asdict(history),
            },
            last_checkpoint_path,
        )

        if patience_counter >= early_stopping_patience:
            print(
                f"[{model_name}] Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val loss {best_val_loss:.4f}."
            )
            break

    model.load_state_dict(best_state_dict)
    history.best_epoch = best_epoch
    history.best_val_loss = best_val_loss

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(asdict(history), f, indent=2)

    return model, history, checkpoint_path
