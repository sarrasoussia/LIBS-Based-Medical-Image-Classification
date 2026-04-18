from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from data.pathmnist import labels_to_long
from utils.metrics import classification_metrics


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    train_grad_norm: List[float]
    val_ece: List[float]
    val_brier: List[float]
    val_f1_macro: List[float]
    best_epoch: int
    best_val_loss: float


def _format_loss(value: float) -> str:
    if abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def reset_weights(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def reset_bn(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.reset_running_stats()


def _max_abs_weight(model: nn.Module) -> float:
    max_abs = 0.0
    for parameter in model.parameters():
        if parameter.numel() == 0:
            continue
        max_abs = max(max_abs, float(parameter.detach().abs().max().item()))
    return max_abs


def _extract_ga_usage_metrics(model: nn.Module) -> Dict[str, float] | None:
    analyzer = getattr(model, "fusion_weight_analysis", None)
    if not callable(analyzer):
        return None
    result = analyzer()
    if not isinstance(result, dict):
        return None
    sanitized: Dict[str, float] = {}
    for key, value in result.items():
        try:
            sanitized[str(key)] = float(value)
        except Exception:
            continue
    return sanitized if sanitized else None


def _logit_temperature_for_epoch(epoch: int, total_epochs: int, start: float = 2.0, end: float = 1.0) -> float:
    if total_epochs <= 1:
        return end
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return float(start + (end - start) * progress)


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    adam_betas: tuple[float, float],
) -> Optimizer:
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=adam_betas,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=False,
        )
    raise ValueError(f"Unsupported optimizer_name='{optimizer_name}'. Use one of: adam, sgd")


def _build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    epochs: int,
    step_size: int,
    gamma: float,
    min_learning_rate: float,
) -> LRScheduler | None:
    name = scheduler_name.lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(epochs), 1),
            eta_min=float(min_learning_rate),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(step_size), 1),
            gamma=float(gamma),
        )
    raise ValueError(f"Unsupported scheduler_name='{scheduler_name}'. Use one of: none, cosine, step")


def _epoch_step(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    expected_num_classes: int | None = None,
    debug_label_transfer: bool = False,
    grad_clip_max_norm: float | None = 1.0,
    logit_abs_threshold: float = 10.0,
    logit_temperature: float = 2.0,
    collect_epoch_metrics: bool = False,
) -> Tuple[
    float,
    float,
    float | None,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    Tuple[float, float, float, float],
    Dict[str, object] | None,
]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    grad_norm_total = 0.0
    grad_norm_batches = 0
    logits_mean_total = 0.0
    logits_std_total = 0.0
    logits_batches = 0
    max_logit_abs = 0.0
    logit_p99_total = 0.0
    printed_transfer_debug = False
    printed_image_transfer_debug = False
    total_batches = 0
    skipped_nan_batches = 0
    input_mean_total = 0.0
    input_std_total = 0.0
    raw_mean_total = 0.0
    raw_std_total = 0.0
    raw_batches = 0
    sobel_mean_total = 0.0
    sobel_std_total = 0.0
    sobel_batches = 0
    fusion_mean_total = 0.0
    fusion_std_total = 0.0
    fusion_batches = 0
    entropy_total = 0.0
    confidence_total = 0.0
    confidence_counts = [0, 0, 0, 0]  # <0.5, [0.5,0.7), [0.7,0.9), >=0.9
    y_true_epoch: list[int] = []
    y_pred_epoch: list[int] = []
    y_conf_epoch: list[float] = []
    y_prob_epoch: list[list[float]] = []
    y_entropy_epoch: list[float] = []
    y_logit_abs_epoch: list[float] = []

    for images, labels in loader:
        total_batches += 1
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

        images_cpu = images.to(torch.float32).contiguous()
        if not torch.isfinite(images_cpu).all():
            images_cpu = torch.nan_to_num(images_cpu, nan=0.0, posinf=1.0, neginf=0.0)

        images = images_cpu.to(device, non_blocking=(device.type != "mps"))
        if not torch.isfinite(images).all():
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        labels = labels_cpu.to(device)

        if not torch.isfinite(images).all():
            skipped_nan_batches += 1
            continue

        batch_input_mean = float(images.mean().item())
        batch_input_std = float(images.std(unbiased=False).item())
        input_mean_total += batch_input_mean
        input_std_total += batch_input_std
        assert labels.dtype == torch.int64, (
            f"Invalid label dtype after transfer: {labels.dtype} (expected torch.int64)"
        )
        num_classes_for_assert = expected_num_classes if expected_num_classes is not None else None
        if num_classes_for_assert is not None:
            assert labels.min().item() >= 0 and labels.max().item() < num_classes_for_assert, (
                f"Invalid labels after transfer: min={labels.min().item()}, "
                f"max={labels.max().item()}, num_classes={num_classes_for_assert}"
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train and hasattr(model, "input_adapter"):
            images = images.requires_grad_(True)

        with torch.set_grad_enabled(is_train):
            raw_logits = model(images)
            if not torch.isfinite(raw_logits).all():
                skipped_nan_batches += 1
                continue

            # Critical calibration/stability fix: temperature-scale logits before CE.
            logits = raw_logits / max(float(logit_temperature), 1e-6)

            raw_stats_fn = getattr(model, "raw_feature_stats", None)
            if callable(raw_stats_fn):
                raw_mean, raw_std = raw_stats_fn()
                raw_mean_total += float(raw_mean)
                raw_std_total += float(raw_std)
                raw_batches += 1

            sobel_stats_fn = getattr(model, "sobel_feature_stats", None)
            if callable(sobel_stats_fn):
                sobel_mean, sobel_std = sobel_stats_fn()
                sobel_mean_total += float(sobel_mean)
                sobel_std_total += float(sobel_std)
                sobel_batches += 1

            fusion_stats_fn = getattr(model, "fusion_output_stats", None)
            if callable(fusion_stats_fn):
                fusion_mean, fusion_std = fusion_stats_fn()
                fusion_mean_total += float(fusion_mean)
                fusion_std_total += float(fusion_std)
                fusion_batches += 1

            batch_logit_mean = float(logits.mean().item())
            batch_logit_std = float(logits.std().item())
            logits_mean_total += batch_logit_mean
            logits_std_total += batch_logit_std
            logits_batches += 1

            abs_logits = logits.detach().abs()
            batch_max_abs = float(abs_logits.max().item())
            batch_p99 = float(torch.quantile(abs_logits.flatten(), 0.99).item())
            max_logit_abs = max(max_logit_abs, batch_max_abs)
            logit_p99_total += batch_p99

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
            if not torch.isfinite(loss):
                skipped_nan_batches += 1
                continue

            probs = torch.softmax(logits, dim=1)
            conf = probs.max(dim=1).values
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            confidence_total += float(conf.sum().item())
            entropy_total += float(entropy.sum().item())
            confidence_counts[0] += int((conf < 0.5).sum().item())
            confidence_counts[1] += int(((conf >= 0.5) & (conf < 0.7)).sum().item())
            confidence_counts[2] += int(((conf >= 0.7) & (conf < 0.9)).sum().item())
            confidence_counts[3] += int((conf >= 0.9).sum().item())

            preds = torch.argmax(logits, dim=1)
            if collect_epoch_metrics:
                y_true_epoch.extend(labels.detach().cpu().tolist())
                y_pred_epoch.extend(preds.detach().cpu().tolist())
                y_conf_epoch.extend(conf.detach().cpu().tolist())
                y_prob_epoch.extend(probs.detach().cpu().tolist())
                y_entropy_epoch.extend(entropy.detach().cpu().tolist())
                y_logit_abs_epoch.extend(logits.detach().abs().max(dim=1).values.cpu().tolist())

            if is_train:
                loss.backward()

                fusion_grad_fn = getattr(model, "last_fusion_grad_norm", None)
                if callable(fusion_grad_fn):
                    fusion_grad_norm = fusion_grad_fn()
                    if fusion_grad_norm is not None:
                        fusion_grad_norm_total += float(fusion_grad_norm)
                        fusion_grad_batches += 1

                fusion_weight_grad_fn = getattr(model, "fusion_weight_grad_norm", None)
                if callable(fusion_weight_grad_fn):
                    fusion_weight_grad_norm = fusion_weight_grad_fn()
                    if fusion_weight_grad_norm is not None:
                        fusion_weight_grad_norm_total += float(fusion_weight_grad_norm)
                        fusion_weight_grad_batches += 1

                if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
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
    avg_logits_mean = logits_mean_total / max(logits_batches, 1)
    avg_logits_std = logits_std_total / max(logits_batches, 1)
    avg_logit_p99 = logit_p99_total / max(logits_batches, 1)
    avg_input_mean = input_mean_total / max(total_batches, 1)
    avg_input_std = input_std_total / max(total_batches, 1)
    avg_raw_mean = raw_mean_total / max(raw_batches, 1) if raw_batches > 0 else 0.0
    avg_raw_std = raw_std_total / max(raw_batches, 1) if raw_batches > 0 else 0.0
    avg_sobel_mean = sobel_mean_total / max(sobel_batches, 1) if sobel_batches > 0 else 0.0
    avg_sobel_std = sobel_std_total / max(sobel_batches, 1) if sobel_batches > 0 else 0.0
    avg_fusion_mean = fusion_mean_total / max(fusion_batches, 1) if fusion_batches > 0 else 0.0
    avg_fusion_std = fusion_std_total / max(fusion_batches, 1) if fusion_batches > 0 else 0.0
    nan_batch_pct = (100.0 * skipped_nan_batches) / max(total_batches, 1)
    avg_entropy = entropy_total / max(total, 1)
    avg_confidence = confidence_total / max(total, 1)
    conf_distribution = tuple((100.0 * c) / max(total, 1) for c in confidence_counts)
    avg_fusion_grad_norm = (
        fusion_grad_norm_total / max(fusion_grad_batches, 1) if fusion_grad_batches > 0 else 0.0
    )
    avg_fusion_weight_grad_norm = (
        fusion_weight_grad_norm_total / max(fusion_weight_grad_batches, 1)
        if fusion_weight_grad_batches > 0
        else 0.0
    )
    epoch_metrics: Dict[str, object] | None = None
    if collect_epoch_metrics and len(y_true_epoch) > 0:
        epoch_metrics = classification_metrics(
            y_true=np.array(y_true_epoch, dtype=np.int64),
            y_pred=np.array(y_pred_epoch, dtype=np.int64),
            y_conf=np.array(y_conf_epoch, dtype=np.float64),
            y_prob=np.array(y_prob_epoch, dtype=np.float64),
            entropy=np.array(y_entropy_epoch, dtype=np.float64),
            logit_abs=np.array(y_logit_abs_epoch, dtype=np.float64),
        )
    return (
        avg_loss,
        accuracy,
        avg_grad_norm,
        avg_logits_mean,
        avg_logits_std,
        max_logit_abs,
        avg_logit_p99,
        avg_input_mean,
        avg_input_std,
        avg_raw_mean,
        avg_raw_std,
        avg_sobel_mean,
        avg_sobel_std,
        avg_fusion_mean,
        avg_fusion_std,
        nan_batch_pct,
        avg_entropy,
        avg_confidence,
        avg_fusion_grad_norm,
        avg_fusion_weight_grad_norm,
        conf_distribution,
        epoch_metrics,
    )


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
    auto_resume: bool = False,
    debug_sanity_checks: bool = False,
    fail_on_metric_inconsistency: bool = True,
    grad_clip_max_norm: float | None = 1.0,
    logit_abs_threshold: float = 10.0,
    exploding_loss_threshold: float = 10.0,
    logit_temperature_start: float = 1.0,
    logit_temperature_end: float = 1.0,
    label_smoothing: float = 0.1,
    calibration_weight: float = 1.0,
    optimizer_name: str = "adam",
    optimizer_momentum: float = 0.9,
    optimizer_adam_beta1: float = 0.9,
    optimizer_adam_beta2: float = 0.999,
    scheduler_name: str = "cosine",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    min_learning_rate: float = 1e-6,
) -> Tuple[nn.Module, TrainingHistory, str]:
    """Train model with early stopping and save best checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pt")
    last_checkpoint_path = os.path.join(save_dir, f"{model_name}_last.pt")
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    epoch_metrics_path = os.path.join(save_dir, f"{model_name}_epoch_metrics.json")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    effective_learning_rate = float(learning_rate)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError(f"[{model_name}] No trainable parameters found. Gradients cannot flow.")
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=effective_learning_rate,
        weight_decay=weight_decay,
        momentum=optimizer_momentum,
        adam_betas=(optimizer_adam_beta1, optimizer_adam_beta2),
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
        min_learning_rate=min_learning_rate,
    )

    best_val_loss = float("inf")
    best_selection_score = float("inf")
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
        val_ece=[],
        val_brier=[],
        val_f1_macro=[],
        best_epoch=0,
        best_val_loss=float("inf"),
    )
    epoch_metrics_log: List[Dict[str, object]] = []

    if len(train_loader) == 0:
        raise RuntimeError(f"[{model_name}] Train loader is empty.")
    if len(val_loader) == 0:
        raise RuntimeError(f"[{model_name}] Validation loader is empty.")

    # Optional sanity checks remain available via debug_sanity_checks, but are
    # disabled by default to keep console output focused on core training progress.
    if debug_sanity_checks:
        _ = next(iter(train_loader))

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
                model.train()
                model.apply(reset_bn)

                start_epoch = int(payload.get("epoch", 0)) + 1
                best_val_loss = float(payload.get("best_val_loss", best_val_loss))
                best_selection_score = float(payload.get("best_selection_score", best_selection_score))
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
                        val_ece=list(loaded_history.get("val_ece", [])),
                        val_brier=list(loaded_history.get("val_brier", [])),
                        val_f1_macro=list(loaded_history.get("val_f1_macro", [])),
                        best_epoch=int(loaded_history.get("best_epoch", 0)),
                        best_val_loss=float(loaded_history.get("best_val_loss", float("inf"))),
                    )

                if os.path.exists(checkpoint_path):
                    best_state_dict = torch.load(checkpoint_path, map_location="cpu")
                else:
                    best_state_dict = _state_dict_to_cpu(model)

                if start_epoch <= epochs:
                    if debug_sanity_checks:
                        print(
                            f"[{model_name}] Resuming from epoch {start_epoch}/{epochs} "
                            f"(best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}, "
                            "optimizer state not restored)."
                        )
                else:
                    if debug_sanity_checks:
                        print(
                            f"[{model_name}] Last checkpoint already reached epoch {start_epoch - 1}. "
                            "Skipping new training epochs."
                        )
        except Exception as exc:
            if debug_sanity_checks:
                print(f"[{model_name}] Could not resume from last checkpoint: {exc}. Starting fresh.")
            start_epoch = 1
            best_val_loss = float("inf")
            best_selection_score = float("inf")
            best_epoch = 0
            patience_counter = 0
            best_state_dict = _state_dict_to_cpu(model)
            history = TrainingHistory(
                train_loss=[],
                val_loss=[],
                train_acc=[],
                val_acc=[],
                train_grad_norm=[],
                val_ece=[],
                val_brier=[],
                val_f1_macro=[],
                best_epoch=0,
                best_val_loss=float("inf"),
            )

    for epoch in range(start_epoch, epochs + 1):
        current_temperature = _logit_temperature_for_epoch(
            epoch,
            epochs,
            start=logit_temperature_start,
            end=logit_temperature_end,
        )
        (
            train_loss,
            train_acc,
            train_grad_norm,
            train_logit_mean,
            train_logit_std,
            train_max_logit_abs,
            train_logit_p99,
            train_input_mean,
            train_input_std,
            train_raw_mean,
            train_raw_std,
            train_sobel_mean,
            train_sobel_std,
            train_fusion_mean,
            train_fusion_std,
            train_nan_batch_pct,
            train_entropy,
            train_confidence,
            train_fusion_grad_norm,
            train_fusion_weight_grad_norm,
            train_conf_dist,
            _,
        ) = _epoch_step(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            expected_num_classes=expected_num_classes,
            debug_label_transfer=debug_sanity_checks,
            grad_clip_max_norm=grad_clip_max_norm,
            logit_abs_threshold=logit_abs_threshold,
            logit_temperature=current_temperature,
            collect_epoch_metrics=False,
        )
        (
            val_loss,
            val_acc,
            _,
            val_logit_mean,
            val_logit_std,
            val_max_logit_abs,
            val_logit_p99,
            val_input_mean,
            val_input_std,
            val_raw_mean,
            val_raw_std,
            val_sobel_mean,
            val_sobel_std,
            val_fusion_mean,
            val_fusion_std,
            val_nan_batch_pct,
            val_entropy,
            val_confidence,
            val_fusion_grad_norm,
            val_fusion_weight_grad_norm,
            val_conf_dist,
            val_epoch_metrics,
        ) = _epoch_step(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            expected_num_classes=expected_num_classes,
            debug_label_transfer=False,
            grad_clip_max_norm=grad_clip_max_norm,
            logit_abs_threshold=logit_abs_threshold,
            logit_temperature=current_temperature,
            collect_epoch_metrics=True,
        )

        val_ece = float((val_epoch_metrics or {}).get("ece", 0.0))
        val_brier = float((val_epoch_metrics or {}).get("brier_score", 0.0))
        val_f1_macro = float((val_epoch_metrics or {}).get("f1_macro", 0.0))
        calibration_score = val_ece
        selection_score = val_loss + calibration_weight * calibration_score

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.train_grad_norm.append(float(train_grad_norm) if train_grad_norm is not None else 0.0)
        history.val_ece.append(val_ece)
        history.val_brier.append(val_brier)
        history.val_f1_macro.append(val_f1_macro)

        epoch_metrics_log.append(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "selection_score": float(selection_score),
                "calibration_score": float(calibration_score),
                "feature_stats": {
                    "train": {
                        "raw_mean": float(train_raw_mean),
                        "raw_std": float(train_raw_std),
                        "sobel_mean": float(train_sobel_mean),
                        "sobel_std": float(train_sobel_std),
                        "geometric_mean": float(train_ga_mean),
                        "geometric_std": float(train_ga_std),
                        "fusion_mean": float(train_fusion_mean),
                        "fusion_std": float(train_fusion_std),
                    },
                    "val": {
                        "raw_mean": float(val_raw_mean),
                        "raw_std": float(val_raw_std),
                        "sobel_mean": float(val_sobel_mean),
                        "sobel_std": float(val_sobel_std),
                        "geometric_mean": float(val_ga_mean),
                        "geometric_std": float(val_ga_std),
                        "fusion_mean": float(val_fusion_mean),
                        "fusion_std": float(val_fusion_std),
                    },
                },
                "fusion_diagnostics": {
                    "train": {
                        "fusion_grad_norm": float(train_fusion_grad_norm),
                        "fusion_weight_grad_norm": float(train_fusion_weight_grad_norm),
                    },
                    "val": {
                        "fusion_grad_norm": float(val_fusion_grad_norm),
                        "fusion_weight_grad_norm": float(val_fusion_weight_grad_norm),
                    },
                },
                "val_metrics": val_epoch_metrics or {},
            }
        )
        with open(epoch_metrics_path, "w", encoding="utf-8") as f:
            json.dump(epoch_metrics_log, f, indent=2)

        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {_format_loss(train_loss)}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {_format_loss(val_loss)}, Val Acc: {val_acc:.4f} | "
            f"Temp: {current_temperature:.2f} | "
            f"Input mean/std (train): {train_input_mean:.4f}/{train_input_std:.4f} | "
            f"Input mean/std (val): {val_input_mean:.4f}/{val_input_std:.4f} | "
            f"Raw feat mean/std (train): {train_raw_mean:.4f}/{train_raw_std:.4f} | "
            f"Raw feat mean/std (val): {val_raw_mean:.4f}/{val_raw_std:.4f} | "
            f"Sobel feat mean/std (train): {train_sobel_mean:.4f}/{train_sobel_std:.4f} | "
            f"Sobel feat mean/std (val): {val_sobel_mean:.4f}/{val_sobel_std:.4f} | "
            f"GA mean/std (train): {train_ga_mean:.4f}/{train_ga_std:.4f} | "
            f"GA mean/std (val): {val_ga_mean:.4f}/{val_ga_std:.4f} | "
            f"Fusion mean/std (train): {train_fusion_mean:.4f}/{train_fusion_std:.4f} | "
            f"Fusion mean/std (val): {val_fusion_mean:.4f}/{val_fusion_std:.4f} | "
            f"GA grad-norm (train): {train_ga_grad_norm:.4f} | "
            f"Fusion grad-norm (train): {train_fusion_grad_norm:.4f} | "
            f"Fusion grad-norm (val): {val_fusion_grad_norm:.4f} | "
            f"Fusion weight grad-norm (train): {train_fusion_weight_grad_norm:.4f} | "
            f"Fusion weight grad-norm (val): {val_fusion_weight_grad_norm:.4f} | "
            f"GradNorm: {0.0 if train_grad_norm is None else train_grad_norm:.4f} | "
            f"LR: {current_lr:.6g} | "
            f"Logits mean/std (train): {train_logit_mean:.4f}/{train_logit_std:.4f} | "
            f"Logits mean/std (val): {val_logit_mean:.4f}/{val_logit_std:.4f} | "
            f"Logit p99 (train/val): {train_logit_p99:.4f}/{val_logit_p99:.4f} | "
            f"Entropy (train/val): {train_entropy:.4f}/{val_entropy:.4f} | "
            f"Conf mean (train/val): {train_confidence:.4f}/{val_confidence:.4f} | "
            f"Val ECE/Brier/F1m: {val_ece:.4f}/{val_brier:.4f}/{val_f1_macro:.4f} | "
            f"Selection score: {selection_score:.6f} | "
            "Conf bins train% [<0.5,0.5-0.7,0.7-0.9,>=0.9]: "
            f"{train_conf_dist[0]:.1f}/{train_conf_dist[1]:.1f}/{train_conf_dist[2]:.1f}/{train_conf_dist[3]:.1f} | "
            "Conf bins val% [<0.5,0.5-0.7,0.7-0.9,>=0.9]: "
            f"{val_conf_dist[0]:.1f}/{val_conf_dist[1]:.1f}/{val_conf_dist[2]:.1f}/{val_conf_dist[3]:.1f} | "
            f"NaN batches % (train/val): {train_nan_batch_pct:.2f}/{val_nan_batch_pct:.2f} | "
            f"Max |W|: {_max_abs_weight(model):.4f}"
        )

        if val_epoch_metrics:
            if debug_sanity_checks:
                print(
                    f"[{model_name}] Val precision/recall per class: "
                    f"{val_epoch_metrics.get('precision_per_class', [])} / "
                    f"{val_epoch_metrics.get('recall_per_class', [])}"
                )
                print(
                    f"[{model_name}] Val per-class accuracy: "
                    f"{val_epoch_metrics.get('accuracy_per_class', [])}"
                )
                print(
                    f"[{model_name}] Val confusion matrix: "
                    f"{val_epoch_metrics.get('confusion_matrix', [])}"
                )

        if debug_sanity_checks:
            if train_ga_std != 0.0 and abs(train_ga_std) < 1e-5:
                print(f"[{model_name}] WARNING: GA train feature std is near-constant ({train_ga_std:.3e}).")
            if val_ga_std != 0.0 and abs(val_ga_std) < 1e-5:
                print(f"[{model_name}] WARNING: GA val feature std is near-constant ({val_ga_std:.3e}).")
            if train_raw_std != 0.0 and abs(train_raw_std) < 1e-5:
                print(f"[{model_name}] WARNING: raw train feature std is near-constant ({train_raw_std:.3e}).")
            if val_raw_std != 0.0 and abs(val_raw_std) < 1e-5:
                print(f"[{model_name}] WARNING: raw val feature std is near-constant ({val_raw_std:.3e}).")
            if train_sobel_std != 0.0 and abs(train_sobel_std) < 1e-5:
                print(f"[{model_name}] WARNING: sobel train feature std is near-constant ({train_sobel_std:.3e}).")
            if val_sobel_std != 0.0 and abs(val_sobel_std) < 1e-5:
                print(f"[{model_name}] WARNING: sobel val feature std is near-constant ({val_sobel_std:.3e}).")

        if (
            not torch.isfinite(torch.tensor(train_loss))
            or not torch.isfinite(torch.tensor(val_loss))
            or train_loss > exploding_loss_threshold
            or val_loss > exploding_loss_threshold
        ):
            break

        if debug_sanity_checks:
            ce_lower_bound = (1.0 - val_acc) * 0.69314718056
            if val_loss + 1e-12 < ce_lower_bound * 0.8 and val_acc < 0.999:
                message = (
                    f"[{model_name}] suspicious val metrics: "
                    f"val_loss={val_loss:.6e}, val_acc={val_acc:.6f}, "
                    f"expected_loss_lower_bound≈{ce_lower_bound:.6e}."
                )
                if fail_on_metric_inconsistency:
                    raise RuntimeError(message)

        if selection_score < best_selection_score:
            best_selection_score = selection_score
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = _state_dict_to_cpu(model)
            torch.save(best_state_dict, checkpoint_path)
        else:
            patience_counter += 1

        # Save resumable training state only if resume is explicitly enabled.
        if auto_resume:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _state_dict_to_cpu(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
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

    ga_usage_metrics = _extract_ga_usage_metrics(model)
    if ga_usage_metrics is not None:
        ga_usage_path = os.path.join(save_dir, f"{model_name}_ga_usage.json")
        payload = {
            "model_name": model_name,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "fusion_weight_analysis": ga_usage_metrics,
        }
        with open(ga_usage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if debug_sanity_checks:
            print(
                f"[{model_name}] Fusion weight analysis | "
                f"raw_mean_abs={ga_usage_metrics.get('raw_mean_abs_weight', 0.0):.6f} "
                f"ga_mean_abs={ga_usage_metrics.get('ga_mean_abs_weight', 0.0):.6f} "
                f"ga/raw={ga_usage_metrics.get('ga_to_raw_importance_ratio', 0.0):.6f}"
            )

    return model, history, checkpoint_path
