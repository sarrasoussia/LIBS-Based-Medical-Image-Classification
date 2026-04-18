from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pathmnist import labels_to_long
from utils.metrics import (
    classification_metrics,
    one_vs_rest_roc_auc,
    predictive_mutual_information,
    support_weighting_analysis,
)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_samples: int = 0,
    enable_mc_dropout: bool = False,
) -> Tuple[list[int], list[int], list[float], list[list[float]], list[float], list[float], list[np.ndarray]]:
    y_true, y_pred, y_conf, y_prob, y_entropy, y_logit_abs = [], [], [], [], [], []
    mc_prob_samples: list[np.ndarray] = []

    for images, labels in tqdm(loader, leave=False):
        images_cpu = images.to(torch.float32).contiguous()
        if not torch.isfinite(images_cpu).all():
            images_cpu = torch.nan_to_num(images_cpu, nan=0.0, posinf=1.0, neginf=0.0)
        images = images_cpu.to(device, non_blocking=(device.type != "mps"))
        if not torch.isfinite(images).all():
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        labels_cpu = labels_to_long(labels)
        labels_cpu = labels_cpu.squeeze()
        labels_cpu = labels_cpu.to(torch.int64)
        labels_cpu = labels_cpu.contiguous()
        if labels_cpu.ndim == 0:
            labels_cpu = labels_cpu.unsqueeze(0)
        labels = labels_cpu.to(device, non_blocking=True)

        if mc_samples and mc_samples > 1:
            sample_probs = []
            model_mode = model.training
            if enable_mc_dropout:
                model.train()
            for _ in range(mc_samples):
                logits = model(images)
                sample_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
            if enable_mc_dropout and not model_mode:
                model.eval()
            probs = torch.from_numpy(np.mean(np.stack(sample_probs, axis=0), axis=0)).to(images.device)
            mc_prob_samples.append(np.stack(sample_probs, axis=0))
            logits = torch.log(torch.clamp(probs, min=1e-12))
        else:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
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
        preds = torch.argmax(logits, dim=1)
        conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        logit_abs = logits.abs().max(dim=1).values

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        y_entropy.extend(entropy.cpu().numpy().tolist())
        y_logit_abs.extend(logit_abs.cpu().numpy().tolist())

    return y_true, y_pred, y_conf, y_prob, y_entropy, y_logit_abs, mc_prob_samples


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str,
    save_predictions: bool = True,
    mc_samples: int = 0,
    enable_mc_dropout: bool = False,
    run_ga_ablation: bool = True,
) -> Dict[str, object]:
    model.eval()

    y_true, y_pred, y_conf, y_prob, y_entropy, y_logit_abs, mc_prob_samples = _collect_predictions(
        model=model,
        loader=loader,
        device=device,
        mc_samples=mc_samples,
        enable_mc_dropout=enable_mc_dropout,
    )

    metrics = classification_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_conf),
        y_prob=np.array(y_prob),
        entropy=np.array(y_entropy),
        logit_abs=np.array(y_logit_abs),
    )
    auc_report = one_vs_rest_roc_auc(np.array(y_true), np.array(y_prob))
    support_report = support_weighting_analysis(
        np.array(y_true),
        np.array(y_pred),
        num_classes=np.array(y_prob).shape[1],
    )
    metrics.update(
        {
            "roc_auc_ovr_per_class": auc_report["per_class_auc"],
            "roc_auc_ovr_macro": auc_report["macro_auc"],
            "roc_auc_ovr_micro": auc_report["micro_auc"],
            "support_analysis": support_report,
        }
    )
    metrics["summary_table"] = {
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "ece": metrics.get("ece"),
        "brier_score": metrics.get("brier_score"),
        "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
    }
    if mc_prob_samples:
        mi = predictive_mutual_information(np.concatenate(mc_prob_samples, axis=1))
        metrics["mutual_information"] = {
            "mean": float(np.mean(mi)),
            "per_sample": mi.tolist(),
        }

    fusion_analysis_fn = getattr(model, "fusion_weight_analysis", None)
    if callable(fusion_analysis_fn):
        metrics["fusion_weight_analysis"] = {
            k: float(v) for k, v in fusion_analysis_fn().items()
        }
        fwa = metrics["fusion_weight_analysis"]
        print(
            "[eval] fusion weights | "
            f"raw_mean_abs={fwa.get('raw_mean_abs_weight', 0.0):.6f} "
            f"ga_mean_abs={fwa.get('ga_mean_abs_weight', 0.0):.6f} "
            f"ga/raw={fwa.get('ga_to_raw_importance_ratio', 0.0):.6f}"
        )

    ablation_fn = getattr(model, "set_ga_ablation", None)
    if run_ga_ablation and callable(ablation_fn):
        ablation_fn(True)
        try:
            (
                y_true_abl,
                y_pred_abl,
                y_conf_abl,
                y_prob_abl,
                y_entropy_abl,
                y_logit_abs_abl,
                _,
            ) = _collect_predictions(
                model=model,
                loader=loader,
                device=device,
                mc_samples=0,
                enable_mc_dropout=False,
            )
        finally:
            ablation_fn(False)

        ablated_metrics = classification_metrics(
            np.array(y_true_abl),
            np.array(y_pred_abl),
            np.array(y_conf_abl),
            y_prob=np.array(y_prob_abl),
            entropy=np.array(y_entropy_abl),
            logit_abs=np.array(y_logit_abs_abl),
        )
        metrics["ga_ablation"] = {
            "accuracy_with_ga": float(metrics.get("accuracy", 0.0)),
            "accuracy_without_ga": float(ablated_metrics.get("accuracy", 0.0)),
            "accuracy_drop": float(metrics.get("accuracy", 0.0) - ablated_metrics.get("accuracy", 0.0)),
            "f1_macro_with_ga": float(metrics.get("f1_macro", 0.0)),
            "f1_macro_without_ga": float(ablated_metrics.get("f1_macro", 0.0)),
            "f1_macro_drop": float(metrics.get("f1_macro", 0.0) - ablated_metrics.get("f1_macro", 0.0)),
        }
        ga_ab = metrics["ga_ablation"]
        print(
            "[eval] GA ablation | "
            f"acc_with={ga_ab.get('accuracy_with_ga', 0.0):.4f} "
            f"acc_without={ga_ab.get('accuracy_without_ga', 0.0):.4f} "
            f"drop={ga_ab.get('accuracy_drop', 0.0):.4f}"
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = Path(save_path).with_suffix(".csv")
    flat_rows = []
    summary_keys = ["accuracy", "f1_macro", "ece", "brier_score", "roc_auc_ovr_macro", "roc_auc_ovr_micro"]
    flat_rows.append({key: metrics.get(key) for key in summary_keys})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerow(flat_rows[0])

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
                    "y_prob": y_prob,
                    "y_entropy": y_entropy,
                    "y_logit_abs": y_logit_abs,
                    "roc_auc_ovr_per_class": metrics.get("roc_auc_ovr_per_class"),
                    "support_analysis": metrics.get("support_analysis"),
                },
                f,
                indent=2,
            )
    return metrics
