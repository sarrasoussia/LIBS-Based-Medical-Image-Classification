from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("talk", font_scale=0.85)


def plot_training_curves(history: Dict[str, List[float]], save_path: str, title: str) -> None:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_gradcam_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    preds: Sequence[int],
    labels: Sequence[int],
    save_path: str,
    title: str,
    max_items: int = 12,
) -> None:
    """Save side-by-side image + Grad-CAM overlays."""
    _setup_style()
    n = min(max_items, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), dpi=170)
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        img = np.transpose(images[i], (1, 2, 0))
        cam = heatmaps[i, 0]
        ax.imshow(img)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"y={labels[i]} | ŷ={preds[i]}", fontsize=10)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: Dict[str, Dict[str, float]], save_path: str) -> None:
    _setup_style()
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"]
    model_names = list(results.keys())

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=180)
    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metric_names]
        offset = (i - (len(model_names) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Test Metrics")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
