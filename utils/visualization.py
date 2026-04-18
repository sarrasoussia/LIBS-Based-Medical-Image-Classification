from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MODEL_DISPLAY_NAME = {
    "baseline": "Baseline CNN",
    "baseline_cnn": "Baseline CNN",
    "libs_cnn": "LIBS-CNN",
    "densenet121": "DenseNet121",
    "libs_densenet121": "LIBS-DenseNet121",
}

MODEL_COLOR = {
    "baseline": "#1f77b4",
    "baseline_cnn": "#1f77b4",
    "libs_cnn": "#ff7f0e",
    "densenet121": "#2ca02c",
    "libs_densenet121": "#d62728",
}


def _display_model_name(name: str) -> str:
    return MODEL_DISPLAY_NAME.get(name, name)


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
        ax.bar(
            x + offset,
            values,
            width=width,
            label=_display_model_name(model_name),
            color=MODEL_COLOR.get(model_name, None),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Test Metrics (LIBS terminology)")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_statistics(
    feature_stats: Dict[str, Dict[str, tuple[float, float]]],
    save_path: str,
    title: str = "Feature Statistics",
) -> None:
    """Plot mean/std for raw, geometric (Sobel), and fused features."""
    _setup_style()
    branches = ["raw", "geometric", "fusion"]
    labels = {
        "raw": "Raw input stream",
        "geometric": "Geometric feature stream",
        "fusion": "Learnable fusion output",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=180)
    for model_name, stats in feature_stats.items():
        means = [stats.get(branch, (np.nan, np.nan))[0] for branch in branches]
        stds = [stats.get(branch, (np.nan, np.nan))[1] for branch in branches]
        x = np.arange(len(branches))
        axes[0].plot(x, means, marker="o", linewidth=2, label=_display_model_name(model_name), color=MODEL_COLOR.get(model_name))
        axes[1].plot(x, stds, marker="s", linewidth=2, label=_display_model_name(model_name), color=MODEL_COLOR.get(model_name))

    for ax, ylab, subtitle in [
        (axes[0], "Mean", "Branch-wise Mean"),
        (axes[1], "Std", "Branch-wise Std"),
    ]:
        ax.set_xticks(np.arange(len(branches)))
        ax.set_xticklabels([labels[b] for b in branches], rotation=15, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(subtitle)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_behavior(
    fusion_weight_stats: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Fusion Behavior",
) -> None:
    """Plot fusion weight magnitudes for raw vs geometric branches."""
    _setup_style()
    models = list(fusion_weight_stats.keys())
    x = np.arange(len(models))
    raw_vals = [fusion_weight_stats[m].get("raw_mean_abs_weight", np.nan) for m in models]
    geom_vals = [fusion_weight_stats[m].get("sobel_mean_abs_weight", np.nan) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    width = 0.35
    ax.bar(x - width / 2, raw_vals, width=width, label="Raw stream weights", color="#1f77b4")
    ax.bar(x + width / 2, geom_vals, width=width, label="Geometric stream weights", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_model_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Mean |weight|")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(ablation_payload: Dict[str, Dict[str, float]], save_path: str, title: str = "Auxiliary Branch Ablation") -> None:
    """Compare full LIBS model vs geometric-branch-disabled inference."""
    _setup_style()
    models = list(ablation_payload.keys())
    x = np.arange(len(models))
    width = 0.35
    with_aux = [ablation_payload[m].get("accuracy_full", np.nan) for m in models]
    without_aux = [ablation_payload[m].get("accuracy_no_geom", np.nan) for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    ax.bar(x - width / 2, with_aux, width=width, label="Full model", color="#ff7f0e")
    ax.bar(x + width / 2, without_aux, width=width, label="Geometric branch disabled", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_model_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
