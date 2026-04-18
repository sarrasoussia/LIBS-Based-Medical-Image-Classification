from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


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


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_pipeline_diagram(save_path: str) -> None:
    """Save a baseline-vs-LIBS pipeline diagram with auxiliary geometric branch."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(13.5, 4.8), dpi=180)
    ax.axis("off")

    # Baseline lane
    baseline_boxes = [
        (0.08, 0.68, "Raw Image"),
        (0.25, 0.68, "Raw Stream"),
        (0.43, 0.68, "Backbone"),
        (0.60, 0.68, "Classifier"),
    ]

    # LIBS lane
    libs_boxes = [
        (0.08, 0.26, "Raw Image"),
        (0.25, 0.38, "Raw Stream"),
        (0.25, 0.14, "Geometric Feature\nStream (Sobel)"),
        (0.43, 0.26, "Learnable Fusion\n(1×1 Conv + BN + ReLU)"),
        (0.60, 0.26, "Backbone"),
        (0.77, 0.26, "Classifier"),
    ]

    def draw_boxes(boxes: list[tuple[float, float, str]], color: str) -> None:
        for x, y, label in boxes:
            rect = plt.Rectangle((x - 0.09, y - 0.08), 0.18, 0.16, fc=color, ec="black", alpha=0.2)
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", fontsize=9)

    def draw_arrows(boxes: list[tuple[float, float, str]]) -> None:
        for i in range(len(boxes) - 1):
            x1, y1, _ = boxes[i]
            x2, y2, _ = boxes[i + 1]
            ax.annotate(
                "",
                xy=(x2 - 0.10, y2),
                xytext=(x1 + 0.10, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )

    draw_boxes(baseline_boxes, "#90caf9")
    draw_boxes(libs_boxes, "#ffcc80")
    draw_arrows(baseline_boxes)
    # custom LIBS arrows for branch merge
    ax.annotate("", xy=(0.16, 0.38), xytext=(0.16, 0.26), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.16, 0.14), xytext=(0.16, 0.26), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.43 - 0.10, 0.30), xytext=(0.25 + 0.10, 0.38), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.43 - 0.10, 0.22), xytext=(0.25 + 0.10, 0.14), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.60 - 0.10, 0.26), xytext=(0.43 + 0.10, 0.26), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.77 - 0.10, 0.26), xytext=(0.60 + 0.10, 0.26), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(0.01, 0.86, "Baseline", fontsize=11, fontweight="bold")
    ax.text(0.01, 0.46, "LIBS Variant", fontsize=11, fontweight="bold")
    ax.text(
        0.50,
        0.04,
        "Geometric features are auxiliary (not a replacement for raw input). Learnable fusion adaptively weights raw and Sobel-based streams.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_study_summary_bars(
    summary: Dict[str, Dict],
    model_keys: Sequence[str],
    metric_names: Sequence[str],
    save_path: str,
) -> None:
    """Plot summary means with 95% CI for each dataset/fraction key."""
    _ensure_dir(save_path)
    keys = list(summary.keys())
    if not keys:
        return

    n_rows = len(keys)
    n_cols = len(metric_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.7 * n_rows), dpi=180)
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r, key in enumerate(keys):
        row = summary[key]
        for c, metric in enumerate(metric_names):
            ax = axes[r, c]
            means = []
            cis = []
            labels = []
            plotted_model_keys = []
            for mk in model_keys:
                stat = row.get(mk, {}).get(metric)
                if not isinstance(stat, dict):
                    continue
                means.append(float(stat.get("mean", 0.0)))
                cis.append(float(stat.get("ci95", 0.0)))
                labels.append(MODEL_DISPLAY_NAME.get(mk, mk))
                plotted_model_keys.append(mk)

            x = np.arange(len(labels))
            bar_colors = [MODEL_COLOR.get(mk, "#808080") for mk in plotted_model_keys]
            ax.bar(x, means, yerr=cis, capsize=4, color=bar_colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{metric} | {key}")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_study_metric_distributions(
    raw_results: List[Dict],
    model_keys: Sequence[str],
    metric_names: Sequence[str],
    save_path: str,
) -> None:
    """Plot per-metric seed distributions for report-ready ablation figures."""
    _ensure_dir(save_path)
    if not raw_results:
        return

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.0 * len(metric_names), 3.6), dpi=180)
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = []
        labels = []
        for mk in model_keys:
            vals = [float(r.get(mk, {}).get(metric, np.nan)) for r in raw_results]
            vals = [v for v in vals if np.isfinite(v)]
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(MODEL_DISPLAY_NAME.get(mk, mk))

        if data:
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
