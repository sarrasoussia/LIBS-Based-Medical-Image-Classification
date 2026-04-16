from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from medmnist import INFO as MEDMNIST_INFO
except Exception:  # pragma: no cover
    MEDMNIST_INFO = {}


def setup_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_seed_dirs(fraction_dir: Path) -> List[Path]:
    if not fraction_dir.exists():
        return []
    dirs = [p for p in fraction_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    return sorted(dirs, key=lambda p: p.name)


def maybe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_dataset_name_from_study_root(study_root: Path) -> str:
    return study_root.name.lower().strip()


def default_class_names_for_dataset(dataset_name: str, n_classes: int) -> List[str]:
    info = MEDMNIST_INFO.get(dataset_name)
    if info:
        labels = info.get("label", {})
        try:
            return [labels[str(i)] for i in range(len(labels))]
        except Exception:
            pass
    return [f"Class {i}" for i in range(n_classes)]


def pad_and_stack(series_list: List[Sequence[float]]) -> np.ndarray:
    if not series_list:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(s) for s in series_list)
    out = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for i, s in enumerate(series_list):
        arr = np.asarray(s, dtype=np.float64)
        out[i, : len(arr)] = arr
    return out


def compute_curve_stats(histories: List[Dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    stacked = pad_and_stack([h[key] for h in histories if key in h])
    if stacked.size == 0:
        return np.array([]), np.array([])
    return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def load_histories_for_fraction(study_root: Path, fraction: float, model_name: str) -> List[Dict]:
    fraction_dir = study_root / f"fraction_{fraction:.2f}"
    histories: List[Dict] = []

    for seed_dir in list_seed_dirs(fraction_dir):
        history_path = seed_dir / f"{model_name}_history.json"
        if history_path.exists():
            histories.append(read_json(history_path))

    return histories


def plot_training_curves(
    left_histories: List[Dict],
    right_histories: List[Dict],
    output_path: Path,
    dataset_name: str,
    left_label: str,
    right_label: str,
) -> None:
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=False)

    colors = {"left": "#2a6fdb", "right": "#f39c12"}
    markers = {"left": "o", "right": "s"}

    for label, histories, color, marker in [
        (left_label, left_histories, colors["left"], markers["left"]),
        (right_label, right_histories, colors["right"], markers["right"]),
    ]:
        train_m, train_s = compute_curve_stats(histories, "train_loss")
        val_m, val_s = compute_curve_stats(histories, "val_loss")

        if train_m.size:
            x = np.arange(1, train_m.size + 1)
            axes[0].plot(x, train_m, color=color, linestyle="-", linewidth=2.0, marker=marker, markersize=3.5,
                         markevery=max(1, train_m.size // 12), label=f"{label} Train")
            axes[0].fill_between(x, train_m - train_s, train_m + train_s, color=color, alpha=0.18)

        if val_m.size:
            xv = np.arange(1, val_m.size + 1)
            axes[0].plot(xv, val_m, color=color, linestyle="--", linewidth=2.0, marker=marker, markersize=3.5,
                         markevery=max(1, val_m.size // 12), label=f"{label} Val")
            axes[0].fill_between(xv, val_m - val_s, val_m + val_s, color=color, alpha=0.12)

    axes[0].set_title("Training/Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(ncol=2, frameon=True)

    for label, histories, color, marker in [
        (left_label, left_histories, colors["left"], markers["left"]),
        (right_label, right_histories, colors["right"], markers["right"]),
    ]:
        train_m, train_s = compute_curve_stats(histories, "train_acc")
        val_m, val_s = compute_curve_stats(histories, "val_acc")

        if train_m.size:
            x = np.arange(1, train_m.size + 1)
            axes[1].plot(x, train_m, color=color, linestyle="-", linewidth=2.0, marker=marker, markersize=3.5,
                         markevery=max(1, train_m.size // 12), label=f"{label} Train")
            axes[1].fill_between(x, train_m - train_s, train_m + train_s, color=color, alpha=0.18)

        if val_m.size:
            xv = np.arange(1, val_m.size + 1)
            axes[1].plot(xv, val_m, color=color, linestyle="--", linewidth=2.0, marker=marker, markersize=3.5,
                         markevery=max(1, val_m.size // 12), label=f"{label} Val")
            axes[1].fill_between(xv, val_m - val_s, val_m + val_s, color=color, alpha=0.12)

    axes[1].set_title("Training/Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(ncol=2, frameon=True)

    fig.suptitle(f"{dataset_name.upper()}: {left_label} vs {right_label} (mean ± std across seeds)", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def aggregate_confusion_matrix(study_root: Path, fraction: float, metric_file: str) -> np.ndarray:
    fraction_dir = study_root / f"fraction_{fraction:.2f}"
    matrices = []

    for seed_dir in list_seed_dirs(fraction_dir):
        path = seed_dir / metric_file
        if not path.exists():
            continue
        cm = np.asarray(read_json(path).get("confusion_matrix"), dtype=np.float64)
        if cm.size:
            matrices.append(cm)

    if not matrices:
        raise FileNotFoundError(f"No confusion matrices found for metric_file='{metric_file}' and fraction={fraction:.2f}")

    return np.sum(np.stack(matrices, axis=0), axis=0)


def normalize_cm_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return (cm / safe) * 100.0


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str, output_path: Path) -> None:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(11.5, 9.2), constrained_layout=False)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)"},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout(pad=1.5)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def gather_fraction_metrics(
    study_root: Path,
    fractions: Sequence[float],
    metric_file: str,
    metric_names: Sequence[str],
) -> Dict[str, Dict[float, List[float]]]:
    out: Dict[str, Dict[float, List[float]]] = {m: {f: [] for f in fractions} for m in metric_names}

    for fraction in fractions:
        fraction_dir = study_root / f"fraction_{fraction:.2f}"
        for seed_dir in list_seed_dirs(fraction_dir):
            path = seed_dir / metric_file
            if not path.exists():
                continue
            data = read_json(path)
            for metric in metric_names:
                if metric in data:
                    out[metric][fraction].append(float(data[metric]))

    return out


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def plot_multi_fraction_comparison(
    left: Dict[str, Dict[float, List[float]]],
    right: Dict[str, Dict[float, List[float]]],
    fractions: Sequence[float],
    output_path: Path,
    dataset_name: str,
    left_label: str,
    right_label: str,
) -> None:
    setup_publication_style()

    metrics = list(left.keys())
    n = len(metrics)
    rows = int(np.ceil(n / 2))
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 4.6 * rows), constrained_layout=False)
    axes = np.array(axes).reshape(-1)

    x = np.array(fractions)
    colors = {"left": "#2a6fdb", "right": "#f39c12"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        l_mean, l_std, r_mean, r_std = [], [], [], []
        for f in fractions:
            lm, ls = mean_std(left[metric][f])
            rm, rs = mean_std(right[metric][f])
            l_mean.append(lm)
            l_std.append(ls)
            r_mean.append(rm)
            r_std.append(rs)

        ax.errorbar(x, l_mean, yerr=l_std, color=colors["left"], marker="o", linewidth=2, capsize=4, label=left_label)
        ax.errorbar(x, r_mean, yerr=r_std, color=colors["right"], marker="s", linewidth=2, capsize=4, label=right_label)

        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Train fraction")
        ax.set_ylabel("Metric value")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
        ax.set_ylim(0.0, 1.02)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)
    fig.suptitle(f"{dataset_name.upper()} Multi-Fraction Performance ({left_label} vs {right_label})", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visuals for DenseNet121 vs GA-DenseNet121.")
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--cm-fraction", type=float, default=1.0)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])
    args = parser.parse_args()

    dataset_name = infer_dataset_name_from_study_root(args.study_root)
    maybe_make_dir(args.output_dir)

    left_model = "densenet121"
    right_model = "ga_densenet121"
    left_label = "DenseNet121"
    right_label = "GA-DenseNet121"

    left_histories = load_histories_for_fraction(args.study_root, args.train_fraction, left_model)
    right_histories = load_histories_for_fraction(args.study_root, args.train_fraction, right_model)
    if not left_histories and not right_histories:
        print(f"Warning: no pretrained history files found in {args.study_root}; creating empty training-curves placeholder.")
    plot_training_curves(
        left_histories,
        right_histories,
        output_path=args.output_dir / "training_curves.png",
        dataset_name=dataset_name,
        left_label=left_label,
        right_label=right_label,
    )

    left_cm = normalize_cm_rows(aggregate_confusion_matrix(args.study_root, args.cm_fraction, "densenet121_metrics.json"))
    right_cm = normalize_cm_rows(aggregate_confusion_matrix(args.study_root, args.cm_fraction, "ga_densenet121_metrics.json"))
    class_names = default_class_names_for_dataset(dataset_name, n_classes=left_cm.shape[0])

    plot_confusion_matrix(
        left_cm,
        class_names,
        title=f"{left_label} Confusion Matrix (fraction={args.cm_fraction:.2f}, normalized %)",
        output_path=args.output_dir / "confusion_matrix_densenet121.png",
    )
    plot_confusion_matrix(
        right_cm,
        class_names,
        title=f"{right_label} Confusion Matrix (fraction={args.cm_fraction:.2f}, normalized %)",
        output_path=args.output_dir / "confusion_matrix_ga_densenet121.png",
    )

    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    left_metrics = gather_fraction_metrics(args.study_root, args.fractions, "densenet121_metrics.json", metric_names)
    right_metrics = gather_fraction_metrics(args.study_root, args.fractions, "ga_densenet121_metrics.json", metric_names)

    plot_multi_fraction_comparison(
        left_metrics,
        right_metrics,
        fractions=args.fractions,
        output_path=args.output_dir / "multi_fraction_comparison.png",
        dataset_name=dataset_name,
        left_label=left_label,
        right_label=right_label,
    )

    print("Saved figures:")
    print(args.output_dir / "training_curves.png")
    print(args.output_dir / "confusion_matrix_densenet121.png")
    print(args.output_dir / "confusion_matrix_ga_densenet121.png")
    print(args.output_dir / "multi_fraction_comparison.png")


if __name__ == "__main__":
    main()
