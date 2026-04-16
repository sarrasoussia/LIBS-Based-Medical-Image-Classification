from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

try:
    from medmnist import INFO as MEDMNIST_INFO
except Exception:  # pragma: no cover
    MEDMNIST_INFO = {}


# ------------------------------ Styling ------------------------------------ #
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


# ------------------------------ IO helpers --------------------------------- #
def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first_existing(directory: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def list_seed_dirs(fraction_dir: Path) -> List[Path]:
    if not fraction_dir.exists():
        return []
    dirs = [p for p in fraction_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    return sorted(dirs, key=lambda p: p.name)


def pad_and_stack(series_list: List[Sequence[float]]) -> np.ndarray:
    if not series_list:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(s) for s in series_list)
    out = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for i, s in enumerate(series_list):
        arr = np.asarray(s, dtype=np.float64)
        out[i, : len(arr)] = arr
    return out


def maybe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ------------------------- Training curve plotting -------------------------- #
@dataclass
class CurveStats:
    mean: np.ndarray
    std: np.ndarray


def compute_curve_stats(histories: List[Dict], key: str) -> CurveStats:
    stacked = pad_and_stack([h[key] for h in histories if key in h])
    if stacked.size == 0:
        return CurveStats(mean=np.array([]), std=np.array([]))
    return CurveStats(
        mean=np.nanmean(stacked, axis=0),
        std=np.nanstd(stacked, axis=0),
    )


def load_histories_for_fraction(study_root: Path, fraction: float, model: str) -> List[Dict]:
    fraction_dir = study_root / f"fraction_{fraction:.2f}"
    histories: List[Dict] = []

    for seed_dir in list_seed_dirs(fraction_dir):
        if model == "baseline":
            history_path = find_first_existing(seed_dir, ["baseline_history.json", "baseline_cnn_history.json"])
        else:
            history_path = find_first_existing(seed_dir, ["ga_history.json", "ga_cnn_history.json"])

        if history_path is None:
            continue
        histories.append(read_json(history_path))

    return histories


def plot_training_curves(
    baseline_histories: List[Dict],
    ga_histories: List[Dict],
    output_path: Path,
    dataset_name: str = "dataset",
) -> None:
    setup_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=False)

    colors = {"baseline": "#1f77b4", "ga": "#ff7f0e"}
    markers = {"baseline": "o", "ga": "s"}

    # Loss subplot
    for model_name, histories, color, marker in [
        ("Baseline CNN", baseline_histories, colors["baseline"], markers["baseline"]),
        ("GA-CNN", ga_histories, colors["ga"], markers["ga"]),
    ]:
        train = compute_curve_stats(histories, "train_loss")
        val = compute_curve_stats(histories, "val_loss")

        if train.mean.size:
            x = np.arange(1, train.mean.size + 1)
            axes[0].plot(
                x,
                train.mean,
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker=marker,
                markersize=3.5,
                markevery=max(1, train.mean.size // 12),
                label=f"{model_name} Train",
            )
            axes[0].fill_between(x, train.mean - train.std, train.mean + train.std, color=color, alpha=0.18)

        if val.mean.size:
            xv = np.arange(1, val.mean.size + 1)
            axes[0].plot(
                xv,
                val.mean,
                color=color,
                linestyle="--",
                linewidth=2.0,
                marker=marker,
                markersize=3.5,
                markevery=max(1, val.mean.size // 12),
                label=f"{model_name} Val",
            )
            axes[0].fill_between(xv, val.mean - val.std, val.mean + val.std, color=color, alpha=0.12)

    axes[0].set_title("Training/Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(ncol=2, frameon=True)

    # Accuracy subplot
    for model_name, histories, color, marker in [
        ("Baseline CNN", baseline_histories, colors["baseline"], markers["baseline"]),
        ("GA-CNN", ga_histories, colors["ga"], markers["ga"]),
    ]:
        train = compute_curve_stats(histories, "train_acc")
        val = compute_curve_stats(histories, "val_acc")

        if train.mean.size:
            x = np.arange(1, train.mean.size + 1)
            axes[1].plot(
                x,
                train.mean,
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker=marker,
                markersize=3.5,
                markevery=max(1, train.mean.size // 12),
                label=f"{model_name} Train",
            )
            axes[1].fill_between(x, train.mean - train.std, train.mean + train.std, color=color, alpha=0.18)

        if val.mean.size:
            xv = np.arange(1, val.mean.size + 1)
            axes[1].plot(
                xv,
                val.mean,
                color=color,
                linestyle="--",
                linewidth=2.0,
                marker=marker,
                markersize=3.5,
                markevery=max(1, val.mean.size // 12),
                label=f"{model_name} Val",
            )
            axes[1].fill_between(xv, val.mean - val.std, val.mean + val.std, color=color, alpha=0.12)

    axes[1].set_title("Training/Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(ncol=2, frameon=True)

    fig.suptitle(f"{dataset_name.upper()}: Baseline CNN vs GA-CNN (mean ± std across seeds)", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------ Confusion matrix plotting ------------------------- #
def aggregate_confusion_matrix(study_root: Path, fraction: float, model: str) -> np.ndarray:
    fraction_dir = study_root / f"fraction_{fraction:.2f}"
    matrices = []
    metric_name = "baseline_metrics.json" if model == "baseline" else "ga_metrics.json"

    for seed_dir in list_seed_dirs(fraction_dir):
        path = seed_dir / metric_name
        if not path.exists():
            continue
        cm = np.asarray(read_json(path).get("confusion_matrix"), dtype=np.float64)
        if cm.size:
            matrices.append(cm)

    if not matrices:
        raise FileNotFoundError(f"No confusion matrices found for model='{model}' and fraction={fraction:.2f}")

    return np.sum(np.stack(matrices, axis=0), axis=0)


def normalize_cm_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return (cm / safe) * 100.0


def default_pathmnist_labels() -> List[str]:
    # medmnist.INFO['pathmnist']['label'] order
    return [
        "Adipose",
        "Background",
        "Debris",
        "Lymphocytes",
        "Mucus",
        "Smooth Muscle",
        "Normal Colon Mucosa",
        "Cancer-Associated Stroma",
        "Colorectal Adenocarcinoma Epithelium",
    ]


def infer_dataset_name_from_study_root(study_root: Path) -> str:
    return study_root.name.lower().strip()


def default_class_names_for_dataset(dataset_name: str) -> List[str]:
    if dataset_name == "pathmnist":
        return default_pathmnist_labels()

    info = MEDMNIST_INFO.get(dataset_name)
    if not info:
        return []

    labels = info.get("label", {})
    try:
        return [labels[str(i)] for i in range(len(labels))]
    except Exception:
        return [str(v) for _, v in sorted(labels.items(), key=lambda kv: kv[0])]


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


# --------------------------- Grad-CAM plotting ----------------------------- #
def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        arr = np.asarray(Image.open(path))
    elif suffix == ".json":
        arr = np.asarray(read_json(path))
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return arr


def to_float_image(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32)
    if out.max() > 1.0:
        out /= 255.0
    return np.clip(out, 0.0, 1.0)


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] in {3, 4}:
        x = x[..., :3]
        return x.astype(np.float32).mean(axis=-1)
    if x.ndim == 3 and x.shape[0] in {1, 3}:
        x = np.transpose(x, (1, 2, 0))
        if x.shape[-1] == 1:
            return x[..., 0].astype(np.float32)
        return x.astype(np.float32).mean(axis=-1)
    raise ValueError(f"Cannot interpret array shape {x.shape} as grayscale heatmap")


def resize_like(img: np.ndarray, target_h: int, target_w: int, mode: str = "bilinear") -> np.ndarray:
    pil_mode = Image.BILINEAR if mode == "bilinear" else Image.NEAREST
    if img.ndim == 2:
        pil = Image.fromarray((to_float_image(img) * 255).astype(np.uint8), mode="L")
        return np.asarray(pil.resize((target_w, target_h), pil_mode)).astype(np.float32) / 255.0

    if img.ndim == 3:
        pil = Image.fromarray((to_float_image(img) * 255).astype(np.uint8))
        return np.asarray(pil.resize((target_w, target_h), pil_mode)).astype(np.float32) / 255.0

    raise ValueError(f"Unsupported image shape for resize: {img.shape}")


def parse_gradcam_samples(
    gradcam_json: Path | None,
    image_path: Path | None,
    baseline_cam_path: Path | None,
    ga_cam_path: Path | None,
) -> List[Dict[str, np.ndarray]]:
    samples: List[Dict[str, np.ndarray]] = []

    # Single-sample mode from direct file paths
    if image_path and baseline_cam_path and ga_cam_path:
        samples.append(
            {
                "image": load_array(image_path),
                "baseline_cam": load_array(baseline_cam_path),
                "ga_cam": load_array(ga_cam_path),
            }
        )

    # JSON mode
    if gradcam_json is not None and gradcam_json.exists():
        payload = read_json(gradcam_json)
        records = payload.get("samples", payload)
        if isinstance(records, dict):
            records = [records]

        for record in records:
            # Accept either embedded arrays or file paths.
            image = record.get("image")
            baseline_cam = record.get("baseline_cam")
            ga_cam = record.get("ga_cam")

            if image is None and record.get("image_path"):
                image = load_array(Path(record["image_path"]))
            if baseline_cam is None and record.get("baseline_cam_path"):
                baseline_cam = load_array(Path(record["baseline_cam_path"]))
            if ga_cam is None and record.get("ga_cam_path"):
                ga_cam = load_array(Path(record["ga_cam_path"]))

            if image is None or baseline_cam is None or ga_cam is None:
                continue

            samples.append(
                {
                    "image": np.asarray(image),
                    "baseline_cam": np.asarray(baseline_cam),
                    "ga_cam": np.asarray(ga_cam),
                }
            )

    return samples


def plot_gradcam_comparison(
    samples: List[Dict[str, np.ndarray]],
    output_path: Path,
    alpha: float = 0.45,
    dataset_name: str = "dataset",
) -> None:
    setup_publication_style()

    if not samples:
        fig, ax = plt.subplots(figsize=(8, 2.2), constrained_layout=True)
        ax.text(
            0.5,
            0.5,
            "No Grad-CAM inputs found. Provide --gradcam-json or\n"
            "(--gradcam-image, --baseline-cam, --ga-cam).",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(9.0, max(3.0, 3.1 * rows)), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    for i, sample in enumerate(samples):
        img = sample["image"]
        baseline_cam = sample["baseline_cam"]
        ga_cam = sample["ga_cam"]

        # Ensure RGB image
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[0] in {1, 3}:
            img_rgb = np.transpose(img, (1, 2, 0)) if img.shape[0] == 3 else np.repeat(np.transpose(img, (1, 2, 0)), 3, axis=2)
        else:
            img_rgb = img[..., :3] if img.ndim == 3 else img

        img_rgb = to_float_image(img_rgb)
        h, w = img_rgb.shape[:2]

        b_cam = resize_like(to_grayscale(baseline_cam), h, w)
        g_cam = resize_like(to_grayscale(ga_cam), h, w)

        for ax, cam, title in [
            (axes[i, 0], b_cam, "Baseline CNN"),
            (axes[i, 1], g_cam, "GA-CNN"),
        ]:
            ax.imshow(img_rgb)
            ax.imshow(cam, cmap="jet", alpha=alpha)
            ax.set_title(title)
            ax.axis("off")

    fig.suptitle(f"Grad-CAM Overlays on {dataset_name.upper()} Samples", y=1.01)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# --------------------- Multi-fraction performance plot ---------------------- #
def gather_fraction_metrics(
    study_root: Path,
    fractions: Sequence[float],
    model: str,
    metric_names: Sequence[str],
) -> Dict[str, Dict[float, List[float]]]:
    metric_file = "baseline_metrics.json" if model == "baseline" else "ga_metrics.json"
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
    baseline: Dict[str, Dict[float, List[float]]],
    ga: Dict[str, Dict[float, List[float]]],
    fractions: Sequence[float],
    output_path: Path,
    dataset_name: str = "dataset",
) -> None:
    setup_publication_style()

    metrics = list(baseline.keys())
    n = len(metrics)
    rows = int(np.ceil(n / 2))
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 4.6 * rows), constrained_layout=False)
    axes = np.array(axes).reshape(-1)

    x = np.array(fractions)
    colors = {"baseline": "#1f77b4", "ga": "#ff7f0e"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        b_mean, b_std, g_mean, g_std = [], [], [], []
        for f in fractions:
            bm, bs = mean_std(baseline[metric][f])
            gm, gs = mean_std(ga[metric][f])
            b_mean.append(bm)
            b_std.append(bs)
            g_mean.append(gm)
            g_std.append(gs)

        ax.errorbar(
            x,
            b_mean,
            yerr=b_std,
            color=colors["baseline"],
            marker="o",
            linewidth=2,
            capsize=4,
            label="Baseline CNN",
        )
        ax.errorbar(
            x,
            g_mean,
            yerr=g_std,
            color=colors["ga"],
            marker="s",
            linewidth=2,
            capsize=4,
            label="GA-CNN",
        )

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
    fig.suptitle(f"{dataset_name.upper()} Multi-Fraction Performance (mean ± std across seeds)", y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------- Main ------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready figures for Baseline CNN vs GA-CNN on PathMNIST.")

    parser.add_argument("--study-root", type=Path, default=Path("experiments/results/study/pathmnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/figures/pathmnist"))

    parser.add_argument("--train-fraction", type=float, default=1.0, help="Fraction used for training-curve figure.")
    parser.add_argument("--cm-fraction", type=float, default=1.0, help="Fraction used for confusion matrices.")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.00])

    parser.add_argument("--class-names", type=str, default="", help="Comma-separated class labels. Defaults to PathMNIST labels.")

    parser.add_argument("--gradcam-json", type=Path, default=None)
    parser.add_argument("--gradcam-image", type=Path, default=None)
    parser.add_argument("--baseline-cam", type=Path, default=None)
    parser.add_argument("--ga-cam", type=Path, default=None)
    parser.add_argument("--gradcam-alpha", type=float, default=0.45)

    args = parser.parse_args()
    dataset_name = infer_dataset_name_from_study_root(args.study_root)

    maybe_make_dir(args.output_dir)

    # 1) Training curves
    baseline_histories = load_histories_for_fraction(args.study_root, args.train_fraction, model="baseline")
    ga_histories = load_histories_for_fraction(args.study_root, args.train_fraction, model="ga")
    plot_training_curves(
        baseline_histories,
        ga_histories,
        output_path=args.output_dir / "training_curves.png",
        dataset_name=dataset_name,
    )

    # 2) Confusion matrices (normalized percentages)
    class_names = (
        [c.strip() for c in args.class_names.split(",") if c.strip()]
        if args.class_names
        else default_class_names_for_dataset(dataset_name)
    )

    baseline_cm = normalize_cm_rows(aggregate_confusion_matrix(args.study_root, args.cm_fraction, model="baseline"))
    ga_cm = normalize_cm_rows(aggregate_confusion_matrix(args.study_root, args.cm_fraction, model="ga"))
    if not class_names:
        class_names = [f"Class {i}" for i in range(baseline_cm.shape[0])]

    plot_confusion_matrix(
        baseline_cm,
        class_names,
        title=f"Baseline CNN Confusion Matrix (fraction={args.cm_fraction:.2f}, normalized %)",
        output_path=args.output_dir / "confusion_matrix_baseline.png",
    )
    plot_confusion_matrix(
        ga_cm,
        class_names,
        title=f"GA-CNN Confusion Matrix (fraction={args.cm_fraction:.2f}, normalized %)",
        output_path=args.output_dir / "confusion_matrix_ga_cnn.png",
    )

    # 3) Grad-CAM comparison
    samples = parse_gradcam_samples(
        gradcam_json=args.gradcam_json,
        image_path=args.gradcam_image,
        baseline_cam_path=args.baseline_cam,
        ga_cam_path=args.ga_cam,
    )
    plot_gradcam_comparison(
        samples,
        output_path=args.output_dir / "gradcam_comparison.png",
        alpha=args.gradcam_alpha,
        dataset_name=dataset_name,
    )

    # 4) Multi-fraction performance
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    baseline_metrics = gather_fraction_metrics(args.study_root, args.fractions, "baseline", metric_names)
    ga_metrics = gather_fraction_metrics(args.study_root, args.fractions, "ga", metric_names)
    plot_multi_fraction_comparison(
        baseline_metrics,
        ga_metrics,
        fractions=args.fractions,
        output_path=args.output_dir / "multi_fraction_comparison.png",
        dataset_name=dataset_name,
    )

    print("Saved figures:")
    print(args.output_dir / "training_curves.png")
    print(args.output_dir / "confusion_matrix_baseline.png")
    print(args.output_dir / "confusion_matrix_ga_cnn.png")
    print(args.output_dir / "gradcam_comparison.png")
    print(args.output_dir / "multi_fraction_comparison.png")


if __name__ == "__main__":
    main()
