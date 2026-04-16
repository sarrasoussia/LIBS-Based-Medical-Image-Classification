from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix

from analysis.ablation import ABLATION_SETTINGS
from data.pathmnist import get_medmnist_dataloaders, labels_to_long, set_global_seed
from models.factory import build_model
from models.ga_representation import GARepresentation
from training.evaluate import evaluate_model
from utils.statistics import cohens_d_paired, paired_t_test, summarize_metric, wilcoxon_signed_rank_test


DEFAULT_DATASETS = ["pathmnist", "bloodmnist", "dermamnist"]
DEFAULT_FAMILIES = ["cnn", "densenet121"]
MODEL_PAIR_BY_FAMILY = {
    "cnn": ("baseline", "ga_cnn"),
    "densenet121": ("densenet121", "ga_densenet121"),
}


def _checkpoint_model_name(study_name: str, model_key: str) -> str:
    if study_name == "study" and model_key == "baseline":
        return "baseline_cnn"
    if study_name == "study" and model_key == "ga_cnn":
        return "ga_cnn"
    return model_key


def _device_from_torch() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _study_root(results_root: Path, study_name: str) -> Path:
    return results_root / study_name


def _load_raw_results(results_root: Path, study_name: str) -> List[Dict[str, object]]:
    path = _study_root(results_root, study_name) / "raw_results.json"
    if not path.exists():
        return []
    data = _load_json(path)
    return data if isinstance(data, list) else []


def _load_summary(results_root: Path, study_name: str) -> Dict[str, object]:
    path = _study_root(results_root, study_name) / "summary_statistics.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    return data if isinstance(data, dict) else {}


def _collect_grouped_results(raw_results: Sequence[Dict[str, object]]) -> Dict[Tuple[str, float], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for row in raw_results:
        key = (str(row["dataset"]), float(row["train_fraction"]))
        grouped[key].append(row)
    return grouped


def _metric_values(rows: Sequence[Dict[str, object]], model_key: str, metric: str) -> List[float]:
    values = []
    for row in rows:
        model_block = row.get(model_key)
        if isinstance(model_block, dict) and metric in model_block:
            values.append(float(model_block[metric]))
    return values


def _full_run_dir(results_root: Path, study_name: str, dataset: str, fraction: float, seed: int) -> Path:
    return results_root / study_name / dataset / f"fraction_{fraction:.2f}" / f"seed_{seed}"


def _resolve_model_config(
    cfg: Dict,
    family: str,
) -> Dict[str, object]:
    include_higher_order = bool(cfg.get("ga", {}).get("include_higher_order", True))
    representation_mode = cfg.get("ga", {}).get("representation_mode")
    pretrained = bool(cfg.get("model", {}).get("pretrained", True))
    adapt_for_small_inputs = bool(cfg.get("model", {}).get("adapt_for_small_inputs", True))
    trainable_backbone = bool(cfg.get("model", {}).get("trainable_backbone", True))

    if family == "cnn":
        return {
            "include_higher_order": include_higher_order,
            "representation_mode": representation_mode,
            "pretrained": False,
            "adapt_for_small_inputs": adapt_for_small_inputs,
            "trainable_backbone": True,
        }
    if family == "densenet121":
        return {
            "include_higher_order": include_higher_order,
            "representation_mode": representation_mode,
            "pretrained": pretrained,
            "adapt_for_small_inputs": adapt_for_small_inputs,
            "trainable_backbone": trainable_backbone,
        }
    raise ValueError(f"Unsupported family '{family}'.")


def _build_model_for_checkpoint(
    cfg: Dict,
    family: str,
    model_name: str,
    in_channels: int,
    num_classes: int,
) -> torch.nn.Module:
    model_cfg = _resolve_model_config(cfg, family)
    return build_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        include_higher_order=bool(model_cfg["include_higher_order"]),
        representation_mode=model_cfg["representation_mode"],
        pretrained=bool(model_cfg["pretrained"]),
        adapt_for_small_inputs=bool(model_cfg["adapt_for_small_inputs"]),
        trainable_backbone=bool(model_cfg["trainable_backbone"]),
    )


@torch.no_grad()
def _load_or_create_predictions(
    cfg: Dict,
    device: torch.device,
    results_root: Path,
    study_name: str,
    dataset: str,
    fraction: float,
    seed: int,
    model_name: str,
) -> Path:
    run_dir = _full_run_dir(results_root, study_name, dataset, fraction, seed)
    checkpoint_model_name = _checkpoint_model_name(study_name, model_name)
    prediction_path = run_dir / f"{checkpoint_model_name}_predictions.json"
    if prediction_path.exists():
        return prediction_path

    loaders = get_medmnist_dataloaders(
        dataset_name=dataset,
        data_dir=cfg["paths"]["data_dir"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 0)),
        normalize=True,
        seed=int(seed),
        train_fraction=float(fraction),
    )

    family = "densenet121" if "densenet" in model_name else "cnn"
    checkpoint_model_name = _checkpoint_model_name(study_name, model_name)
    model = _build_model_for_checkpoint(
        cfg=cfg,
        family=family,
        model_name=checkpoint_model_name,
        in_channels=loaders.in_channels,
        num_classes=loaders.num_classes,
    ).to(device)

    checkpoint_path = run_dir / f"{checkpoint_model_name}_best.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    metrics_path = run_dir / f"{checkpoint_model_name}_metrics.json"
    evaluate_model(
        model=model,
        loader=loaders.test,
        device=device,
        save_path=str(metrics_path),
        save_predictions=True,
    )
    return prediction_path


def _load_predictions(path: Path) -> Dict[str, np.ndarray]:
    data = _load_json(path)
    return {
        "y_true": np.asarray(data["y_true"], dtype=np.int64),
        "y_pred": np.asarray(data["y_pred"], dtype=np.int64),
        "y_conf": np.asarray(data["y_conf"], dtype=np.float64),
    }


def _calibration_curve(y_true: np.ndarray, y_pred: np.ndarray, y_conf: np.ndarray, n_bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for idx in range(n_bins):
        lo, hi = bin_edges[idx], bin_edges[idx + 1]
        if idx == n_bins - 1:
            mask = (y_conf >= lo) & (y_conf <= hi)
        else:
            mask = (y_conf >= lo) & (y_conf < hi)
        if not np.any(mask):
            bins.append((float((lo + hi) / 2.0), np.nan, np.nan, 0))
            continue
        acc = float(np.mean((y_pred[mask] == y_true[mask]).astype(np.float64)))
        conf = float(np.mean(y_conf[mask]))
        bins.append((float((lo + hi) / 2.0), acc, conf, int(mask.sum())))
    return bins


def _plot_reliability_and_confidence(
    predictions_by_model: Dict[str, Dict[str, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for model_name, preds in predictions_by_model.items():
        bins = _calibration_curve(preds["y_true"], preds["y_pred"], preds["y_conf"], n_bins=15)
        xs = [item[0] for item in bins if not np.isnan(item[1])]
        accs = [item[1] for item in bins if not np.isnan(item[1])]
        ax.plot(xs, accs, marker="o", linewidth=1.8, label=model_name)
    ax.set_title("Reliability diagram")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    ax = axes[1]
    bins = np.linspace(0.0, 1.0, 16)
    for model_name, preds in predictions_by_model.items():
        ax.hist(preds["y_conf"], bins=bins, alpha=0.45, density=True, label=model_name)
    ax.set_title("Confidence histogram")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _aggregate_history(histories: Sequence[Dict[str, object]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    series = [np.asarray(hist.get(key, []), dtype=np.float64) for hist in histories if hist.get(key)]
    if not series:
        return np.asarray([]), np.asarray([])
    min_length = min(len(item) for item in series)
    stacked = np.stack([item[:min_length] for item in series], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _plot_training_diagnostics(
    histories_by_model: Dict[str, List[Dict[str, object]]],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)
    for model_name, histories in histories_by_model.items():
        if not histories:
            continue
        mean_train_loss, std_train_loss = _aggregate_history(histories, "train_loss")
        mean_val_loss, std_val_loss = _aggregate_history(histories, "val_loss")
        mean_train_acc, std_train_acc = _aggregate_history(histories, "train_acc")
        mean_val_acc, std_val_acc = _aggregate_history(histories, "val_acc")
        mean_grad_norm, std_grad_norm = _aggregate_history(histories, "train_grad_norm")

        epochs = np.arange(1, len(mean_train_loss) + 1)
        axes[0].plot(epochs, mean_train_loss, label=model_name)
        axes[0].fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.15)
        axes[0].plot(epochs, mean_val_loss, linestyle="--")
        axes[0].fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.10)

        axes[1].plot(epochs, mean_train_acc, label=model_name)
        axes[1].fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.15)
        axes[1].plot(epochs, mean_val_acc, linestyle="--")
        axes[1].fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.10)

        if mean_grad_norm.size > 0:
            axes[2].plot(np.arange(1, len(mean_grad_norm) + 1), mean_grad_norm, label=model_name)
            axes[2].fill_between(
                np.arange(1, len(mean_grad_norm) + 1),
                mean_grad_norm - std_grad_norm,
                mean_grad_norm + std_grad_norm,
                alpha=0.15,
            )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 norm")
    if not axes[2].lines:
        axes[2].text(0.5, 0.5, "Gradient norms unavailable in legacy runs", ha="center", va="center")
    else:
        axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _get_input_batch(
    cfg: Dict,
    dataset_name: str,
    seed: int = 42,
    batch_size: int = 256,
    num_batches: int = 4,
):
    loaders = get_medmnist_dataloaders(
        dataset_name=dataset_name,
        data_dir=cfg["paths"]["data_dir"],
        batch_size=batch_size,
        num_workers=0,
        normalize=True,
        seed=seed,
        train_fraction=1.0,
    )
    batches = []
    for batch_idx, (images, labels) in enumerate(loaders.test):
        batches.append(images)
        if batch_idx + 1 >= num_batches:
            break
    return torch.cat(batches, dim=0), loaders.class_names


def _apply_representation(x: torch.Tensor, representation_mode: str) -> torch.Tensor:
    if representation_mode == "rgb_only":
        return x
    encoder = GARepresentation(
        in_channels=x.shape[1],
        include_higher_order=(representation_mode == "full"),
        representation_mode=representation_mode,
    )
    return encoder(x)


def _channel_stats(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = x.detach().cpu().numpy()
    channel_mean = flat.mean(axis=(0, 2, 3))
    channel_var = flat.var(axis=(0, 2, 3))
    channel_flat = flat.transpose(1, 0, 2, 3).reshape(flat.shape[1], -1)
    corr = np.corrcoef(channel_flat)
    return channel_mean, channel_var, corr


def _plot_distribution_analysis(
    cfg: Dict,
    datasets: Sequence[str],
    output_dir: Path,
) -> None:
    for dataset_name in datasets:
        x, _ = _get_input_batch(cfg, dataset_name)
        representations = {
            "RGB": x,
            "Sobel XY": _apply_representation(x, "sobel_xy"),
            "Magnitude": _apply_representation(x, "magnitude"),
            "Full GA": _apply_representation(x, "full"),
        }

        fig, axes = plt.subplots(len(representations), 3, figsize=(16, 3.4 * len(representations)), dpi=180)
        if len(representations) == 1:
            axes = np.array([axes])

        full_ga_magnitude = None
        for row_idx, (name, tensor) in enumerate(representations.items()):
            mean, var, corr = _channel_stats(tensor)
            ax = axes[row_idx, 0]
            ax.bar(np.arange(len(mean)), mean, color="#4c72b0")
            ax.set_title(f"{name} - channel mean")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Mean")

            ax = axes[row_idx, 1]
            ax.bar(np.arange(len(var)), var, color="#55a868")
            ax.set_title(f"{name} - channel variance")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Variance")

            ax = axes[row_idx, 2]
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_title(f"{name} - channel correlation")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Channel")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if name == "Full GA":
                # The magnitude channels live in the middle third of the 5C encoding.
                c = x.shape[1]
                full_ga_magnitude = tensor[:, 3 * c : 4 * c].detach().cpu().numpy().ravel()

        fig.suptitle(f"{dataset_name} input distribution analysis")
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_distribution_summary.png", bbox_inches="tight")
        plt.close(fig)

        if full_ga_magnitude is not None:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
            ax.hist(full_ga_magnitude, bins=50, color="#c44e52", alpha=0.8, density=True)
            ax.set_title(f"{dataset_name} - full GA gradient magnitude histogram")
            ax.set_xlabel("Magnitude")
            ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(output_dir / f"{dataset_name}_gradient_magnitude_hist.png", bbox_inches="tight")
            plt.close(fig)


def _class_recall_from_cm(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / cm.sum(axis=1)
    recall = np.nan_to_num(recall, nan=0.0)
    return recall


def _plot_failure_cases(
    cm_by_model: Dict[str, np.ndarray],
    class_names: Sequence[str],
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    x = np.arange(len(class_names))
    width = 0.35 if len(cm_by_model) == 2 else 0.25
    for idx, (model_name, cm) in enumerate(cm_by_model.items()):
        recall = _class_recall_from_cm(cm)
        ax.bar(x + (idx - (len(cm_by_model) - 1) / 2) * width, recall, width=width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-class recall")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _top_misclassifications(cm: np.ndarray, class_names: Sequence[str], top_k: int = 5) -> List[Dict[str, object]]:
    miscls = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            miscls.append(
                {
                    "true_class": class_names[true_idx],
                    "pred_class": class_names[pred_idx],
                    "count": count,
                }
            )
    miscls.sort(key=lambda row: row["count"], reverse=True)
    return miscls[:top_k]


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_metrics_table(
    raw_results: List[Dict[str, object]],
    output_dir: Path,
    table_name: str,
) -> None:
    rows = []
    grouped = _collect_grouped_results(raw_results)
    for (dataset, fraction), group_rows in grouped.items():
        for pair_name, pair_key in [("cnn", ("baseline", "ga_cnn")), ("densenet121", ("densenet121", "ga_densenet121"))]:
            baseline_vals_accuracy = _metric_values(group_rows, pair_key[0], "accuracy")
            challenger_vals_accuracy = _metric_values(group_rows, pair_key[1], "accuracy")
            baseline_vals_f1 = _metric_values(group_rows, pair_key[0], "f1_macro")
            challenger_vals_f1 = _metric_values(group_rows, pair_key[1], "f1_macro")
            if not baseline_vals_accuracy or not challenger_vals_accuracy:
                continue
            if not baseline_vals_f1 or not challenger_vals_f1:
                continue
            for metric, base_vals, challenger_vals in [
                ("accuracy", baseline_vals_accuracy, challenger_vals_accuracy),
                ("f1_macro", baseline_vals_f1, challenger_vals_f1),
            ]:
                t_test = paired_t_test(base_vals, challenger_vals)
                w_test = wilcoxon_signed_rank_test(base_vals, challenger_vals)
                d_val = cohens_d_paired(base_vals, challenger_vals)
                base_summary = summarize_metric(base_vals)
                challenger_summary = summarize_metric(challenger_vals)
                rows.append(
                    {
                        "dataset": dataset,
                        "fraction": fraction,
                        "family": pair_name,
                        "metric": metric,
                        "baseline_mean": base_summary["mean"],
                        "baseline_ci95_low": base_summary["ci95_low"],
                        "baseline_ci95_high": base_summary["ci95_high"],
                        "challenger_mean": challenger_summary["mean"],
                        "challenger_ci95_low": challenger_summary["ci95_low"],
                        "challenger_ci95_high": challenger_summary["ci95_high"],
                        "paired_t_pvalue": t_test["pvalue"],
                        "wilcoxon_pvalue": w_test["pvalue"],
                        "cohens_d": d_val,
                        "significant": "*" if min(t_test["pvalue"], w_test["pvalue"]) < 0.05 else "",
                    }
                )

    _write_csv(output_dir / f"{table_name}.csv", rows, list(rows[0].keys()) if rows else [])

    tex_lines = [
        r"\begin{tabular}{lllrcccccc}",
        r"\hline",
        r"Dataset & Family & Metric & Baseline & Challenger & $p_{t}$ & $p_{W}$ & $d$ & Sig \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['dataset']} & {row['family']} & {row['metric']} & "
            f"{row['baseline_mean']:.4f} [{row['baseline_ci95_low']:.4f}, {row['baseline_ci95_high']:.4f}] & "
            f"{row['challenger_mean']:.4f} [{row['challenger_ci95_low']:.4f}, {row['challenger_ci95_high']:.4f}] & "
            f"{row['paired_t_pvalue']:.4f} & {row['wilcoxon_pvalue']:.4f} & {row['cohens_d']:.3f} & {row['significant']} \\")
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    with open(output_dir / f"{table_name}.tex", "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex_lines))


def _make_final_summary(summary: Dict[str, object], output_dir: Path) -> None:
    lines = ["# Thesis-Style Summary", ""]
    # Use fraction 1.00 where the strongest signal is easiest to interpret.
    for dataset in DEFAULT_DATASETS:
        key = f"{dataset}|fraction=1.00"
        if key not in summary:
            continue
        block = summary[key]
        lines.append(f"## {dataset}")
        if "baseline" in block and "ga_cnn" in block:
            base = block["baseline"]
            ga = block["ga_cnn"]
            lines.append(
                f"- GA-CNN changes accuracy from {base['accuracy']['mean']:.4f} to {ga['accuracy']['mean']:.4f} and macro-F1 from {base['f1_macro']['mean']:.4f} to {ga['f1_macro']['mean']:.4f}."
            )
        if "densenet121" in block and "ga_densenet121" in block:
            base = block["densenet121"]
            ga = block["ga_densenet121"]
            lines.append(
                f"- DenseNet121 reaches {base['accuracy']['mean']:.4f} accuracy with ECE {base['ece']['mean']:.4f}, while the GA-DenseNet variant reaches {ga['accuracy']['mean']:.4f} accuracy with ECE {ga['ece']['mean']:.4f}."
            )
        if dataset == "pathmnist":
            lines.append("- GA is beneficial here: the Sobel-augmented representation slightly improves the small CNN and the full DenseNet variant improves macro-F1.")
        elif dataset == "bloodmnist":
            lines.append("- GA is mostly harmful or neutral here: the CNN baseline is stronger, and both DenseNet variants collapse to a near-chance classifier with very high calibration error.")
        elif dataset == "dermamnist":
            lines.append("- GA is mixed here: it can improve recall and F1 marginally in some fractions, but it does not consistently improve accuracy and the DenseNet family remains collapsed.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "- GA helps when Sobel-derived edge information complements the backbone without overly distorting the input distribution.",
            "- GA hurts when the added channels amplify distribution shift or when the backbone is sensitive to small-image preprocessing mismatches.",
            "- DenseNet behaves differently from the simple CNN because it is pretrained, deeper, and more sensitive to the statistical profile of the input stem; on BloodMNIST and DermaMNIST the model shows degenerate predictions and extreme ECE, consistent with pretrained-feature mismatch rather than a lack of capacity.",
        ]
    )
    with open(output_dir / "thesis_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _generate_calibration_and_failure_analysis(
    cfg: Dict,
    device: torch.device,
    results_root: Path,
    analysis_dir: Path,
    figures_dir: Path,
    raw_results: List[Dict[str, object]],
) -> None:
    grouped = _collect_grouped_results(raw_results)
    calibration_rows = []
    failure_rows = []
    misclassification_rows = []

    for (dataset, fraction), rows in grouped.items():
        if abs(fraction - 1.0) > 1e-9:
            continue
        class_names = get_medmnist_dataloaders(
            dataset_name=dataset,
            data_dir=cfg["paths"]["data_dir"],
            batch_size=32,
            num_workers=0,
            normalize=True,
            seed=42,
            train_fraction=1.0,
        ).class_names

        for family, (baseline_key, ga_key) in MODEL_PAIR_BY_FAMILY.items():
            predictions = {}
            cm_by_model = {}
            for model_key in [baseline_key, ga_key]:
                seed_preds = []
                cm_sum = None
                for row in rows:
                    if model_key not in row:
                        continue
                    seed = int(row["seed"])
                    study_name = "study" if family == "cnn" else "study_pretrained"
                    pred_path = _load_or_create_predictions(
                        cfg=cfg,
                        device=device,
                        results_root=results_root,
                        study_name=study_name,
                        dataset=dataset,
                        fraction=fraction,
                        seed=seed,
                        model_name=_checkpoint_model_name(study_name, model_key),
                    )
                    preds = _load_predictions(pred_path)
                    seed_preds.append(preds)
                    cm = confusion_matrix(preds["y_true"], preds["y_pred"])
                    cm_sum = cm if cm_sum is None else cm_sum + cm
                    calibration_rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "model": model_key,
                            "seed": seed,
                            "accuracy": float((preds["y_true"] == preds["y_pred"]).mean()),
                            "mean_confidence": float(preds["y_conf"].mean()),
                        }
                    )
                if seed_preds:
                    y_true = np.concatenate([item["y_true"] for item in seed_preds])
                    y_pred = np.concatenate([item["y_pred"] for item in seed_preds])
                    y_conf = np.concatenate([item["y_conf"] for item in seed_preds])
                    predictions[model_key] = {"y_true": y_true, "y_pred": y_pred, "y_conf": y_conf}
                    if cm_sum is not None:
                        cm_by_model[model_key] = cm_sum

            if predictions:
                _plot_reliability_and_confidence(
                    predictions,
                    title=f"{dataset} - {family} calibration",
                    save_path=figures_dir / f"{dataset}_{family}_calibration.png",
                )

            if cm_by_model:
                _plot_failure_cases(
                    cm_by_model,
                    class_names=class_names,
                    title=f"{dataset} - {family} per-class recall",
                    save_path=figures_dir / f"{dataset}_{family}_per_class_recall.png",
                )
                for model_key, cm in cm_by_model.items():
                    recall = _class_recall_from_cm(cm)
                    worst_idx = int(np.argmin(recall))
                    failure_rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "model": model_key,
                            "worst_class": class_names[worst_idx],
                            "worst_class_recall": float(recall[worst_idx]),
                        }
                    )
                    for item in _top_misclassifications(cm, class_names, top_k=5):
                        misclassification_rows.append(
                            {
                                "dataset": dataset,
                                "family": family,
                                "model": model_key,
                                **item,
                            }
                        )

    _write_csv(analysis_dir / "calibration_summary.csv", calibration_rows, ["dataset", "family", "model", "seed", "accuracy", "mean_confidence"])
    _write_csv(analysis_dir / "failure_cases.csv", failure_rows, ["dataset", "family", "model", "worst_class", "worst_class_recall"])
    _write_csv(analysis_dir / "misclassification_clusters.csv", misclassification_rows, ["dataset", "family", "model", "true_class", "pred_class", "count"])


def _generate_training_dynamics(
    results_root: Path,
    figures_dir: Path,
    raw_results: List[Dict[str, object]],
) -> None:
    grouped = _collect_grouped_results(raw_results)
    for (dataset, fraction), rows in grouped.items():
        if abs(fraction - 1.0) > 1e-9:
            continue
        for family, (baseline_key, ga_key) in MODEL_PAIR_BY_FAMILY.items():
            histories_by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
            study_name = "study" if family == "cnn" else "study_pretrained"
            for row in rows:
                seed = int(row["seed"])
                run_dir = _full_run_dir(results_root, study_name, dataset, fraction, seed)
                for model_key in [baseline_key, ga_key]:
                    candidate_names = [
                        f"{_checkpoint_model_name(study_name, model_key)}_history.json",
                        f"{model_key}_history.json",
                    ]
                    for candidate in candidate_names:
                        history_path = run_dir / candidate
                        if history_path.exists():
                            histories_by_model[model_key].append(_load_json(history_path))
                            break
            if histories_by_model:
                _plot_training_diagnostics(
                    histories_by_model=histories_by_model,
                    title=f"{dataset} - {family} training dynamics",
                    save_path=figures_dir / f"{dataset}_{family}_training_dynamics.png",
                )


def _generate_distribution_analysis(cfg: Dict, figures_dir: Path, datasets: Sequence[str]) -> None:
    for dataset in datasets:
        x, class_names = _get_input_batch(cfg, dataset)
        representations = {
            "RGB": x,
            "Sobel XY": _apply_representation(x, "sobel_xy"),
            "Magnitude": _apply_representation(x, "magnitude"),
            "Full GA": _apply_representation(x, "full"),
        }

        fig, axes = plt.subplots(len(representations), 3, figsize=(16, 3.4 * len(representations)), dpi=180)
        if len(representations) == 1:
            axes = np.array([axes])

        for row_idx, (name, tensor) in enumerate(representations.items()):
            mean = tensor.mean(dim=(0, 2, 3)).cpu().numpy()
            var = tensor.var(dim=(0, 2, 3), unbiased=False).cpu().numpy()
            flat = tensor.detach().cpu().numpy()
            corr = np.corrcoef(flat.transpose(1, 0, 2, 3).reshape(flat.shape[1], -1))

            ax = axes[row_idx, 0]
            ax.bar(np.arange(len(mean)), mean, color="#4c72b0")
            ax.set_title(f"{name} channel mean")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Mean")

            ax = axes[row_idx, 1]
            ax.bar(np.arange(len(var)), var, color="#55a868")
            ax.set_title(f"{name} channel variance")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Variance")

            ax = axes[row_idx, 2]
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_title(f"{name} channel correlation")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Channel")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"{dataset} input distribution analysis")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{dataset}_distribution_summary.png", bbox_inches="tight")
        plt.close(fig)

        full_ga = representations["Full GA"].detach().cpu().numpy()
        c = x.shape[1]
        gradient_magnitude = full_ga[:, 3 * c : 4 * c].reshape(-1)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        ax.hist(gradient_magnitude, bins=50, color="#c44e52", alpha=0.85, density=True)
        ax.set_title(f"{dataset} gradient magnitude histogram")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{dataset}_gradient_magnitude_hist.png", bbox_inches="tight")
        plt.close(fig)


def _generate_statistical_tables(
    raw_results: List[Dict[str, object]],
    analysis_dir: Path,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for row in raw_results:
        dataset = str(row["dataset"])
        fraction = float(row["train_fraction"])
        key = f"{dataset}|fraction={fraction:.2f}"
        block = summary.setdefault(key, defaultdict(dict))
        for model_key in ["baseline", "ga_cnn", "densenet121", "ga_densenet121"]:
            if model_key not in row:
                continue
            metrics = row[model_key]
            model_block = block.setdefault(model_key, defaultdict(list))
            for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"]:
                if metric in metrics:
                    model_block[metric].append(float(metrics[metric]))

    summary_output = {}
    table_rows = []
    for key, block in summary.items():
        summary_output[key] = {}
        dataset, fraction_part = key.split("|fraction=")
        fraction = float(fraction_part)
        for model_key, metrics in block.items():
            summary_output[key][model_key] = {
                metric: summarize_metric(values) for metric, values in metrics.items()
            }
        if "baseline" in block and "ga_cnn" in block:
            for metric in ["accuracy", "f1_macro"]:
                base_vals = block["baseline"][metric]
                ga_vals = block["ga_cnn"][metric]
                table_rows.append(
                    {
                        "dataset": dataset,
                        "fraction": fraction,
                        "family": "cnn",
                        "metric": metric,
                        "baseline_mean": summarize_metric(base_vals)["mean"],
                        "baseline_ci95_low": summarize_metric(base_vals)["ci95_low"],
                        "baseline_ci95_high": summarize_metric(base_vals)["ci95_high"],
                        "ga_mean": summarize_metric(ga_vals)["mean"],
                        "ga_ci95_low": summarize_metric(ga_vals)["ci95_low"],
                        "ga_ci95_high": summarize_metric(ga_vals)["ci95_high"],
                        "paired_t_pvalue": paired_t_test(base_vals, ga_vals)["pvalue"],
                        "wilcoxon_pvalue": wilcoxon_signed_rank_test(base_vals, ga_vals)["pvalue"],
                        "cohens_d": cohens_d_paired(base_vals, ga_vals),
                        "significant": "*" if min(paired_t_test(base_vals, ga_vals)["pvalue"], wilcoxon_signed_rank_test(base_vals, ga_vals)["pvalue"]) < 0.05 else "",
                    }
                )
        if "densenet121" in block and "ga_densenet121" in block:
            for metric in ["accuracy", "f1_macro"]:
                base_vals = block["densenet121"][metric]
                ga_vals = block["ga_densenet121"][metric]
                table_rows.append(
                    {
                        "dataset": dataset,
                        "fraction": fraction,
                        "family": "densenet121",
                        "metric": metric,
                        "baseline_mean": summarize_metric(base_vals)["mean"],
                        "baseline_ci95_low": summarize_metric(base_vals)["ci95_low"],
                        "baseline_ci95_high": summarize_metric(base_vals)["ci95_high"],
                        "ga_mean": summarize_metric(ga_vals)["mean"],
                        "ga_ci95_low": summarize_metric(ga_vals)["ci95_low"],
                        "ga_ci95_high": summarize_metric(ga_vals)["ci95_high"],
                        "paired_t_pvalue": paired_t_test(base_vals, ga_vals)["pvalue"],
                        "wilcoxon_pvalue": wilcoxon_signed_rank_test(base_vals, ga_vals)["pvalue"],
                        "cohens_d": cohens_d_paired(base_vals, ga_vals),
                        "significant": "*" if min(paired_t_test(base_vals, ga_vals)["pvalue"], wilcoxon_signed_rank_test(base_vals, ga_vals)["pvalue"]) < 0.05 else "",
                    }
                )

    _save_json(analysis_dir / "summary_statistics_enriched.json", summary_output)
    _write_csv(
        analysis_dir / "summary_statistics_enriched.csv",
        table_rows,
        list(table_rows[0].keys()) if table_rows else [],
    )

    tex_lines = [
        r"\begin{tabular}{lllrcccccc}",
        r"\hline",
        r"Dataset & Family & Metric & Baseline & GA & $p_t$ & $p_W$ & $d$ & Sig \\",
        r"\hline",
    ]
    for row in table_rows:
        tex_lines.append(
            f"{row['dataset']} & {row['family']} & {row['metric']} & "
            f"{row['baseline_mean']:.4f} [{row['baseline_ci95_low']:.4f}, {row['baseline_ci95_high']:.4f}] & "
            f"{row['ga_mean']:.4f} [{row['ga_ci95_low']:.4f}, {row['ga_ci95_high']:.4f}] & "
            f"{row['paired_t_pvalue']:.4f} & {row['wilcoxon_pvalue']:.4f} & {row['cohens_d']:.3f} & {row['significant']} \\")
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    with open(analysis_dir / "summary_statistics_enriched.tex", "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex_lines))

    return summary_output


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-grade post-hoc analysis artifacts.")
    parser.add_argument("--config", default="experiments/config.yaml", type=str)
    parser.add_argument("--results-root", default="experiments/results", type=str)
    parser.add_argument("--figures-dir", default="experiments/results/figures/analysis", type=str)
    parser.add_argument("--analysis-dir", default="experiments/results/analysis", type=str)
    parser.add_argument("--datasets", default="pathmnist,bloodmnist,dermamnist", type=str)
    parser.add_argument("--recompute-predictions", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    results_root = Path(args.results_root)
    figures_dir = Path(args.figures_dir)
    analysis_dir = Path(args.analysis_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(analysis_dir)

    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    device = _device_from_torch()
    set_global_seed(int(cfg.get("seed", 42)))

    raw_results = _load_raw_results(results_root, "study") + _load_raw_results(results_root, "study_pretrained")
    summary = _load_summary(results_root, "study")
    summary.update(_load_summary(results_root, "study_pretrained"))

    # Statistical tables from the saved run-level metrics.
    if raw_results:
        summary_output = _generate_statistical_tables(raw_results, analysis_dir)
    else:
        summary_output = summary

    # Analysis modules that can be built directly from the saved checkpoints and data.
    _generate_distribution_analysis(cfg=cfg, figures_dir=figures_dir, datasets=datasets)
    _generate_training_dynamics(results_root=results_root, figures_dir=figures_dir, raw_results=raw_results)
    _generate_calibration_and_failure_analysis(
        cfg=cfg,
        device=device,
        results_root=results_root,
        analysis_dir=analysis_dir,
        figures_dir=figures_dir,
        raw_results=raw_results,
    )
    _make_metrics_table(raw_results, analysis_dir, table_name="metrics_comparison")
    _make_final_summary(summary_output, analysis_dir)

    print(f"Analysis complete. Figures saved to: {figures_dir}")
    print(f"Analysis artifacts saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
