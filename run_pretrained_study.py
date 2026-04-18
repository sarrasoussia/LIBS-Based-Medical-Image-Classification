from __future__ import annotations

import argparse
import json
import os
import gc
from dataclasses import asdict
from typing import Dict, List

import torch
import yaml

from data.pathmnist import get_medmnist_dataloaders, set_global_seed
from models.factory import build_model
from training.evaluate import evaluate_model
from training.train import train_model
from utils.report_figures import (
    plot_study_metric_distributions,
    plot_study_summary_bars,
    save_pipeline_diagram,
)
from utils.statistics import paired_permutation_pvalue, summarize_metric
from data.pathmnist import labels_to_long


def _read_metrics(metrics_path: str) -> Dict[str, float] | None:
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "accuracy": data["accuracy"],
        "precision_macro": data["precision_macro"],
        "recall_macro": data["recall_macro"],
        "f1_macro": data["f1_macro"],
        "ece": data["ece"],
    }


def _read_history_max_val_acc(history_path: str) -> float | None:
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vals = data.get("val_acc", [])
        if not vals:
            return None
        return float(max(vals))
    except Exception:
        return None


@torch.no_grad()
def _compute_val_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels_cpu = labels_to_long(labels)
        labels_cpu = labels_cpu.squeeze()
        labels_cpu = labels_cpu.to(torch.int64)
        labels_cpu = labels_cpu.contiguous()
        if labels_cpu.ndim == 0:
            labels_cpu = labels_cpu.unsqueeze(0)
        labels = labels_cpu.to(device)

        logits = model(images)
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
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def _effective_batch_size(cfg: Dict, dataset_name: str, device: torch.device) -> int:
    base_bs = int(cfg["training"]["batch_size"])
    if dataset_name.lower() == "pcam" and device.type == "mps":
        return min(base_bs, 16)
    return base_bs


def _effective_num_workers(cfg: Dict, dataset_name: str) -> int:
    base_workers = int(cfg["training"].get("num_workers", 0))
    if dataset_name.lower() != "pcam":
        return base_workers

    # On macOS (spawn start method), torchvision PCAM objects can fail to pickle
    # in worker processes. Keep workers at 0 to avoid runtime crashes.
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin":
        return 0

    cpu_count = os.cpu_count() or 4
    recommended = max(2, min(8, cpu_count - 1))
    return max(base_workers, recommended)


def _effective_adapt_for_small_inputs(cfg: Dict, dataset_name: str) -> bool:
    default_value = bool(cfg.get("model", {}).get("adapt_for_small_inputs", True))
    if dataset_name.lower() == "pcam":
        return False
    return default_value


def _effective_training_schedule(cfg: Dict, dataset_name: str) -> Dict[str, int]:
    training_cfg = cfg.get("training", {})
    default_epochs = int(training_cfg.get("epochs", 50))
    default_patience = int(training_cfg.get("early_stopping_patience", 7))

    if dataset_name.lower() != "pcam":
        return {
            "epochs": default_epochs,
            "early_stopping_patience": default_patience,
        }

    pcam_cfg = training_cfg.get("pcam_fast", {})
    return {
        "epochs": int(pcam_cfg.get("epochs", min(default_epochs, 20))),
        "early_stopping_patience": int(
            pcam_cfg.get("early_stopping_patience", min(default_patience, 4))
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pretrained study experiments.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names to run (e.g., bloodmnist,dermamnist).",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="",
        help="Comma-separated train fractions to run (e.g., 0.25,0.5,1.0).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to run (e.g., 13,42,101).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain/re-evaluate even if metrics already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open("experiments/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    use_mps = os.environ.get("USE_MPS", "1") != "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    study_cfg = cfg.get("pretrained_study", cfg.get("study", {}))
    datasets = study_cfg.get("datasets", [cfg.get("dataset", {}).get("name", "pathmnist")])
    seeds = study_cfg.get("seeds", [cfg.get("seed", 42)])
    train_fractions = study_cfg.get("train_fractions", [1.0])

    if args.datasets.strip():
        datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if args.fractions.strip():
        train_fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    print(
        "Pretrained study plan: "
        f"datasets={datasets}, fractions={train_fractions}, seeds={seeds}"
    )

    out_root = os.path.join(cfg["paths"]["results_dir"], "study_pretrained")
    os.makedirs(out_root, exist_ok=True)

    models = ["densenet121", "libs_densenet121"]
    raw_results: List[Dict] = []

    for dataset_name in datasets:
        for train_fraction in train_fractions:
            for seed in seeds:
                run_dir = os.path.join(
                    out_root,
                    dataset_name,
                    f"fraction_{train_fraction:.2f}",
                    f"seed_{seed}",
                )
                os.makedirs(run_dir, exist_ok=True)

                existing_all = all(
                    os.path.exists(os.path.join(run_dir, f"{m}_metrics.json"))
                    for m in models
                )
                if existing_all and not args.force:
                    continue

                set_global_seed(seed)
                print(f"[pretrained-study] dataset={dataset_name} fraction={train_fraction:.2f} seed={seed}")

                batch_size = _effective_batch_size(cfg, dataset_name, device)

                num_workers = _effective_num_workers(cfg, dataset_name)

                adapt_for_small_inputs = _effective_adapt_for_small_inputs(cfg, dataset_name)

                schedule = _effective_training_schedule(cfg, dataset_name)

                loaders = get_medmnist_dataloaders(
                    dataset_name=dataset_name,
                    data_dir=cfg["paths"]["data_dir"],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    normalize=True,
                    seed=seed,
                    train_fraction=float(train_fraction),
                )

                for model_name in models:
                    metrics_path = os.path.join(run_dir, f"{model_name}_metrics.json")
                    if os.path.exists(metrics_path) and not args.force:
                        continue

                    model_kwargs = {
                        "model_name": model_name,
                        "in_channels": loaders.in_channels,
                        "num_classes": loaders.num_classes,
                        "pretrained": cfg.get("model", {}).get("pretrained", True),
                        "adapt_for_small_inputs": adapt_for_small_inputs,
                        "libs_use_sobel": bool(cfg.get("libs", {}).get("use_sobel", True)),
                        "libs_use_fusion": bool(cfg.get("libs", {}).get("use_fusion", True)),
                        "libs_sobel_mode": str(cfg.get("libs", {}).get("sobel_mode", "magnitude")),
                        "libs_raw_use_conv": bool(cfg.get("libs", {}).get("raw_use_conv", False)),
                    }

                    model = build_model(
                        model_name=model_name,
                        in_channels=loaders.in_channels,
                        num_classes=loaders.num_classes,
                        pretrained=cfg.get("model", {}).get("pretrained", True),
                        adapt_for_small_inputs=adapt_for_small_inputs,
                        libs_use_sobel=bool(cfg.get("libs", {}).get("use_sobel", True)),
                        libs_use_fusion=bool(cfg.get("libs", {}).get("use_fusion", True)),
                        libs_sobel_mode=str(cfg.get("libs", {}).get("sobel_mode", "magnitude")),
                        libs_raw_use_conv=bool(cfg.get("libs", {}).get("raw_use_conv", False)),
                    ).to(device)

                    checkpoint_path = os.path.join(run_dir, f"{model_name}_best.pt")
                    should_train = True
                    need_fresh_init = False
                    if os.path.exists(checkpoint_path):
                        state = torch.load(checkpoint_path, map_location=device)
                        try:
                            model.load_state_dict(state)
                            should_train = False

                            history_path = os.path.join(run_dir, f"{model_name}_history.json")
                            history_max_val_acc = _read_history_max_val_acc(history_path)
                            if history_max_val_acc is not None:
                                ckpt_val_acc = _compute_val_accuracy(model, loaders.val, device)
                                if history_max_val_acc - ckpt_val_acc > 0.20:
                                    should_train = True
                                    need_fresh_init = True
                        except RuntimeError as exc:
                            need_fresh_init = True

                    if should_train:
                        if need_fresh_init:
                            del model
                            gc.collect()
                            if device.type == "mps":
                                torch.mps.empty_cache()
                            model = build_model(**model_kwargs).to(device)

                        model, history, _ = train_model(
                            model=model,
                            model_name=model_name,
                            train_loader=loaders.train,
                            val_loader=loaders.val,
                            device=device,
                            save_dir=run_dir,
                            epochs=schedule["epochs"],
                            learning_rate=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"],
                            early_stopping_patience=schedule["early_stopping_patience"],
                            optimizer_name=cfg.get("training", {}).get("optimizer", {}).get("name", "adam"),
                            optimizer_momentum=float(cfg.get("training", {}).get("optimizer", {}).get("momentum", 0.9)),
                            optimizer_adam_beta1=float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta1", 0.9)),
                            optimizer_adam_beta2=float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta2", 0.999)),
                            scheduler_name=cfg.get("training", {}).get("scheduler", {}).get("name", "cosine"),
                            scheduler_step_size=int(cfg.get("training", {}).get("scheduler", {}).get("step_size", 10)),
                            scheduler_gamma=float(cfg.get("training", {}).get("scheduler", {}).get("gamma", 0.1)),
                            min_learning_rate=float(cfg.get("training", {}).get("scheduler", {}).get("min_lr", 1e-6)),
                            logit_temperature_start=float(cfg.get("training", {}).get("logit_temperature", {}).get("start", 1.0)),
                            logit_temperature_end=float(cfg.get("training", {}).get("logit_temperature", {}).get("end", 1.0)),
                        )

                        with open(os.path.join(run_dir, f"{model_name}_history.json"), "w", encoding="utf-8") as f:
                            json.dump(asdict(history), f, indent=2)

                    _ = evaluate_model(
                        model=model,
                        loader=loaders.test,
                        device=device,
                        save_path=metrics_path,
                    )

                    del model
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()

    for dataset_name in datasets:
        for train_fraction in train_fractions:
            for seed in seeds:
                run_dir = os.path.join(
                    out_root,
                    dataset_name,
                    f"fraction_{train_fraction:.2f}",
                    f"seed_{seed}",
                )
                dense_metrics = _read_metrics(os.path.join(run_dir, "densenet121_metrics.json"))
                libs_dense_metrics = _read_metrics(os.path.join(run_dir, "libs_densenet121_metrics.json"))
                if dense_metrics is None or libs_dense_metrics is None:
                    continue
                raw_results.append(
                    {
                        "dataset": dataset_name,
                        "train_fraction": float(train_fraction),
                        "seed": int(seed),
                        "densenet121": dense_metrics,
                        "libs_densenet121": libs_dense_metrics,
                    }
                )

    raw_path = os.path.join(out_root, "raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    metrics_for_summary = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"]
    summary = {}

    for dataset_name in datasets:
        for train_fraction in train_fractions:
            key = f"{dataset_name}|fraction={train_fraction:.2f}"
            group = [
                r
                for r in raw_results
                if r["dataset"] == dataset_name and abs(r["train_fraction"] - float(train_fraction)) < 1e-9
            ]
            if not group:
                continue

            dense_vals = {m: [r["densenet121"][m] for r in group] for m in metrics_for_summary}
            libs_dense_vals = {m: [r["libs_densenet121"][m] for r in group] for m in metrics_for_summary}

            pvals = {
                m: paired_permutation_pvalue(dense_vals[m], libs_dense_vals[m])
                for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            }

            summary[key] = {
                "n_seeds": len(group),
                "densenet121": {m: summarize_metric(dense_vals[m]) for m in metrics_for_summary},
                "libs_densenet121": {m: summarize_metric(libs_dense_vals[m]) for m in metrics_for_summary},
                "paired_permutation_pvalue": pvals,
            }

    summary_path = os.path.join(out_root, "summary_statistics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    figures_dir = os.path.join(out_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    try:
        save_pipeline_diagram(os.path.join(figures_dir, "pipeline_baseline_vs_libs.png"))
        plot_study_metric_distributions(
            raw_results=raw_results,
            model_keys=["densenet121", "libs_densenet121"],
            metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"],
            save_path=os.path.join(figures_dir, "seed_metric_distributions.png"),
        )
        plot_study_summary_bars(
            summary=summary,
            model_keys=["densenet121", "libs_densenet121"],
            metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"],
            save_path=os.path.join(figures_dir, "summary_means_ci95.png"),
        )
    except Exception as exc:
        print(f"WARNING: could not generate pretrained-study figures: {exc}")

    print(f"Pretrained study complete. Results saved in: {out_root}")


if __name__ == "__main__":
    main()
