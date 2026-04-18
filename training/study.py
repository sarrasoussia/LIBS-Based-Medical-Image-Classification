from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import torch

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


def _run_single_model_pair(
    cfg: Dict,
    dataset_name: str,
    train_fraction: float,
    seed: int,
    device: torch.device,
    run_dir: str,
) -> Dict[str, Dict[str, float]]:
    set_global_seed(seed)

    loaders = get_medmnist_dataloaders(
        dataset_name=dataset_name,
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        normalize=True,
        seed=seed,
        train_fraction=train_fraction,
    )

    train_kwargs = {
        "train_loader": loaders.train,
        "val_loader": loaders.val,
        "device": device,
        "save_dir": run_dir,
        "epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "early_stopping_patience": cfg["training"]["early_stopping_patience"],
        "optimizer_name": cfg.get("training", {}).get("optimizer", {}).get("name", "adam"),
        "optimizer_momentum": float(cfg.get("training", {}).get("optimizer", {}).get("momentum", 0.9)),
        "optimizer_adam_beta1": float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta1", 0.9)),
        "optimizer_adam_beta2": float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta2", 0.999)),
        "scheduler_name": cfg.get("training", {}).get("scheduler", {}).get("name", "cosine"),
        "scheduler_step_size": int(cfg.get("training", {}).get("scheduler", {}).get("step_size", 10)),
        "scheduler_gamma": float(cfg.get("training", {}).get("scheduler", {}).get("gamma", 0.1)),
        "min_learning_rate": float(cfg.get("training", {}).get("scheduler", {}).get("min_lr", 1e-6)),
        "logit_temperature_start": float(cfg.get("training", {}).get("logit_temperature", {}).get("start", 1.0)),
        "logit_temperature_end": float(cfg.get("training", {}).get("logit_temperature", {}).get("end", 1.0)),
    }

    baseline = build_model(
        model_name="baseline_cnn",
        in_channels=loaders.in_channels,
        num_classes=loaders.num_classes,
    ).to(device)
    baseline, baseline_hist, _ = train_model(
        model=baseline,
        model_name="baseline_cnn",
        **train_kwargs,
    )
    baseline_metrics = evaluate_model(
        model=baseline,
        loader=loaders.test,
        device=device,
        save_path=os.path.join(run_dir, "baseline_metrics.json"),
    )
    with open(os.path.join(run_dir, "baseline_history.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(baseline_hist), f, indent=2)

    libs_model = build_model(
        model_name="libs_cnn",
        in_channels=loaders.in_channels,
        num_classes=loaders.num_classes,
    ).to(device)
    libs_model, libs_hist, _ = train_model(
        model=libs_model,
        model_name="libs_cnn",
        **train_kwargs,
    )
    libs_metrics = evaluate_model(
        model=libs_model,
        loader=loaders.test,
        device=device,
        save_path=os.path.join(run_dir, "libs_metrics.json"),
    )
    with open(os.path.join(run_dir, "libs_history.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(libs_hist), f, indent=2)

    return {
        "baseline_cnn": baseline_metrics,
        "libs_cnn": libs_metrics,
    }


def run_study(cfg: Dict, device: torch.device) -> None:
    """Run controlled study across datasets, seeds, and train-data fractions."""
    study_cfg = cfg.get("study", {})
    datasets = study_cfg.get("datasets", [cfg.get("dataset", {}).get("name", "pathmnist")])
    seeds = study_cfg.get("seeds", [cfg.get("seed", 42)])
    train_fractions = study_cfg.get("train_fractions", [1.0])

    root_out = os.path.join(cfg["paths"]["results_dir"], "study")
    os.makedirs(root_out, exist_ok=True)

    raw_results: List[Dict] = []
    for dataset_name in datasets:
        for train_fraction in train_fractions:
            for seed in seeds:
                run_dir = os.path.join(
                    root_out,
                    dataset_name,
                    f"fraction_{train_fraction:.2f}",
                    f"seed_{seed}",
                )
                os.makedirs(run_dir, exist_ok=True)
                print(
                    f"[Study] dataset={dataset_name} fraction={train_fraction:.2f} "
                    f"seed={seed}"
                )
                metrics_pair = _run_single_model_pair(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    train_fraction=train_fraction,
                    seed=seed,
                    device=device,
                    run_dir=run_dir,
                )
                raw_results.append(
                    {
                        "dataset": dataset_name,
                        "train_fraction": train_fraction,
                        "seed": seed,
                        "baseline_cnn": {
                            k: metrics_pair["baseline_cnn"][k]
                            for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                        },
                        "libs_cnn": {
                            k: metrics_pair["libs_cnn"][k]
                            for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                        },
                    }
                )

    with open(os.path.join(root_out, "raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    summary: Dict[str, Dict] = {}
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    for dataset_name in datasets:
        for train_fraction in train_fractions:
            key = f"{dataset_name}|fraction={train_fraction:.2f}"
            group = [
                r
                for r in raw_results
                if r["dataset"] == dataset_name and abs(r["train_fraction"] - train_fraction) < 1e-9
            ]
            baseline_vals = {m: [r["baseline_cnn"][m] for r in group] for m in metrics}
            libs_vals = {m: [r["libs_cnn"][m] for r in group] for m in metrics}

            summary[key] = {
                "n_seeds": len(group),
                "baseline_cnn": {m: summarize_metric(baseline_vals[m]) for m in metrics},
                "libs_cnn": {m: summarize_metric(libs_vals[m]) for m in metrics},
                "paired_permutation_pvalue": {
                    m: paired_permutation_pvalue(baseline_vals[m], libs_vals[m]) for m in metrics
                },
            }

    with open(os.path.join(root_out, "summary_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    figures_dir = os.path.join(root_out, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    try:
        save_pipeline_diagram(os.path.join(figures_dir, "pipeline_baseline_vs_libs.png"))
        plot_study_metric_distributions(
            raw_results=raw_results,
            model_keys=["baseline_cnn", "libs_cnn"],
            metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            save_path=os.path.join(figures_dir, "seed_metric_distributions.png"),
        )
        plot_study_summary_bars(
            summary=summary,
            model_keys=["baseline_cnn", "libs_cnn"],
            metric_names=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            save_path=os.path.join(figures_dir, "summary_means_ci95.png"),
        )
        print(f"Study figures saved to: {figures_dir}")
    except Exception as exc:
        print(f"WARNING: could not generate study figures: {exc}")

    print(f"Study complete. Saved to: {root_out}")
