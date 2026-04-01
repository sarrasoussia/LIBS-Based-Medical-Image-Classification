from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import torch

from data.pathmnist import get_medmnist_dataloaders, set_global_seed
from models.cnn_baseline import BaselineCNN
from models.ga_cnn_model import GACNN
from training.evaluate import evaluate_model
from training.train import train_model
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
    }

    baseline = BaselineCNN(
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

    ga_model = GACNN(
        in_channels=loaders.in_channels,
        num_classes=loaders.num_classes,
        include_higher_order=cfg["ga"]["include_higher_order"],
    ).to(device)
    ga_model, ga_hist, _ = train_model(
        model=ga_model,
        model_name="ga_cnn",
        **train_kwargs,
    )
    ga_metrics = evaluate_model(
        model=ga_model,
        loader=loaders.test,
        device=device,
        save_path=os.path.join(run_dir, "ga_metrics.json"),
    )
    with open(os.path.join(run_dir, "ga_history.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(ga_hist), f, indent=2)

    return {
        "baseline": baseline_metrics,
        "ga_cnn": ga_metrics,
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
                        "baseline": {
                            k: metrics_pair["baseline"][k]
                            for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                        },
                        "ga_cnn": {
                            k: metrics_pair["ga_cnn"][k]
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
            baseline_vals = {m: [r["baseline"][m] for r in group] for m in metrics}
            ga_vals = {m: [r["ga_cnn"][m] for r in group] for m in metrics}

            summary[key] = {
                "n_seeds": len(group),
                "baseline": {m: summarize_metric(baseline_vals[m]) for m in metrics},
                "ga_cnn": {m: summarize_metric(ga_vals[m]) for m in metrics},
                "paired_permutation_pvalue": {
                    m: paired_permutation_pvalue(baseline_vals[m], ga_vals[m]) for m in metrics
                },
            }

    with open(os.path.join(root_out, "summary_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Study complete. Saved to: {root_out}")
