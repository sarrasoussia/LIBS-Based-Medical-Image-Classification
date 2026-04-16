from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.pathmnist import get_medmnist_dataloaders, set_global_seed
from models.factory import build_model
from training.evaluate import evaluate_model
from training.train import train_model
from utils.statistics import (
    cohens_d_paired,
    paired_t_test,
    summarize_metric,
    wilcoxon_signed_rank_test,
)

ABLATION_SETTINGS = ["baseline", "sobel_xy", "magnitude", "full"]
BACKBONE_TO_BASELINE = {
    "cnn": "baseline_cnn",
    "densenet121": "densenet121",
}
BACKBONE_TO_GA = {
    "cnn": "ga_cnn",
    "densenet121": "ga_densenet121",
}
SETTING_TO_REPRESENTATION = {
    "baseline": None,
    "sobel_xy": "sobel_xy",
    "magnitude": "magnitude",
    "full": "full",
}


def _ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _device_from_torch() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_models_for_setting(
    backbone: str,
    setting: str,
    in_channels: int,
    num_classes: int,
    include_higher_order: bool,
    pretrained: bool,
    adapt_for_small_inputs: bool,
) -> Tuple[str, Dict[str, object]]:
    if backbone not in BACKBONE_TO_BASELINE:
        raise ValueError(f"Unsupported backbone '{backbone}'.")
    if setting not in ABLATION_SETTINGS:
        raise ValueError(f"Unsupported ablation setting '{setting}'.")

    model_name = BACKBONE_TO_BASELINE[backbone] if setting == "baseline" else BACKBONE_TO_GA[backbone]
    model_kwargs: Dict[str, object] = {
        "model_name": model_name,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "include_higher_order": include_higher_order,
        "representation_mode": SETTING_TO_REPRESENTATION[setting],
        "pretrained": pretrained,
        "adapt_for_small_inputs": adapt_for_small_inputs,
    }
    return model_name, model_kwargs


def run_ablation_study(
    cfg: Dict,
    device: torch.device,
    datasets: Iterable[str],
    seeds: Iterable[int],
    train_fractions: Iterable[float],
    backbones: Iterable[str] = ("cnn", "densenet121"),
    output_root: str | Path | None = None,
) -> None:
    """Run A1-A4 representation ablations for the selected backbones and datasets."""
    datasets = [d.lower() for d in datasets]
    backbones = [b.lower() for b in backbones]
    output_root = Path(output_root or Path(cfg["paths"]["results_dir"]) / "analysis" / "ablation")
    _ensure_dir(output_root)

    raw_results: List[Dict[str, object]] = []

    for dataset_name in datasets:
        for train_fraction in train_fractions:
            for seed in seeds:
                set_global_seed(int(seed))
                loaders = get_medmnist_dataloaders(
                    dataset_name=dataset_name,
                    data_dir=cfg["paths"]["data_dir"],
                    batch_size=int(cfg["training"]["batch_size"]),
                    num_workers=int(cfg["training"].get("num_workers", 0)),
                    normalize=True,
                    seed=int(seed),
                    train_fraction=float(train_fraction),
                )

                run_dir = output_root / dataset_name / f"fraction_{train_fraction:.2f}" / f"seed_{seed}"
                _ensure_dir(run_dir)

                for backbone in backbones:
                    print(
                        f"[ablation] dataset={dataset_name} backbone={backbone} "
                        f"fraction={train_fraction:.2f} seed={seed}"
                    )
                    for setting in ABLATION_SETTINGS:
                        model_name, model_kwargs = _build_models_for_setting(
                            backbone=backbone,
                            setting=setting,
                            in_channels=loaders.in_channels,
                            num_classes=loaders.num_classes,
                            include_higher_order=bool(cfg.get("ga", {}).get("include_higher_order", True)),
                            pretrained=bool(cfg.get("model", {}).get("pretrained", True)),
                            adapt_for_small_inputs=bool(cfg.get("model", {}).get("adapt_for_small_inputs", True)),
                        )

                        model = build_model(**model_kwargs).to(device)
                        setting_dir = run_dir / backbone / setting
                        _ensure_dir(setting_dir)

                        model, history, _ = train_model(
                            model=model,
                            model_name=model_name,
                            train_loader=loaders.train,
                            val_loader=loaders.val,
                            device=device,
                            save_dir=str(setting_dir),
                            epochs=int(cfg["training"]["epochs"]),
                            learning_rate=float(cfg["training"]["learning_rate"]),
                            weight_decay=float(cfg["training"]["weight_decay"]),
                            early_stopping_patience=int(cfg["training"]["early_stopping_patience"]),
                        )

                        metrics = evaluate_model(
                            model=model,
                            loader=loaders.test,
                            device=device,
                            save_path=str(setting_dir / f"{model_name}_metrics.json"),
                        )

                        with open(setting_dir / f"{model_name}_history.json", "w", encoding="utf-8") as handle:
                            json.dump(asdict(history), handle, indent=2)

                        raw_results.append(
                            {
                                "dataset": dataset_name,
                                "backbone": backbone,
                                "train_fraction": float(train_fraction),
                                "seed": int(seed),
                                "setting": setting,
                                "metrics": {k: metrics[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"]},
                            }
                        )

    raw_path = output_root / "raw_results.json"
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(raw_results, handle, indent=2)

    summary_rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        for backbone in backbones:
            for setting in ABLATION_SETTINGS:
                values = [
                    row["metrics"]
                    for row in raw_results
                    if row["dataset"] == dataset_name
                    and row["backbone"] == backbone
                    and row["setting"] == setting
                ]
                if not values:
                    continue
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "backbone": backbone,
                        "setting": setting,
                        **{
                            metric: summarize_metric([row[metric] for row in values])
                            for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                        },
                    }
                )

    summary_path = output_root / "summary_statistics.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    # Paired significance summaries versus the baseline setting.
    significance_rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        for backbone in backbones:
            baseline = [
                row["metrics"]
                for row in raw_results
                if row["dataset"] == dataset_name
                and row["backbone"] == backbone
                and row["setting"] == "baseline"
            ]
            for setting in ["sobel_xy", "magnitude", "full"]:
                challenger = [
                    row["metrics"]
                    for row in raw_results
                    if row["dataset"] == dataset_name
                    and row["backbone"] == backbone
                    and row["setting"] == setting
                ]
                if not baseline or not challenger:
                    continue
                paired = {
                    metric: {
                        "paired_t_test": paired_t_test(
                            [item[metric] for item in baseline],
                            [item[metric] for item in challenger],
                        ),
                        "wilcoxon": wilcoxon_signed_rank_test(
                            [item[metric] for item in baseline],
                            [item[metric] for item in challenger],
                        ),
                        "cohens_d": cohens_d_paired(
                            [item[metric] for item in baseline],
                            [item[metric] for item in challenger],
                        ),
                    }
                    for metric in ["accuracy", "f1_macro"]
                }
                significance_rows.append(
                    {
                        "dataset": dataset_name,
                        "backbone": backbone,
                        "setting": setting,
                        "paired_tests": paired,
                    }
                )

    with open(output_root / "significance_tests.json", "w", encoding="utf-8") as handle:
        json.dump(significance_rows, handle, indent=2)

    # Simple comparison plot for each backbone and dataset.
    for dataset_name in datasets:
        for backbone in backbones:
            rows = [
                row
                for row in raw_results
                if row["dataset"] == dataset_name and row["backbone"] == backbone
            ]
            if not rows:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=180)
            for ax, metric in zip(axes, ["accuracy", "f1_macro"]):
                means = []
                labels = []
                for setting in ABLATION_SETTINGS:
                    vals = [row["metrics"][metric] for row in rows if row["setting"] == setting]
                    if not vals:
                        continue
                    means.append(float(np.mean(vals)))
                    labels.append(setting)
                ax.bar(labels, means, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"][: len(labels)])
                ax.set_title(f"{dataset_name} - {backbone} - {metric}")
                ax.set_ylabel(metric)
                ax.set_ylim(0.0, 1.0)
                ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(output_root / f"{dataset_name}_{backbone}_ablation.png", bbox_inches="tight")
            plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA ablation experiments.")
    parser.add_argument("--config", default="experiments/config.yaml", type=str)
    parser.add_argument("--datasets", default="pathmnist", type=str)
    parser.add_argument("--seeds", default="13,42,101,202,777", type=str)
    parser.add_argument("--fractions", default="1.0", type=str)
    parser.add_argument("--backbones", default="cnn,densenet121", type=str)
    parser.add_argument("--output-root", default="", type=str)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    device = _device_from_torch()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    fractions = [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
    backbones = [item.strip().lower() for item in args.backbones.split(",") if item.strip()]
    output_root = args.output_root or None
    run_ablation_study(
        cfg=cfg,
        device=device,
        datasets=datasets,
        seeds=seeds,
        train_fractions=fractions,
        backbones=backbones,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
