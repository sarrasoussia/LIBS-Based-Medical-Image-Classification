from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import torch
import yaml

from data.pathmnist import get_medmnist_dataloaders, set_global_seed
from models.factory import build_model
from training.evaluate import evaluate_model
from training.train import train_model
from utils.statistics import paired_permutation_pvalue, summarize_metric


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


def main() -> None:
    with open("experiments/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    study_cfg = cfg.get("study", {})
    datasets = study_cfg.get("datasets", [cfg.get("dataset", {}).get("name", "pathmnist")])
    seeds = study_cfg.get("seeds", [cfg.get("seed", 42)])
    train_fractions = study_cfg.get("train_fractions", [1.0])

    out_root = os.path.join(cfg["paths"]["results_dir"], "study_pretrained")
    os.makedirs(out_root, exist_ok=True)

    models = ["densenet121", "ga_densenet121"]
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
                    for m in ["densenet121", "ga_densenet121"]
                )
                if existing_all:
                    print(
                        f"[pretrained-study] skip completed dataset={dataset_name} "
                        f"fraction={train_fraction:.2f} seed={seed}"
                    )
                    continue

                set_global_seed(seed)
                print(f"[pretrained-study] dataset={dataset_name} fraction={train_fraction:.2f} seed={seed}")

                loaders = get_medmnist_dataloaders(
                    dataset_name=dataset_name,
                    data_dir=cfg["paths"]["data_dir"],
                    batch_size=cfg["training"]["batch_size"],
                    num_workers=cfg["training"]["num_workers"],
                    normalize=True,
                    seed=seed,
                    train_fraction=float(train_fraction),
                )

                for model_name in models:
                    metrics_path = os.path.join(run_dir, f"{model_name}_metrics.json")
                    if os.path.exists(metrics_path):
                        print(f"  - {model_name}: metrics already exist, skipping training")
                        continue

                    model = build_model(
                        model_name=model_name,
                        in_channels=loaders.in_channels,
                        num_classes=loaders.num_classes,
                        include_higher_order=cfg["ga"].get("include_higher_order", True),
                        pretrained=cfg.get("model", {}).get("pretrained", True),
                        adapt_for_small_inputs=cfg.get("model", {}).get("adapt_for_small_inputs", True),
                    ).to(device)

                    checkpoint_path = os.path.join(run_dir, f"{model_name}_best.pt")
                    if os.path.exists(checkpoint_path):
                        print(f"  - {model_name}: found checkpoint, evaluating without retraining")
                        state = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(state)
                    else:
                        model, history, _ = train_model(
                            model=model,
                            model_name=model_name,
                            train_loader=loaders.train,
                            val_loader=loaders.val,
                            device=device,
                            save_dir=run_dir,
                            epochs=cfg["training"]["epochs"],
                            learning_rate=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"],
                            early_stopping_patience=cfg["training"]["early_stopping_patience"],
                        )

                        with open(os.path.join(run_dir, f"{model_name}_history.json"), "w", encoding="utf-8") as f:
                            json.dump(asdict(history), f, indent=2)

                    _ = evaluate_model(
                        model=model,
                        loader=loaders.test,
                        device=device,
                        save_path=metrics_path,
                    )

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
                ga_dense_metrics = _read_metrics(os.path.join(run_dir, "ga_densenet121_metrics.json"))
                if dense_metrics is None or ga_dense_metrics is None:
                    continue
                raw_results.append(
                    {
                        "dataset": dataset_name,
                        "train_fraction": float(train_fraction),
                        "seed": int(seed),
                        "densenet121": dense_metrics,
                        "ga_densenet121": ga_dense_metrics,
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
            ga_dense_vals = {m: [r["ga_densenet121"][m] for r in group] for m in metrics_for_summary}

            pvals = {
                m: paired_permutation_pvalue(dense_vals[m], ga_dense_vals[m])
                for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            }

            summary[key] = {
                "n_seeds": len(group),
                "densenet121": {m: summarize_metric(dense_vals[m]) for m in metrics_for_summary},
                "ga_densenet121": {m: summarize_metric(ga_dense_vals[m]) for m in metrics_for_summary},
                "paired_permutation_pvalue": pvals,
            }

    summary_path = os.path.join(out_root, "summary_statistics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
