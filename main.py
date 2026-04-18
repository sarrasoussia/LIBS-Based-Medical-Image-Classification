from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import yaml

from data.pathmnist import get_medmnist_dataloaders, set_global_seed
from models.factory import build_model
from training.evaluate import evaluate_model
from training.train import train_model

STANDARD_MODELS: List[str] = [
    "baseline_cnn",
    "densenet121",
    "libs_cnn",
    "libs_densenet121",
]
DEFAULT_SEEDS: List[int] = [13, 42, 101, 202, 777]
SUMMARY_METRICS: List[str] = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "ece",
    "brier",
]


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device() -> torch.device:
    use_mps = os.environ.get("USE_MPS", "1") != "0"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and use_mps:
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_datasets(cfg: Dict) -> List[str]:
    dataset_name = str(cfg.get("dataset", {}).get("name", "pathmnist")).lower()
    datasets = cfg.get("experiments", {}).get("datasets")
    if datasets:
        return [str(d).lower() for d in datasets]
    return [dataset_name]


def _resolve_seeds(cfg: Dict) -> List[int]:
    seeds = cfg.get("experiments", {}).get("seeds")
    if not seeds:
        return list(DEFAULT_SEEDS)
    return [int(s) for s in seeds]


def _validate_model_list(cfg: Dict) -> List[str]:
    requested = cfg.get("experiments", {}).get("run_models")
    if requested:
        req = [str(m).lower() for m in requested]
        unknown = sorted(set(req) - set(STANDARD_MODELS))
        if unknown:
            raise ValueError(
                f"Unsupported models in experiments.run_models: {unknown}. "
                f"Only supported models are: {STANDARD_MODELS}"
            )

        missing = sorted(set(STANDARD_MODELS) - set(req))
        if missing:
            raise ValueError(
                f"experiments.run_models must contain all standardized models. Missing: {missing}"
            )
    return list(STANDARD_MODELS)


def _summary(values: List[float]) -> Dict[str, float | int | List[float]]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "per_seed": [],
        }

    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - ci95),
        "ci95_high": float(mean + ci95),
        "per_seed": arr.tolist(),
    }


def _extract_metrics_for_aggregation(metrics: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in SUMMARY_METRICS:
        if key == "brier":
            value = metrics.get("brier", metrics.get("brier_score", 0.0))
        else:
            value = metrics.get(key, 0.0)
        out[key] = float(value)
    return out


def _aggregate_seed_metrics(seed_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float | int | List[float]]]:
    by_metric: Dict[str, List[float]] = {k: [] for k in SUMMARY_METRICS}
    for row in seed_metrics:
        for metric_name in SUMMARY_METRICS:
            by_metric[metric_name].append(float(row[metric_name]))

    return {metric_name: _summary(values) for metric_name, values in by_metric.items()}


def _build_train_kwargs(cfg: Dict, loaders, device: torch.device, save_dir: str) -> Dict:
    return {
        "train_loader": loaders.train,
        "val_loader": loaders.val,
        "device": device,
        "save_dir": save_dir,
        "epochs": int(cfg["training"]["epochs"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "early_stopping_patience": int(cfg["training"]["early_stopping_patience"]),
        "optimizer_name": str(cfg.get("training", {}).get("optimizer", {}).get("name", "adam")),
        "optimizer_momentum": float(cfg.get("training", {}).get("optimizer", {}).get("momentum", 0.9)),
        "optimizer_adam_beta1": float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta1", 0.9)),
        "optimizer_adam_beta2": float(cfg.get("training", {}).get("optimizer", {}).get("adam_beta2", 0.999)),
        "scheduler_name": str(cfg.get("training", {}).get("scheduler", {}).get("name", "cosine")),
        "scheduler_step_size": int(cfg.get("training", {}).get("scheduler", {}).get("step_size", 10)),
        "scheduler_gamma": float(cfg.get("training", {}).get("scheduler", {}).get("gamma", 0.1)),
        "min_learning_rate": float(cfg.get("training", {}).get("scheduler", {}).get("min_lr", 1e-6)),
        "logit_temperature_start": float(cfg.get("training", {}).get("logit_temperature", {}).get("start", 1.0)),
        "logit_temperature_end": float(cfg.get("training", {}).get("logit_temperature", {}).get("end", 1.0)),
        "auto_resume": bool(cfg.get("training", {}).get("auto_resume", False)),
    }


def _build_model(cfg: Dict, model_name: str, in_channels: int, num_classes: int):
    return build_model(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=bool(cfg.get("model", {}).get("pretrained", True)),
        adapt_for_small_inputs=bool(cfg.get("model", {}).get("adapt_for_small_inputs", True)),
        trainable_backbone=bool(cfg.get("model", {}).get("trainable_backbone", True)),
        libs_use_sobel=bool(cfg.get("libs", {}).get("use_sobel", True)),
        libs_use_fusion=bool(cfg.get("libs", {}).get("use_fusion", True)),
        libs_sobel_mode=str(cfg.get("libs", {}).get("sobel_mode", "magnitude")),
        libs_raw_use_conv=bool(cfg.get("libs", {}).get("raw_use_conv", False)),
    )


def _is_libs_model(model_name: str) -> bool:
    return model_name.startswith("libs_")


def _run_single_seed_model(
    cfg: Dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    device: torch.device,
    dataset_dir: str,
) -> Dict[str, Dict[str, float]]:
    set_global_seed(seed)

    loaders = get_medmnist_dataloaders(
        dataset_name=dataset_name,
        data_dir=cfg["paths"]["data_dir"],
        batch_size=int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"].get("num_workers", 0)),
        normalize=True,
        seed=seed,
        train_fraction=float(cfg.get("dataset", {}).get("train_fraction", 1.0)),
    )

    seed_dir = os.path.join(dataset_dir, model_name, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    model = _build_model(
        cfg=cfg,
        model_name=model_name,
        in_channels=loaders.in_channels,
        num_classes=loaders.num_classes,
    ).to(device)

    train_kwargs = _build_train_kwargs(cfg=cfg, loaders=loaders, device=device, save_dir=seed_dir)
    model, history, _ = train_model(
        model=model,
        model_name=model_name,
        **train_kwargs,
    )

    with open(os.path.join(seed_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(history), f, indent=2)

    full_metrics = evaluate_model(
        model=model,
        loader=loaders.test,
        device=device,
        save_path=os.path.join(seed_dir, "metrics.json"),
        use_geom_branch=True if _is_libs_model(model_name) else None,
    )

    results: Dict[str, Dict[str, float]] = {
        "full": _extract_metrics_for_aggregation(full_metrics)
    }

    if _is_libs_model(model_name):
        no_geom_metrics = evaluate_model(
            model=model,
            loader=loaders.test,
            device=device,
            save_path=os.path.join(seed_dir, "metrics_no_geom.json"),
            use_geom_branch=False,
        )
        results["no_geom"] = _extract_metrics_for_aggregation(no_geom_metrics)

        with open(os.path.join(seed_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "libs_full": results["full"],
                    "libs_no_geom": results["no_geom"],
                    "delta_full_minus_no_geom": {
                        m: float(results["full"][m] - results["no_geom"][m]) for m in SUMMARY_METRICS
                    },
                },
                f,
                indent=2,
            )

    return results


def _aggregate_dataset_results(
    dataset_name: str,
    model_names: List[str],
    seeds: List[int],
    run_results: Dict[str, Dict[int, Dict[str, Dict[str, float]]]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset": dataset_name,
        "models": model_names,
        "seeds": seeds,
        "metrics": SUMMARY_METRICS,
    }

    for model_name in model_names:
        seed_rows_full = [run_results[model_name][seed]["full"] for seed in seeds]
        model_payload: Dict[str, object] = {
            "full": _aggregate_seed_metrics(seed_rows_full),
        }

        if _is_libs_model(model_name):
            seed_rows_no_geom = [run_results[model_name][seed]["no_geom"] for seed in seeds]
            model_payload["no_geom"] = _aggregate_seed_metrics(seed_rows_no_geom)
            model_payload["ablation_delta_full_minus_no_geom"] = {
                metric_name: _summary(
                    [
                        float(run_results[model_name][seed]["full"][metric_name])
                        - float(run_results[model_name][seed]["no_geom"][metric_name])
                        for seed in seeds
                    ]
                )
                for metric_name in SUMMARY_METRICS
            }

        summary[model_name] = model_payload

    return summary


def run_experiments(cfg: Dict, device: torch.device) -> None:
    datasets = _resolve_datasets(cfg)
    seeds = _resolve_seeds(cfg)
    model_names = _validate_model_list(cfg)

    results_root = cfg["paths"]["results_dir"]
    os.makedirs(results_root, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Datasets: {datasets}")
    print(f"Models: {model_names}")
    print(f"Seeds: {seeds}")

    for dataset_name in datasets:
        dataset_dir = os.path.join(results_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        run_results: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {
            model_name: {} for model_name in model_names
        }

        for seed in seeds:
            for model_name in model_names:
                print(
                    f"\n=== dataset={dataset_name} | model={model_name} | seed={seed} ==="
                )
                run_results[model_name][seed] = _run_single_seed_model(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=seed,
                    device=device,
                    dataset_dir=dataset_dir,
                )

        summary = _aggregate_dataset_results(
            dataset_name=dataset_name,
            model_names=model_names,
            seeds=seeds,
            run_results=run_results,
        )

        summary_path = os.path.join(dataset_dir, "comparison_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardized 4-model LIBS experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = _resolve_device()
    run_experiments(cfg=cfg, device=device)


if __name__ == "__main__":
    main()
