from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import yaml

from data.pathmnist import denormalize_image, get_medmnist_dataloaders, labels_to_long, set_global_seed
from models.factory import build_model
from training.evaluate import evaluate_model
from training.study import run_study
from training.train import train_model
from utils.gradcam import GradCAM
from utils.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_curves,
    save_gradcam_grid,
)
from utils.report_figures import save_pipeline_diagram


def _load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "gradcam_target_layer"):
        return model.gradcam_target_layer()  # type: ignore[no-any-return]
    if hasattr(model, "model") and hasattr(model.model, "features"):
        return model.model.features[-1]
    if hasattr(model, "classifier") and hasattr(model.classifier, "features"):
        return model.classifier.features[-1]
    raise ValueError("Could not infer Grad-CAM target layer for model")


def _resolve_model_names(cfg: Dict) -> List[str]:
    runs = cfg.get("experiments", {}).get("run_models")
    if runs:
        return [str(name).lower() for name in runs]

    model_name = str(cfg.get("model", {}).get("name", "baseline_cnn")).lower()

    if model_name in {
        "baseline_cnn",
        "densenet121",
        "libs_cnn",
        "libs_densenet121",
    }:
        return [model_name]
    raise ValueError(
        "Unsupported model.name='{}'. Use one of: baseline_cnn, densenet121, libs_cnn, libs_densenet121".format(
            model_name
        )
    )


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _display_model_name(model_name: str) -> str:
    mapping = {
        "baseline_cnn": "Baseline CNN",
        "libs_cnn": "LIBS-CNN",
        "densenet121": "DenseNet121",
        "libs_densenet121": "LIBS-DenseNet121",
    }
    return mapping.get(model_name, model_name)


def _run_gradcam(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    save_path: str,
    title: str,
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images[:12].to(device)
    labels = labels_to_long(labels[:12]).cpu().numpy()

    target_layer = _select_target_layer(model)
    cam = GradCAM(model, target_layer)
    try:
        heatmaps = cam.generate(images).detach().cpu().numpy()
        preds = torch.argmax(model(images), dim=1).detach().cpu().numpy()
    finally:
        cam.close()

    vis_images = (
        denormalize_image(images.detach().cpu())
        .numpy()
        .astype(np.float32)
    )
    save_gradcam_grid(
        images=vis_images,
        heatmaps=heatmaps,
        preds=preds,
        labels=labels,
        save_path=save_path,
        title=title,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical image experiments: baseline, GA, and DenseNet121")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="Run multi-seed/multi-dataset study mode",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    set_global_seed(cfg["seed"])

    use_mps = os.environ.get("USE_MPS", "1") != "0"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.study or cfg.get("study", {}).get("enabled", False):
        run_study(cfg=cfg, device=device)
        return

    dataset_name = cfg.get("dataset", {}).get("name", "pathmnist")
    base_results_dir = cfg["paths"]["results_dir"]
    outputs_dir = os.path.join(base_results_dir, dataset_name)
    ckpt_dir = os.path.join(outputs_dir, "checkpoints")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_pipeline_diagram(os.path.join(outputs_dir, "pipeline_baseline_vs_libs.png"))

    train_fraction = float(cfg.get("dataset", {}).get("train_fraction", 1.0))
    loaders = get_medmnist_dataloaders(
        dataset_name=dataset_name,
        data_dir=cfg["paths"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        normalize=True,
        seed=cfg["seed"],
        train_fraction=train_fraction,
    )
    common_train_kwargs = {
        "train_loader": loaders.train,
        "val_loader": loaders.val,
        "device": device,
        "save_dir": ckpt_dir,
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
        "auto_resume": bool(cfg.get("training", {}).get("auto_resume", False)),
    }

    model_names = _resolve_model_names(cfg)
    skip_completed = bool(cfg.get("experiments", {}).get("skip_completed", False))
    comparison = {}

    for model_name in model_names:
        artifact_name = _artifact_model_name(model_name)
        display_name = _display_model_name(model_name)
        metrics_path = os.path.join(outputs_dir, f"{artifact_name}_metrics.json")
        if skip_completed and os.path.exists(metrics_path):
            print(f"\n=== Skipping already completed model: {model_name} ===")
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
            comparison[model_name] = {
                k: existing_metrics[k] for k in existing_metrics if k != "confusion_matrix"
            }
            continue

        print(f"\n=== Running model: {model_name} on dataset: {dataset_name} ===")
        model = build_model(
            model_name=model_name,
            in_channels=loaders.in_channels,
            num_classes=loaders.num_classes,
            include_higher_order=cfg["ga"].get("include_higher_order", True),
            representation_mode=cfg["ga"].get("representation_mode"),
            pretrained=cfg.get("model", {}).get("pretrained", True),
            adapt_for_small_inputs=cfg.get("model", {}).get("adapt_for_small_inputs", True),
            ga_normalize_output=cfg.get("ga", {}).get("normalize_output", False),
            libs_use_sobel=bool(cfg.get("libs", {}).get("use_sobel", True)),
            libs_use_fusion=bool(cfg.get("libs", {}).get("use_fusion", True)),
            libs_sobel_mode=str(cfg.get("libs", {}).get("sobel_mode", "magnitude")),
            libs_raw_use_conv=bool(cfg.get("libs", {}).get("raw_use_conv", False)),
        ).to(device)
        model, hist, _ = train_model(
            model=model,
            model_name=model_name,
            **common_train_kwargs,
        )
        plot_training_curves(
            history=asdict(hist),
            save_path=os.path.join(outputs_dir, f"{artifact_name}_training_curves.png"),
            title=display_name,
        )

        metrics = evaluate_model(
            model=model,
            loader=loaders.test,
            device=device,
            save_path=metrics_path,
        )
        plot_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_names=loaders.class_names,
            save_path=os.path.join(outputs_dir, f"{artifact_name}_confusion_matrix.png"),
            title=f"{display_name} - Confusion Matrix",
        )
        _run_gradcam(
            model=model,
            loader=loaders.test,
            device=device,
            save_path=os.path.join(outputs_dir, f"{artifact_name}_gradcam.png"),
            title=f"{display_name} Grad-CAM",
        )

        comparison[model_name] = {k: metrics[k] for k in metrics if k != "confusion_matrix"}

    with open(os.path.join(outputs_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    plot_model_comparison(
        results=comparison,
        save_path=os.path.join(outputs_dir, "model_comparison.png"),
    )

    print("Experiment complete. Results saved in:", outputs_dir)


if __name__ == "__main__":
    main()
