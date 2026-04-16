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
    ga_enabled = bool(cfg.get("ga", {}).get("enabled", model_name.startswith("ga_")))

    if model_name == "densenet121":
        return ["ga_densenet121" if ga_enabled else "densenet121"]
    if model_name in {"baseline_cnn", "ga_cnn", "ga_densenet121"}:
        return [model_name]
    raise ValueError(
        "Unsupported model.name='{}'. Use one of: baseline_cnn, ga_cnn, densenet121".format(
            model_name
        )
    )


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
        if torch.backends.mps.is_available() and not use_mps:
            print("Using CPU because USE_MPS=0 was explicitly requested.")
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
    print(
        "Data sanity | "
        f"dataset={dataset_name} "
        f"train_size={len(loaders.train.dataset)} val_size={len(loaders.val.dataset)} test_size={len(loaders.test.dataset)} "
        f"train_batches={len(loaders.train)} val_batches={len(loaders.val)} test_batches={len(loaders.test)}"
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
    }

    model_names = _resolve_model_names(cfg)
    comparison = {}

    for model_name in model_names:
        print(f"\n=== Running model: {model_name} on dataset: {dataset_name} ===")
        model = build_model(
            model_name=model_name,
            in_channels=loaders.in_channels,
            num_classes=loaders.num_classes,
            include_higher_order=cfg["ga"].get("include_higher_order", True),
            representation_mode=cfg["ga"].get("representation_mode"),
            pretrained=cfg.get("model", {}).get("pretrained", True),
            adapt_for_small_inputs=cfg.get("model", {}).get("adapt_for_small_inputs", True),
        ).to(device)

        model, hist, _ = train_model(
            model=model,
            model_name=model_name,
            **common_train_kwargs,
        )
        plot_training_curves(
            history=asdict(hist),
            save_path=os.path.join(outputs_dir, f"{model_name}_training_curves.png"),
            title=model_name,
        )

        metrics = evaluate_model(
            model=model,
            loader=loaders.test,
            device=device,
            save_path=os.path.join(outputs_dir, f"{model_name}_metrics.json"),
        )
        plot_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_names=loaders.class_names,
            save_path=os.path.join(outputs_dir, f"{model_name}_confusion_matrix.png"),
            title=f"{model_name} - Confusion Matrix",
        )
        _run_gradcam(
            model=model,
            loader=loaders.test,
            device=device,
            save_path=os.path.join(outputs_dir, f"{model_name}_gradcam.png"),
            title=f"{model_name} Grad-CAM",
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
