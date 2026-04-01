from __future__ import annotations

from dataclasses import dataclass
from typing import List

import medmnist
import numpy as np
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import PCAM
from torchvision import transforms


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: List[str]
    in_channels: int
    num_classes: int


def build_medmnist_transforms(
    n_channels: int,
    normalize: bool = True,
) -> transforms.Compose:
    """Build torchvision transforms for MedMNIST datasets.

    We use a simple and consistent normalization setup for all experiments to
    preserve fairness.
    """
    transform_steps = [transforms.ToTensor()]
    if normalize:
        mean = [0.5] * n_channels
        std = [0.5] * n_channels
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


def _build_class_names(info_dict: dict) -> List[str]:
    labels = info_dict["label"]
    return [labels[str(i)] for i in range(len(labels))]


def get_medmnist_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    normalize: bool = True,
    seed: int = 42,
    train_fraction: float = 1.0,
) -> DataLoaders:
    """Create train/val/test loaders for a MedMNIST dataset or PCam.

    Note: pin_memory is disabled on macOS with MPS to avoid device compatibility issues.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "pcam":
        transform = build_medmnist_transforms(n_channels=3, normalize=normalize)
        train_dataset = PCAM(root=data_dir, split="train", transform=transform, download=True)
        val_dataset = PCAM(root=data_dir, split="val", transform=transform, download=True)
        test_dataset = PCAM(root=data_dir, split="test", transform=transform, download=True)
        class_names = ["normal", "tumor"]
        in_channels = 3
        num_classes = 2
    elif dataset_name in INFO:
        info = INFO[dataset_name]
        transform = build_medmnist_transforms(
            n_channels=info["n_channels"],
            normalize=normalize,
        )
        dataset_cls = getattr(medmnist, info["python_class"])

        train_dataset = dataset_cls(
            split="train", transform=transform, download=True, root=data_dir
        )
        val_dataset = dataset_cls(split="val", transform=transform, download=True, root=data_dir)
        test_dataset = dataset_cls(
            split="test", transform=transform, download=True, root=data_dir
        )
        class_names = _build_class_names(info)
        in_channels = info["n_channels"]
        num_classes = len(class_names)
    else:
        available = ", ".join(sorted(INFO.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available MedMNIST: {available}, plus pcam"
        )

    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")
    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        subset_size = max(1, int(len(train_dataset) * train_fraction))
        subset_indices = rng.choice(len(train_dataset), size=subset_size, replace=False)
        train_dataset = Subset(train_dataset, subset_indices.tolist())

    generator = torch.Generator()
    generator.manual_seed(seed)

    use_pin_memory = not torch.backends.mps.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=class_names,
        in_channels=in_channels,
        num_classes=num_classes,
    )


def get_pathmnist_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    normalize: bool = True,
    seed: int = 42,
    train_fraction: float = 1.0,
) -> DataLoaders:
    """Backward-compatible wrapper for PathMNIST-specific experiments."""
    return get_medmnist_dataloaders(
        dataset_name="pathmnist",
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        seed=seed,
        train_fraction=train_fraction,
    )


def labels_to_long(labels: torch.Tensor) -> torch.Tensor:
    """Convert MedMNIST labels shape [B, 1] to [B] long tensor."""
    return labels.view(-1).long()


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor back to [0, 1] range for visualization."""
    if image.ndim == 3:
        channels = image.shape[0]
        view_shape = (channels, 1, 1)
    elif image.ndim == 4:
        channels = image.shape[1]
        view_shape = (1, channels, 1, 1)
    else:
        raise ValueError(
            "denormalize_image expects tensor of shape (C,H,W) or (B,C,H,W)."
        )

    mean = torch.tensor([0.5] * channels, device=image.device, dtype=image.dtype).view(
        *view_shape
    )
    std = torch.tensor([0.5] * channels, device=image.device, dtype=image.dtype).view(
        *view_shape
    )
    img = image * std + mean
    return torch.clamp(img, 0.0, 1.0)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
