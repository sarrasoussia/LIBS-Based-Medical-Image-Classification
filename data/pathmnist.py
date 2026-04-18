from __future__ import annotations

from dataclasses import dataclass
import random
import time
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


def _to_float_tensor_01(image) -> torch.Tensor:
    """Convert image-like input to float32 CHW tensor in [0, 1]."""
    if torch.is_tensor(image):
        t = image.detach().clone().to(torch.float32)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] not in (1, 3):
            # likely HWC
            t = t.permute(2, 0, 1)
        if t.max().item() > 1.0:
            t = t / 255.0
        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
        return t.contiguous()

    # Make an explicit writable copy to avoid undefined behavior warnings when
    # downstream transforms create derived tensors.
    arr = np.array(image, copy=True)
    if arr.ndim == 2:
        arr = arr[..., None]
    t = torch.from_numpy(arr).to(torch.float32).permute(2, 0, 1)
    if t.max().item() > 1.0:
        t = t / 255.0
    # Defensive sanitization for any upstream corrupted samples.
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    return t.contiguous()


def build_medmnist_transforms(
    n_channels: int,
    normalize: bool = True,
) -> transforms.Compose:
    """Build torchvision transforms for MedMNIST datasets.

    We use a simple and consistent normalization setup for all experiments to
    preserve fairness.
    """
    transform_steps = [transforms.Lambda(_to_float_tensor_01)]
    if normalize:
        mean = [0.5] * n_channels
        std = [0.5] * n_channels
        transform_steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_steps)


def _build_class_names(info_dict: dict) -> List[str]:
    labels = info_dict["label"]
    return [labels[str(i)] for i in range(len(labels))]


def _target_to_class_index(label) -> int:
    """Convert MedMNIST/PCAM labels to a scalar class index safely.

    MedMNIST labels can come as numpy arrays of shape (1,). We always squeeze
    and convert to Python int here so the DataLoader collate step builds a clean
    integer tensor batch.
    """
    arr = np.asarray(label).squeeze()
    if np.asarray(arr).size != 1:
        raise ValueError(f"Expected scalar label after squeeze, got shape={np.asarray(label).shape}")
    return int(arr)


def get_medmnist_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
    normalize: bool = True,
    seed: int = 42,
    train_fraction: float = 1.0,
    download_retries: int = 3,
    retry_delay_seconds: float = 3.0,
) -> DataLoaders:
    """Create train/val/test loaders for a MedMNIST dataset or PCam.

    Note: pin_memory is disabled on macOS with MPS to avoid device compatibility issues.
    """
    dataset_name = dataset_name.lower()

    def _init_with_retry(factory, split_name: str):
        last_error: Exception | None = None
        attempts = max(1, int(download_retries))
        for attempt in range(1, attempts + 1):
            try:
                return factory(download=True)
            except Exception as exc:
                last_error = exc
                # If data is already present locally, try without download.
                try:
                    return factory(download=False)
                except Exception:
                    pass

                if attempt < attempts:
                    print(
                        f"[data] Download failed for dataset='{dataset_name}' split='{split_name}' "
                        f"(attempt {attempt}/{attempts}): {exc}. Retrying in {retry_delay_seconds:.1f}s..."
                    )
                    time.sleep(max(0.0, float(retry_delay_seconds)))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected dataset initialization failure.")

    if dataset_name == "pcam":
        transform = build_medmnist_transforms(n_channels=3, normalize=normalize)
        train_dataset = _init_with_retry(
            lambda download: PCAM(
                root=data_dir,
                split="train",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
            ),
            split_name="train",
        )
        val_dataset = _init_with_retry(
            lambda download: PCAM(
                root=data_dir,
                split="val",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
            ),
            split_name="val",
        )
        test_dataset = _init_with_retry(
            lambda download: PCAM(
                root=data_dir,
                split="test",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
            ),
            split_name="test",
        )
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

        train_dataset = _init_with_retry(
            lambda download: dataset_cls(
                split="train",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
                root=data_dir,
            ),
            split_name="train",
        )
        val_dataset = _init_with_retry(
            lambda download: dataset_cls(
                split="val",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
                root=data_dir,
            ),
            split_name="val",
        )
        test_dataset = _init_with_retry(
            lambda download: dataset_cls(
                split="test",
                transform=transform,
                target_transform=_target_to_class_index,
                download=download,
                root=data_dir,
            ),
            split_name="test",
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

    loader_common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_common_kwargs["persistent_workers"] = True
        loader_common_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_common_kwargs,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_common_kwargs,
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
    """Convert labels to class-index tensor shape [B] with strict checks."""
    if torch.is_tensor(labels):
        # Avoid unsafe reconstruction paths; keep a detached copy.
        out = labels.clone().detach()
    else:
        out = torch.as_tensor(labels)

    out = out.squeeze()
    if out.ndim == 0:
        out = out.unsqueeze(0)
    elif out.ndim > 1:
        # Keep one class index per sample if an extra singleton/non-singleton axis appears.
        out = out.reshape(out.shape[0], -1)[:, 0]

    if out.is_floating_point():
        if not torch.allclose(out, out.round()):
            raise ValueError("Label tensor contains non-integer floating values.")
        out = out.round()

    return out.to(torch.int64).contiguous()


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
