from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_baseline import SimpleCNN
from models.pretrained_models import PretrainedDenseNet


class SobelFeatureModule(nn.Module):
    """Fixed, differentiable Sobel feature extractor.

    Input:  [B, C, H, W]
    Output: [B, C, H, W] for mode='magnitude', or [B, 2C, H, W] for mode='xy'
    """

    def __init__(self, in_channels: int, mode: str = "magnitude", eps: float = 1e-6) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.mode = str(mode).lower()
        self.eps = float(eps)
        if self.mode not in {"magnitude", "xy"}:
            raise ValueError("Sobel mode must be one of: magnitude, xy")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_kernel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_kernel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

        self.conv_x = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )
        self.conv_y = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=False,
        )

        with torch.no_grad():
            kx = self.sobel_kernel_x.repeat(self.in_channels, 1, 1, 1)
            ky = self.sobel_kernel_y.repeat(self.in_channels, 1, 1, 1)
            self.conv_x.weight.copy_(kx)
            self.conv_y.weight.copy_(ky)
        self.conv_x.weight.requires_grad_(False)
        self.conv_y.weight.requires_grad_(False)

        self.last_feature_mean = 0.0
        self.last_feature_std = 0.0

    @property
    def out_channels(self) -> int:
        return self.in_channels if self.mode == "magnitude" else 2 * self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        if self.mode == "xy":
            features = torch.cat([gx, gy], dim=1)
        else:
            features = torch.sqrt(gx.pow(2) + gy.pow(2) + self.eps)
        features = torch.tanh(features / 4.0)
        self.last_feature_mean = float(features.mean().item())
        self.last_feature_std = float(features.std(unbiased=False).item())
        return features


class RawFeatureBranch(nn.Module):
    """Raw input branch with identity or optional shallow conv processing."""

    def __init__(self, in_channels: int, use_conv: bool = False) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        if use_conv:
            self.block: nn.Module = nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Identity()

        self.last_feature_mean = 0.0
        self.last_feature_std = 0.0

    @property
    def out_channels(self) -> int:
        return self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        self.last_feature_mean = float(out.mean().item())
        self.last_feature_std = float(out.std(unbiased=False).item())
        return out


class LIBSFusionLayer(nn.Module):
    """Option-A learnable fusion: concat -> 1x1 conv -> BN -> ReLU."""

    def __init__(self, raw_channels: int, sobel_channels: int) -> None:
        super().__init__()
        self.raw_channels = int(raw_channels)
        self.sobel_channels = int(sobel_channels)
        self.conv = nn.Conv2d(
            in_channels=self.raw_channels + self.sobel_channels,
            out_channels=self.raw_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.raw_channels)
        self.relu = nn.ReLU(inplace=True)

        self.last_output_mean = 0.0
        self.last_output_std = 0.0

    def forward(self, raw_features: torch.Tensor, sobel_features: torch.Tensor) -> torch.Tensor:
        fused_input = torch.cat([raw_features, sobel_features], dim=1)
        fused = self.relu(self.bn(self.conv(fused_input)))
        self.last_output_mean = float(fused.mean().item())
        self.last_output_std = float(fused.std(unbiased=False).item())
        return fused

    def weight_grad_norm(self) -> float | None:
        if self.conv.weight.grad is None:
            return None
        return float(self.conv.weight.grad.detach().norm(2).item())

    def weight_analysis(self) -> Dict[str, float]:
        w = self.conv.weight.detach().abs()
        raw_w = w[:, : self.raw_channels, :, :]
        sobel_w = w[:, self.raw_channels :, :, :]
        raw_mean_abs = float(raw_w.mean().item())
        sobel_mean_abs = float(sobel_w.mean().item())
        ratio = sobel_mean_abs / max(raw_mean_abs, 1e-12)
        return {
            "raw_mean_abs_weight": raw_mean_abs,
            "sobel_mean_abs_weight": sobel_mean_abs,
            "sobel_to_raw_importance_ratio": ratio,
            # compatibility with existing GA analysis keys
            "ga_mean_abs_weight": sobel_mean_abs,
            "ga_to_raw_importance_ratio": ratio,
        }


class LIBSInputAdapter(nn.Module):
    """Reusable LIBS input adapter with ablation controls."""

    def __init__(
        self,
        in_channels: int,
        sobel_mode: str = "magnitude",
        raw_use_conv: bool = False,
        use_sobel: bool = True,
        use_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.raw_branch = RawFeatureBranch(in_channels=in_channels, use_conv=raw_use_conv)
        self.sobel_branch = SobelFeatureModule(in_channels=in_channels, mode=sobel_mode)
        self.fusion_layer = LIBSFusionLayer(
            raw_channels=self.raw_branch.out_channels,
            sobel_channels=self.sobel_branch.out_channels,
        )

        self.use_sobel = bool(use_sobel)
        self.use_fusion = bool(use_fusion)
        self._ablate_sobel = False
        self._last_fusion_output: torch.Tensor | None = None

        if not self.use_fusion and self.sobel_branch.out_channels != self.raw_branch.out_channels:
            raise ValueError(
                "When use_fusion=False, sobel output channels must match raw channels. "
                "Use sobel_mode='magnitude' for shape compatibility."
            )

    @property
    def out_channels(self) -> int:
        return self.raw_branch.out_channels

    def set_use_sobel(self, enabled: bool) -> None:
        self.use_sobel = bool(enabled)

    def set_use_fusion(self, enabled: bool) -> None:
        self.use_fusion = bool(enabled)

    def set_ga_ablation(self, enabled: bool) -> None:
        # compatibility with existing evaluation ablation hook
        self._ablate_sobel = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_features = self.raw_branch(x)
        sobel_features = self.sobel_branch(x)

        if self._ablate_sobel or not self.use_sobel:
            sobel_features = torch.zeros_like(sobel_features)

        if self.use_fusion and self.use_sobel:
            fused = self.fusion_layer(raw_features, sobel_features)
        elif self.use_sobel:
            fused = sobel_features
        else:
            fused = raw_features

        self._last_fusion_output = fused
        if fused.requires_grad:
            fused.retain_grad()
        return fused

    def raw_feature_stats(self) -> tuple[float, float]:
        return self.raw_branch.last_feature_mean, self.raw_branch.last_feature_std

    def sobel_feature_stats(self) -> tuple[float, float]:
        return self.sobel_branch.last_feature_mean, self.sobel_branch.last_feature_std

    def fusion_output_stats(self) -> tuple[float, float]:
        return self.fusion_layer.last_output_mean, self.fusion_layer.last_output_std

    def last_fusion_grad_norm(self) -> float | None:
        if self._last_fusion_output is None or self._last_fusion_output.grad is None:
            return None
        return float(self._last_fusion_output.grad.detach().norm(2).item())

    def fusion_weight_grad_norm(self) -> float | None:
        return self.fusion_layer.weight_grad_norm()

    def fusion_weight_analysis(self) -> Dict[str, float]:
        return self.fusion_layer.weight_analysis()

    def ga_feature_stats(self) -> tuple[float, float]:
        # compatibility with existing training logs using GA naming
        return self.sobel_feature_stats()


class LIBSModel(nn.Module):
    """Full LIBS wrapper: input -> (raw/sobel/fusion) -> backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        use_sobel: bool = True,
        use_fusion: bool = True,
        sobel_mode: str = "magnitude",
        raw_use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.input_adapter = LIBSInputAdapter(
            in_channels=in_channels,
            sobel_mode=sobel_mode,
            raw_use_conv=raw_use_conv,
            use_sobel=use_sobel,
            use_fusion=use_fusion,
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_input = self.input_adapter(x)
        return self.backbone(fused_input)

    def set_use_sobel(self, enabled: bool) -> None:
        self.input_adapter.set_use_sobel(enabled)

    def set_use_fusion(self, enabled: bool) -> None:
        self.input_adapter.set_use_fusion(enabled)

    def set_ga_ablation(self, enabled: bool) -> None:
        self.input_adapter.set_ga_ablation(enabled)

    def raw_feature_stats(self) -> tuple[float, float]:
        return self.input_adapter.raw_feature_stats()

    def sobel_feature_stats(self) -> tuple[float, float]:
        return self.input_adapter.sobel_feature_stats()

    def fusion_output_stats(self) -> tuple[float, float]:
        return self.input_adapter.fusion_output_stats()

    def ga_feature_stats(self) -> tuple[float, float]:
        return self.input_adapter.ga_feature_stats()

    def last_fusion_grad_norm(self) -> float | None:
        return self.input_adapter.last_fusion_grad_norm()

    def fusion_weight_grad_norm(self) -> float | None:
        return self.input_adapter.fusion_weight_grad_norm()

    def fusion_weight_analysis(self) -> Dict[str, float]:
        return self.input_adapter.fusion_weight_analysis()

    def gradcam_target_layer(self) -> nn.Module:
        if hasattr(self.backbone, "gradcam_target_layer"):
            return self.backbone.gradcam_target_layer()
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "features"):
            return self.backbone.model.features[-1]
        if hasattr(self.backbone, "features"):
            return self.backbone.features[-1]
        raise ValueError("Could not infer Grad-CAM target layer for LIBS backbone")


class LIBSCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_sobel: bool = True,
        use_fusion: bool = True,
        sobel_mode: str = "magnitude",
        raw_use_conv: bool = False,
    ) -> None:
        super().__init__()
        backbone = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        self.model = LIBSModel(
            backbone=backbone,
            in_channels=in_channels,
            use_sobel=use_sobel,
            use_fusion=use_fusion,
            sobel_mode=sobel_mode,
            raw_use_conv=raw_use_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class LIBSDenseNet121(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = True,
        adapt_for_small_inputs: bool = True,
        trainable_backbone: bool = True,
        use_sobel: bool = True,
        use_fusion: bool = True,
        sobel_mode: str = "magnitude",
        raw_use_conv: bool = False,
    ) -> None:
        super().__init__()
        backbone = PretrainedDenseNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
        )
        self.model = LIBSModel(
            backbone=backbone,
            in_channels=in_channels,
            use_sobel=use_sobel,
            use_fusion=use_fusion,
            sobel_mode=sobel_mode,
            raw_use_conv=raw_use_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
