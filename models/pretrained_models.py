from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from models.ga_representation import GARepresentation


class PretrainedDenseNet(nn.Module):
    """DenseNet121 wrapper with configurable input channels and classifier head."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pretrained: bool = True,
        adapt_for_small_inputs: bool = True,
        trainable_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        if adapt_for_small_inputs:
            self.model.features.conv0 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.model.features.pool0 = nn.Identity()
        else:
            self.model.features.conv0 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        if not trainable_backbone:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False
            for parameter in self.model.classifier.parameters():
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def gradcam_target_layer(self) -> nn.Module:
        return self.model.features.denseblock4


class GADenseNet121(nn.Module):
    """DenseNet121 preceded by GA-inspired input representation."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        include_higher_order: bool = True,
        representation_mode: str | None = None,
        pretrained: bool = True,
        adapt_for_small_inputs: bool = True,
        trainable_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.ga_encoder = GARepresentation(
            in_channels=in_channels,
            include_higher_order=include_higher_order,
            representation_mode=representation_mode,
        )
        self.backbone = PretrainedDenseNet(
            in_channels=self.ga_encoder.out_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.ga_encoder(x))

    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.gradcam_target_layer()
