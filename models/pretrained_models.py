from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


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
