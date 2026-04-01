from __future__ import annotations

import torch
import torch.nn as nn

from models.cnn_baseline import SimpleCNN
from models.ga_representation import GARepresentation


class GACNN(nn.Module):
    """Pipeline: Input -> GA representation -> CNN classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        include_higher_order: bool = True,
    ) -> None:
        super().__init__()
        self.ga_encoder = GARepresentation(
            in_channels=in_channels,
            include_higher_order=include_higher_order,
        )
        self.classifier = SimpleCNN(
            in_channels=self.ga_encoder.out_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ga_encoder(x)
        return self.classifier(x)
