from __future__ import annotations

import torch.nn as nn

from models.cnn_baseline import BaselineCNN
from models.ga_cnn_model import GACNN
from models.pretrained_models import GADenseNet121, PretrainedDenseNet


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    include_higher_order: bool = True,
    representation_mode: str | None = None,
    pretrained: bool = True,
    adapt_for_small_inputs: bool = True,
    trainable_backbone: bool = True,
) -> nn.Module:
    name = model_name.lower()

    if name == "baseline_cnn":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "ga_cnn":
        return GACNN(
            in_channels=in_channels,
            num_classes=num_classes,
            include_higher_order=include_higher_order,
            representation_mode=representation_mode,
        )
    if name == "densenet121":
        return PretrainedDenseNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
        )
    if name == "ga_densenet121":
        return GADenseNet121(
            in_channels=in_channels,
            num_classes=num_classes,
            include_higher_order=include_higher_order,
            representation_mode=representation_mode,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
        )

    raise ValueError(
        "Unknown model_name='{}'. Supported: baseline_cnn, ga_cnn, densenet121, ga_densenet121".format(
            model_name
        )
    )
