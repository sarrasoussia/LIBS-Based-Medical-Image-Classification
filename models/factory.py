from __future__ import annotations

import torch.nn as nn

from models.cnn_baseline import BaselineCNN
from models.ga_cnn_model import GACNN
from models.pretrained_models import GADenseNet121, PretrainedDenseNet


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    include_higher_order: bool,
    pretrained: bool,
    adapt_for_small_inputs: bool,
) -> nn.Module:
    name = model_name.lower()

    if name == "baseline_cnn":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "ga_cnn":
        return GACNN(
            in_channels=in_channels,
            num_classes=num_classes,
            include_higher_order=include_higher_order,
        )
    if name == "densenet121":
        return PretrainedDenseNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
        )
    if name == "ga_densenet121":
        return GADenseNet121(
            in_channels=in_channels,
            num_classes=num_classes,
            include_higher_order=include_higher_order,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
        )

    raise ValueError(
        "Unknown model_name='{}'. Supported: baseline_cnn, ga_cnn, densenet121, ga_densenet121".format(
            model_name
        )
    )
