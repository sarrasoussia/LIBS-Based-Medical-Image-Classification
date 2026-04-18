from __future__ import annotations

import torch.nn as nn

from models.cnn_baseline import BaselineCNN
from models.libs_model import LIBSCNN, LIBSDenseNet121
from models.pretrained_models import PretrainedDenseNet


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    pretrained: bool = True,
    adapt_for_small_inputs: bool = True,
    trainable_backbone: bool = True,
    libs_use_sobel: bool = True,
    libs_use_fusion: bool = True,
    libs_sobel_mode: str = "magnitude",
    libs_raw_use_conv: bool = False,
) -> nn.Module:
    name = model_name.lower()

    if name == "baseline_cnn":
        return BaselineCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "densenet121":
        return PretrainedDenseNet(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
        )
    if name == "libs_cnn":
        return LIBSCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            use_sobel=libs_use_sobel,
            use_fusion=libs_use_fusion,
            sobel_mode=libs_sobel_mode,
            raw_use_conv=libs_raw_use_conv,
        )
    if name == "libs_densenet121":
        return LIBSDenseNet121(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            adapt_for_small_inputs=adapt_for_small_inputs,
            trainable_backbone=trainable_backbone,
            use_sobel=libs_use_sobel,
            use_fusion=libs_use_fusion,
            sobel_mode=libs_sobel_mode,
            raw_use_conv=libs_raw_use_conv,
        )

    raise ValueError(
        "Unknown model_name='{}'. Supported: baseline_cnn, densenet121, libs_cnn, libs_densenet121".format(
            model_name
        )
    )
