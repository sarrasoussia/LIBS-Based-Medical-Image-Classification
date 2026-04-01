from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GARepresentation(nn.Module):
    """GA-inspired input encoding using gradient-based geometric components.

    Formal definition for each channel c and pixel (u, v):

    Let I_c(u, v) be the normalized input image.

    Sobel kernels:
        Sx = [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]

        Sy = [[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]

    Vector-like components:
        Gx_c = I_c * Sx
        Gy_c = I_c * Sy

    Higher-order components (optional):
        M_c = sqrt(Gx_c^2 + Gy_c^2 + eps)
        Theta_c = atan2(Gy_c, Gx_c + eps) / pi

    Concatenation:
        include_higher_order = False:
            F = concat[I, Gx, Gy], shape (B, 3C, H, W)

        include_higher_order = True:
            F = concat[I, Gx, Gy, M, Theta], shape (B, 5C, H, W)

    Components:
    1) Scalar part: original image intensities
    2) Vector-like parts: Sobel x-gradient and y-gradient
    3) Optional higher-order terms: gradient magnitude and orientation

    Output is a concatenated multivector-inspired tensor.
    """

    def __init__(self, in_channels: int, include_higher_order: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.include_higher_order = include_higher_order

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    @property
    def out_channels(self) -> int:
        base_multiplier = 3  # scalar + grad_x + grad_y
        if self.include_higher_order:
            base_multiplier += 2  # magnitude + orientation
        return self.in_channels * base_multiplier

    def _apply_sobel(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        repeated_kernel = kernel.repeat(self.in_channels, 1, 1, 1).to(x.device).to(x.dtype)
        return F.conv2d(x, repeated_kernel, padding=1, groups=self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = self._apply_sobel(x, self.sobel_x)
        grad_y = self._apply_sobel(x, self.sobel_y)

        components = [x, grad_x, grad_y]

        if self.include_higher_order:
            magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
            orientation = torch.atan2(grad_y, grad_x + 1e-8) / math.pi
            components.extend([magnitude, orientation])

        return torch.cat(components, dim=1)
