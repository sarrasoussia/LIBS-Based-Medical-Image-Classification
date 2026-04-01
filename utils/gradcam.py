from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Minimal Grad-CAM implementation for CNN-style models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        selected = logits[torch.arange(logits.size(0), device=logits.device), class_idx]
        selected.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
