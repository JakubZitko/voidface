# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# MTCNN P-Net detector surrogate (Phase R1 stand-in).
#
# MTCNN is a three-stage cascade (P-Net -> R-Net -> O-Net). The full
# cascade is not end-to-end differentiable because the pyramid
# resampling and NMS between stages are not. But P-Net alone is a
# fully-convolutional differentiable face detector, and its output is
# the choke-point signal we need to suppress in Phase R1.
#
# This target is a stand-in. Phase R2 replaces it with a proper
# RetinaFace + SCRFD + YuNet + MTCNN ensemble as documented in
# Documentation/models/detectors.md.

"""MTCNN P-Net face-detector surrogate (Phase R1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec

__all__ = ["MtcnnPnet"]


class MtcnnPnet(nn.Module):
    """Differentiable P-Net face detector.

    P-Net produces a per-pixel face-vs-background 2-way softmax score
    over a fully convolutional receptive field. The face-present
    channel is ``TargetOutputs.logits[..., 1]``; the R1 suppression
    loss lives on that tensor.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="mtcnn-pnet",
        family="detectors",
        input_resolution=None,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        from facenet_pytorch.models.mtcnn import PNet

        self._device = torch.device(device)
        self._net = PNet().to(self._device)
        self._net.eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> TargetOutputs:
        """Run P-Net on ``image``.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``,
                RGB order. Any resolution >= 12x12 pixels.

        Returns:
            :class:`TargetOutputs` with ``logits`` set to per-pixel
            softmax face confidence of shape ``(N, H', W', 2)`` where
            ``H' = H // 2 - 5`` and ``W' = W // 2 - 5``.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got shape {tuple(image.shape)}."
            raise ValueError(msg)

        # facenet-pytorch's PNet expects images in [-1, 1] and NCHW.
        # We produce that from the canonical [0, 1] input.
        normalized = image.sub(0.5).mul(2.0)
        reg, cls = self._net(normalized)
        # cls: (N, 2, H', W'). Softmax over the class dim, then permute
        # to a channels-last layout that matches the RetinaFace surrogate
        # so the R2 replacement can be dropped in.
        cls = cls.softmax(dim=1)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        return TargetOutputs(logits=cls, aux={"bbox_regression": reg})

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]
