# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# FaceNet (InceptionResnetV1) identity encoder — Phase R1 stand-in.
#
# InceptionResnetV1 pretrained on VGGFace2 outputs a 512-D L2-normalized
# identity embedding. It is not ArcFace, but its embedding geometry and
# the way it is used downstream (cosine similarity for verification)
# are close enough that R1 can validate the loss and PGD kernel against
# it. Phase R2 replaces this with proper ArcFace + MagFace + AdaFace as
# documented in Documentation/models/recognizers.md.

"""FaceNet identity encoder surrogate (Phase R1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec

__all__ = ["Facenet"]

_INPUT_SIZE = 160


class Facenet(nn.Module):
    """Differentiable InceptionResnetV1 identity encoder.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="facenet-vggface2",
        family="recognizers",
        input_resolution=_INPUT_SIZE,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        from facenet_pytorch import InceptionResnetV1

        self._device = torch.device(device)
        self._net = InceptionResnetV1(pretrained="vggface2").to(self._device)
        self._net.eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> TargetOutputs:
        """Run the encoder on ``image``.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``,
                RGB order. Inputs are resized to :data:`_INPUT_SIZE`.

        Returns:
            :class:`TargetOutputs` with ``embedding`` set to an
            L2-normalized ``(N, 512)`` identity vector.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got shape {tuple(image.shape)}."
            raise ValueError(msg)

        resized = (
            image
            if image.shape[-2:] == (_INPUT_SIZE, _INPUT_SIZE)
            else F.interpolate(image, size=_INPUT_SIZE, mode="bilinear", align_corners=False)
        )
        # facenet-pytorch's InceptionResnetV1 expects [-1, 1] normalized.
        normalized = resized.sub(0.5).mul(2.0)
        raw = self._net(normalized)
        embedding = F.normalize(raw, dim=-1)
        return TargetOutputs(embedding=embedding)

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]
