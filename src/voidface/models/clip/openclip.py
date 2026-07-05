# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# CLIP-family image-encoder surrogate.
#
# Every 2024-2026 personalization pipeline (IP-Adapter, InstantID,
# PhotoMaker, PuLID) conditions on a CLIP-family image embedding.
# The attacker's actual pipeline usually reaches for ViT-H/14 laion2b;
# we ship ViT-B/32 as a Phase-R4 bootstrap because the smaller model
# fits Intel-Mac memory limits and validates the attack shape. Ensemble
# transfer numbers under low ViT-B/32 weight are honest (20-40% to
# H-scale), so the CLI defaults keep this target at a modest weight.
#
# See Documentation/models/clip.md.
#
# We use transformers.CLIPVisionModel rather than open_clip_torch to
# avoid pulling timm + ftfy + regex + sentencepiece into the runtime.
# transformers is heavier by download but shares HF hub caching with
# our VAE and RetinaFace paths.

"""OpenAI CLIP ViT-B/32 image-encoder surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec

__all__ = ["OpenClip"]

_INPUT_SIZE = 224
_MODEL_ID = "openai/clip-vit-base-patch32"
# CLIP image preprocessing constants — trained with this exact mean/std.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OpenClip(nn.Module):
    """Differentiable CLIP ViT-B/32 image encoder.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="openclip-vit-b-32",
        family="clip",
        input_resolution=_INPUT_SIZE,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self._device = torch.device(device)
        self._net = CLIPVisionModel.from_pretrained(_MODEL_ID).to(self._device)
        self._net.eval()
        for parameter in self._net.parameters():
            parameter.requires_grad_(False)

        mean = torch.tensor(_CLIP_MEAN, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=self._device).view(1, 3, 1, 1)
        self.register_buffer("_pixel_mean", mean, persistent=False)
        self.register_buffer("_pixel_std", std, persistent=False)

    def forward(self, image: Tensor) -> TargetOutputs:
        """Encode ``image`` into the CLIP pooled embedding.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``,
                RGB order. Antialias-resized to ``_INPUT_SIZE``.

        Returns:
            :class:`TargetOutputs` with ``embedding`` set to the
            L2-normalized ``(N, 768)`` pooled CLS output.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got shape {tuple(image.shape)}."
            raise ValueError(msg)

        resized = (
            image
            if image.shape[-2:] == (_INPUT_SIZE, _INPUT_SIZE)
            else F.interpolate(
                image,
                size=_INPUT_SIZE,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
        # Standard CLIP normalization. Input already RGB [0, 1].
        normalized = (resized - self._pixel_mean) / self._pixel_std
        outputs = self._net(pixel_values=normalized)
        # pooler_output shape: (N, hidden_size). For ViT-B/32, hidden_size=768.
        raw = outputs.pooler_output
        embedding = F.normalize(raw, dim=-1)
        return TargetOutputs(embedding=embedding)

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]
