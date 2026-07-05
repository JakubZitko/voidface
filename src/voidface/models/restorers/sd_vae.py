# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Stable Diffusion 1.5 VAE round-trip restorer.
#
# This is the first "real" bilevel restorer target in Voidface. It
# passes the perturbed image through ``decode(encode(x))`` of the SD
# 1.5 VAE, which strips high-frequency information the VAE was not
# trained to preserve — a strong stand-in for the "diffusion
# purification" strip attack (DiffPure, IMPRESS) that broke every
# prior adversarial-cloaking tool.
#
# The restorer shares the underlying AutoencoderKL with the encoder
# target :class:`voidface.models.vaes.sd15.Sd15Vae` — no duplicated
# weights, no second download.
#
# See Documentation/training/bilevel-adversarial.md.

"""Stable Diffusion 1.5 VAE round-trip restorer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F

from voidface.models.restorers.base import RestorerSpec

if TYPE_CHECKING:
    from torch import Tensor

    from voidface.models.vaes.sd15 import Sd15Vae

__all__ = ["Sd15VaeRestorer"]

_ROUNDING = 8
_MAX_SIDE = 512


class Sd15VaeRestorer:
    """Encode-then-decode round-trip through the SD 1.5 VAE.

    Semantically: given ``x``, produce ``decode(encode(x))``. Because
    the VAE is not lossless, the round-trip strips information the
    encoder could not represent — in particular, the high-frequency
    adversarial noise our own perturbation might place there. Training
    with this restorer in the loop forces the generator to place its
    disruptive signal in components that survive VAE compression.

    This is not a face-restoration model (GFPGAN, CodeFormer,
    Real-ESRGAN). It approximates the "diffusion purification" strip
    the attacker's pipeline may run — a different flavor of the same
    bilevel objective. Real face restorers land alongside this one in
    Phase R4.
    """

    spec = RestorerSpec(
        name="sd15-vae-roundtrip",
        expects_face_crop=False,
        weight_url=None,
    )

    def __init__(self, encoder: Sd15Vae) -> None:
        self._encoder = encoder

    def __call__(self, image: Tensor) -> Tensor:
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got {tuple(image.shape)}."
            raise ValueError(msg)

        original_hw = image.shape[-2:]
        target_hw = _round_and_cap(original_hw[0], original_hw[1])
        resized = (
            image
            if image.shape[-2:] == target_hw
            else F.interpolate(image, size=target_hw, mode="bilinear", align_corners=False)
        )

        vae = self._encoder.underlying_vae
        normalized = resized.sub(0.5).mul(2.0)                        # -> [-1, 1]
        latent = vae.encode(normalized).latent_dist.mean
        decoded = vae.decode(latent).sample                           # -> [-1, 1]
        restored = decoded.add(1.0).div(2.0).clamp(0.0, 1.0)          # -> [0, 1]

        if restored.shape[-2:] != original_hw:
            restored = F.interpolate(
                restored, size=original_hw, mode="bilinear", align_corners=False
            )
        return restored


def _round_and_cap(height: int, width: int) -> tuple[int, int]:
    h = min(_MAX_SIDE, max(_ROUNDING, (height // _ROUNDING) * _ROUNDING))
    w = min(_MAX_SIDE, max(_ROUNDING, (width // _ROUNDING) * _ROUNDING))
    return h, w
