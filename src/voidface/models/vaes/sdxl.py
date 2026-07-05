# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# SDXL VAE encoder surrogate.
#
# SDXL is used by every 2024-2026 personalization pipeline: IP-Adapter,
# InstantID, PhotoMaker, PuLID, Kolors, HunyuanDiT. The VAE has the
# same 4-channel latent shape as SD 1.5 but different weights and a
# different scaling factor (0.13025 vs 0.18215). PhotoGuard-style
# encoder attacks against SD 1.5 transfer to SDXL VAE at only 20-40%,
# which is why this must be a separate ensemble target — not just a
# weight-swap of Sd15Vae.
#
# We ship the ``madebyollin/sdxl-vae-fp16-fix`` weights because that
# is what ComfyUI + Automatic1111 + every published inpainting workflow
# actually uses. The stock Stability weights suffer NaN issues in fp16
# and are avoided by nearly the whole community.
#
# See Documentation/models/vaes.md.

"""SDXL VAE encoder surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.vaes._diffusers_loader import load_autoencoder_kl

__all__ = ["SdxlVae"]

# 256 preferred at Phase R4 for Intel-Mac MPS memory headroom. The
# arch supports arbitrary multiples of 8; the CLI can override.
_PREFERRED_SIZE = 256
_ROUNDING = 8
_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
_LATENT_SCALING_FACTOR = 0.13025


class SdxlVae(nn.Module):
    """Differentiable SDXL VAE encoder.

    The forward returns the *mean* of the VAE posterior (not a sample);
    optimizing against the mean gives a well-defined deterministic
    target. The exposed ``latent`` field in :class:`TargetOutputs` is
    the raw pre-scaling latent, shape ``(N, 4, H // 8, W // 8)``.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="sdxl-vae",
        family="vaes",
        input_resolution=_PREFERRED_SIZE,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self._vae = load_autoencoder_kl(_MODEL_ID, self._device)

    @property
    def underlying_vae(self):
        """Expose the wrapped ``AutoencoderKL`` for shared use (round-trip restorer)."""
        return self._vae

    @property
    def device(self) -> torch.device:
        """Torch device this VAE lives on."""
        return self._device

    def forward(self, image: Tensor) -> TargetOutputs:
        """Encode ``image`` to the SDXL latent.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``
                RGB order. Sides are rounded to stride 8 and capped
                at :data:`_PREFERRED_SIZE`.

        Returns:
            :class:`TargetOutputs` with ``latent`` set to the posterior
            mean, shape ``(N, 4, H // 8, W // 8)``, pre-scaling.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got shape {tuple(image.shape)}."
            raise ValueError(msg)

        target_h, target_w = _round_and_cap(image.shape[-2], image.shape[-1])
        resized = (
            image
            if image.shape[-2:] == (target_h, target_w)
            else F.interpolate(
                image,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
        normalized = resized.sub(0.5).mul(2.0)
        posterior = self._vae.encode(normalized).latent_dist
        return TargetOutputs(latent=posterior.mean)

    def encode_gray_target(
        self,
        height: int = _PREFERRED_SIZE,
        width: int = _PREFERRED_SIZE,
    ) -> Tensor:
        """Compute the fixed gray latent used as the SDXL VAE attack target.

        The gray target is ``E(0.5 * ones)``. Callers cache this once
        at training start. Height and width are rounded to stride 8
        and capped at :data:`_PREFERRED_SIZE`.
        """
        h, w = _round_and_cap(height, width)
        with torch.no_grad():
            gray = torch.full((1, 3, h, w), 0.5, device=self._device)
            gray_normalized = gray.sub(0.5).mul(2.0)
            posterior = self._vae.encode(gray_normalized).latent_dist
            return posterior.mean.detach()

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]


def _round_and_cap(height: int, width: int) -> tuple[int, int]:
    h = min(_PREFERRED_SIZE, max(_ROUNDING, (height // _ROUNDING) * _ROUNDING))
    w = min(_PREFERRED_SIZE, max(_ROUNDING, (width // _ROUNDING) * _ROUNDING))
    return h, w
