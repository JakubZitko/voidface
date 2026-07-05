# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Stable Diffusion 1.5 VAE encoder surrogate.
#
# The SD 1.5 VAE is the encoder every Automatic1111 / ComfyUI SD 1.5
# workflow passes through: img2img, inpainting (including nudify), and
# LoRA/IP-Adapter personalization. If we can drive the VAE latent for
# the perturbed image toward a fixed gray target, the downstream
# generation loses the identity information encoded in the latent.
#
# The attack objective per this target is:
#     L_vae = || E(x + delta) - z_gray ||^2 / (4 * H_lat * W_lat)
# with z_gray = E(0.5 * ones_like(x)) cached at load time.
#
# See Documentation/models/vaes.md.

"""Stable Diffusion 1.5 VAE encoder surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.vaes._diffusers_loader import load_autoencoder_kl

__all__ = ["Sd15Vae"]

# Preferred input resolution. The forward accepts any size where H, W
# are both divisible by 8; large images are downsampled here rather
# than forced up to 512 (which causes MPS OOM on low-memory devices).
_PREFERRED_SIZE = 512
_ROUNDING = 8
# The standalone Stability-published fine-tuned SD 1.5 VAE. Cleaner
# repo than the full SD 1.5 checkpoint (no gated files, no LFS
# variants), and it is the same weights every SD 1.5 workflow uses.
_MODEL_ID = "stabilityai/sd-vae-ft-mse"
_LATENT_SCALING_FACTOR = 0.18215


class Sd15Vae(nn.Module):
    """Differentiable Stable Diffusion 1.5 VAE encoder.

    The forward returns the *mean* of the VAE posterior (not a sample);
    optimizing against the mean gives a well-defined deterministic
    target. The exposed ``latent`` field in :class:`TargetOutputs` is
    the raw pre-scaling latent, shape ``(N, 4, H // 8, W // 8)``.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="sd15-vae",
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
        """Expose the wrapped ``AutoencoderKL`` for shared use.

        The restorer :class:`voidface.models.restorers.sd_vae.Sd15VaeRestorer`
        reuses this reference so encode+decode round-trip can share
        weights with the encoder-only target instead of loading a
        second copy.
        """
        return self._vae

    @property
    def device(self) -> torch.device:
        """Torch device this VAE lives on."""
        return self._device

    def forward(self, image: Tensor) -> TargetOutputs:
        """Encode ``image`` to the SD 1.5 latent.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``
                RGB order. Non-``_INPUT_SIZE`` sides are bilinearly
                resized; the caller may resize outside for efficiency.

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
            else F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
        )
        # AutoencoderKL expects [-1, 1] input, NCHW, float32.
        normalized = resized.sub(0.5).mul(2.0)
        posterior = self._vae.encode(normalized).latent_dist
        return TargetOutputs(latent=posterior.mean)

    def encode_gray_target(
        self,
        height: int = _PREFERRED_SIZE,
        width: int = _PREFERRED_SIZE,
    ) -> Tensor:
        """Compute the fixed gray latent used as the attack's target.

        The gray target is ``E(0.5 * ones)``. Callers typically cache
        this once at training start. Height and width are automatically
        rounded to the VAE's stride and capped to the preferred size.
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
    """Round to the VAE stride and cap at the preferred size."""
    h = min(_PREFERRED_SIZE, max(_ROUNDING, (height // _ROUNDING) * _ROUNDING))
    w = min(_PREFERRED_SIZE, max(_ROUNDING, (width // _ROUNDING) * _ROUNDING))
    return h, w


# NOTE: the diffusers bypass loader used to live here as
# ``_load_vae_bypassing_diffusers_loader`` and ``_normalize_vae_state_dict``.
# Both moved to ``voidface.models.vaes._diffusers_loader`` in Phase R4.1
# so they can be shared with the SDXL VAE surrogate. Import them from
# there if you need to reference them from another VAE.
