# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Expectation-over-transformation (EOT) sampler.
#
# Phase R1 provides resize + Gaussian blur only. Differentiable JPEG
# (Reich et al. 2024) and the WEBP/AVIF BPDA surrogates land in R2.
#
# See Documentation/training/eot.md for the design.

"""Expectation-over-transformation transform sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["EotConfig", "EotSampler"]


@dataclass(frozen=True)
class EotConfig:
    """Configuration for the EOT sampler.

    Attributes:
        samples: Number of transform samples per PGD step (``k``).
        resize_factors: Bilinear-resize factors to sample from.
        gaussian_sigma: Gaussian blur sigmas to sample from
            (``0`` means "skip the blur").
        seed: If not ``None``, initialize the internal RNG for
            deterministic sampling.
    """

    samples: int = 4
    resize_factors: tuple[float, ...] = (0.75, 1.0, 1.5)
    gaussian_sigma: tuple[float, ...] = (0.0, 0.5, 1.0)
    jpeg_qualities: tuple[int, ...] = ()
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        object.__setattr__(self, "_rng", rng)


class EotSampler:
    """Draw ``k`` transform pipelines and apply them to a batch.

    Usage in a PGD step::

        sampler = EotSampler(config)
        for step in range(N):
            xt = sampler.apply(x + delta)   # shape (k * N, 3, H, W)
            loss = ensemble_loss(xt)
            loss.backward()

    Every call to :meth:`apply` samples fresh transforms.
    """

    def __init__(self, config: EotConfig) -> None:
        self._config = config

    def apply(self, image: Tensor) -> Tensor:
        """Return the concatenation of ``k`` transformed copies of ``image``.

        Args:
            image: A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.

        Returns:
            A tensor of shape ``(k * N, 3, H, W)`` where consecutive
            blocks of ``N`` samples share the same transform.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W), got {tuple(image.shape)}."
            raise ValueError(msg)

        outputs: list[Tensor] = []
        for _ in range(self._config.samples):
            outputs.append(self._sample_and_apply(image))
        return torch.cat(outputs, dim=0)

    def _sample_and_apply(self, image: Tensor) -> Tensor:
        rng = self._config._rng
        scale = rng.choice(self._config.resize_factors)
        sigma = rng.choice(self._config.gaussian_sigma)

        original_hw = image.shape[-2:]
        if scale != 1.0:
            work = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
            work = F.interpolate(work, size=original_hw, mode="bilinear", align_corners=False)
        else:
            work = image

        if sigma > 0.0:
            work = _gaussian_blur(work, sigma=sigma)

        if self._config.jpeg_qualities:
            from voidface.core.jpeg import differentiable_jpeg

            quality = rng.choice(self._config.jpeg_qualities)
            # JPEG requires H, W multiples of 8 — pad if needed and crop back.
            h, w = work.shape[-2:]
            pad_h = (-h) % 8
            pad_w = (-w) % 8
            if pad_h or pad_w:
                work = F.pad(work, (0, pad_w, 0, pad_h), mode="reflect")
            work = differentiable_jpeg(work, quality=int(quality))
            if pad_h or pad_w:
                work = work[..., :h, :w]

        return work.clamp(0.0, 1.0)


def _gaussian_blur(image: Tensor, sigma: float, kernel_size: int | None = None) -> Tensor:
    """Depthwise Gaussian blur. Differentiable through the input tensor."""
    if kernel_size is None:
        kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = image.size(1)
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    weight = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()

    padding = kernel_size // 2
    return F.conv2d(image, weight, padding=padding, groups=channels)
