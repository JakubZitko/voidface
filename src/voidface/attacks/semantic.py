# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Semantic geometric warp attack.
#
# Face restorers regenerate face pixels from a StyleGAN2 face prior
# but they preserve overall face structure. If we change the
# underlying geometry — sub-millimeter shifts in jawline, cheekbone
# position, chin, and eye placement — humans do not notice at ordinary
# viewing distance, but the restorer produces a differently-restored
# face which then hashes to a different ArcFace / CLIP identity.
#
# This module gives the PGD kernel a differentiable warp field to
# optimize alongside the pixel-space delta. Total pixel displacement
# is bounded, then Gaussian-smoothed so the warp is C^1 (no sharp
# discontinuities), then applied via ``grid_sample``.
#
# See Documentation/attacks/semantic.md.

"""Semantic geometric warp attack."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SemanticWarp", "apply_semantic_warp"]


class SemanticWarp:
    """A learnable, epsilon-bounded 2-D warp field.

    The warp is parametrized by a small displacement tensor
    ``(N, 2, H_w, W_w)`` where ``H_w << H`` and ``W_w << W``. Full
    resolution is recovered by bilinear upsample + Gaussian smoothing
    so gradients still flow through the low-resolution parameter but
    the applied warp is C^1.

    Attributes:
        field: The learnable ``(N, 2, H_w, W_w)`` displacement tensor
            in pixel units. Bounded to ``[-max_displacement_pixels,
            +max_displacement_pixels]`` via :meth:`project`.
    """

    def __init__(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        field_scale: int = 8,
        max_displacement_pixels: float = 2.0,
        smoothing_sigma_pixels: float = 4.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self._h = height
        self._w = width
        self._max = float(max_displacement_pixels)
        self._sigma = float(smoothing_sigma_pixels)
        self._device = torch.device(device)

        field_h = max(1, height // field_scale)
        field_w = max(1, width // field_scale)
        self.field = torch.zeros(batch, 2, field_h, field_w, device=self._device, requires_grad=True)

    def project(self) -> None:
        """Clamp the field into the ``max_displacement_pixels`` L-inf ball."""
        with torch.no_grad():
            self.field.clamp_(-self._max, +self._max)

    def apply(self, image: Tensor) -> Tensor:
        """Warp ``image`` by the current field.

        Args:
            image: A ``(N, 3, H, W)`` tensor in ``[0, 1]``. Batch
                dimension must match the field's batch dimension.

        Returns:
            A ``(N, 3, H, W)`` warped tensor.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) image, got {tuple(image.shape)}."
            raise ValueError(msg)
        if image.shape[-2:] != (self._h, self._w):
            msg = (
                f"Image shape {tuple(image.shape[-2:])} does not match "
                f"warp field's declared shape ({self._h}, {self._w})."
            )
            raise ValueError(msg)
        return apply_semantic_warp(
            image, self.field, sigma_pixels=self._sigma, max_displacement=self._max
        )

    def parameters(self) -> list[Tensor]:
        """PyTorch parameters(...) analog for optimizer registration."""
        return [self.field]


def apply_semantic_warp(
    image: Tensor,
    field: Tensor,
    *,
    sigma_pixels: float = 4.0,
    max_displacement: float | None = 2.0,
) -> Tensor:
    """Apply a smoothed, epsilon-bounded warp to ``image``.

    Args:
        image: ``(N, 3, H, W)`` in ``[0, 1]``.
        field: ``(N, 2, H_w, W_w)`` displacement in pixels.
            Channel 0 is dx (horizontal), channel 1 is dy (vertical).
        sigma_pixels: Gaussian smoothing sigma applied to the
            full-resolution field. Higher = smoother.
        max_displacement: If not ``None``, hard-clip the final field.

    Returns:
        A ``(N, 3, H, W)`` warped tensor.
    """
    if image.dim() != 4 or image.size(1) != 3:
        msg = f"Expected (N, 3, H, W) image, got {tuple(image.shape)}."
        raise ValueError(msg)
    if field.dim() != 4 or field.size(1) != 2:
        msg = f"Expected (N, 2, H_w, W_w) field, got {tuple(field.shape)}."
        raise ValueError(msg)

    n, _, height, width = image.shape
    if field.size(0) != n:
        msg = f"Field batch {field.size(0)} does not match image batch {n}."
        raise ValueError(msg)

    # Upsample the low-res field to full res.
    full = F.interpolate(
        field, size=(height, width), mode="bilinear", align_corners=False
    )
    if sigma_pixels > 0.0:
        full = _gaussian_blur_2ch(full, sigma_pixels)
    if max_displacement is not None:
        full = full.clamp(-max_displacement, +max_displacement)

    # Convert pixel displacement to grid_sample's normalized coords.
    # We use align_corners=True everywhere so a base grid of
    # ``torch.linspace(-1, 1, N)`` produces the identity mapping.
    device = image.device
    dtype = image.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    dx_norm = full[:, 0] * (2.0 / max(1, width - 1))
    dy_norm = full[:, 1] * (2.0 / max(1, height - 1))
    offset = torch.stack([dx_norm, dy_norm], dim=-1)
    grid = base_grid + offset
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def _gaussian_blur_2ch(image: Tensor, sigma: float) -> Tensor:
    """Depthwise Gaussian blur on a 2-channel tensor."""
    if sigma <= 0:
        return image
    kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    weight = kernel_2d.expand(2, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    return F.conv2d(image, weight, padding=padding, groups=2)
