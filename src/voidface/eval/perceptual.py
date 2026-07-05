# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Perceptual metric helpers.
#
# All functions accept the canonical tensor layout: (N, 3, H, W) float
# in [0, 1]. Values returned are Python floats, not tensors; the
# per-sample distinction is squashed by mean-reduction. Wrap in a
# tensor at the callsite if you need to backprop (only LPIPS supports
# that, via :func:`load_lpips`).

"""Perceptual metric helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["load_lpips", "psnr", "ssim"]


_SSIM_C1 = (0.01) ** 2
_SSIM_C2 = (0.03) ** 2


def psnr(clean: Tensor, other: Tensor) -> float:
    """Peak signal-to-noise ratio in dB.

    Args:
        clean: ``(N, 3, H, W)`` reference in ``[0, 1]``.
        other: ``(N, 3, H, W)`` comparison in ``[0, 1]``.

    Returns:
        The PSNR in decibels. ``inf`` when the two tensors are equal.
    """
    _check_shapes(clean, other)
    mse = torch.mean((clean - other) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim(clean: Tensor, other: Tensor, window_size: int = 11) -> float:
    """Structural similarity index.

    Simple depthwise implementation over ``window_size`` × ``window_size``
    Gaussian windows. Returned as a Python float in ``[-1, 1]``.

    Args:
        clean: ``(N, 3, H, W)`` reference in ``[0, 1]``.
        other: ``(N, 3, H, W)`` comparison in ``[0, 1]``.
        window_size: Gaussian window side, odd.
    """
    _check_shapes(clean, other)
    if window_size % 2 == 0:
        window_size += 1

    channels = clean.size(1)
    sigma = 1.5
    coords = torch.arange(window_size, device=clean.device, dtype=clean.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, window_size, window_size)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    padding = window_size // 2

    def _conv(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, padding=padding, groups=channels)

    mu_x = _conv(clean)
    mu_y = _conv(other)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv(clean * clean) - mu_x2
    sigma_y2 = _conv(other * other) - mu_y2
    sigma_xy = _conv(clean * other) - mu_xy

    numerator = (2 * mu_xy + _SSIM_C1) * (2 * sigma_xy + _SSIM_C2)
    denominator = (mu_x2 + mu_y2 + _SSIM_C1) * (sigma_x2 + sigma_y2 + _SSIM_C2)
    return (numerator / denominator).mean().item()


def load_lpips(net: str = "alex", device: torch.device | str = "cpu"):
    """Load the LPIPS network as a callable ``(clean, adv) -> Tensor``.

    Returns:
        A callable that takes two ``(N, 3, H, W)`` tensors in
        ``[0, 1]`` and returns a scalar-shaped tensor suitable for
        backprop. Uses ``lpips.LPIPS`` under the hood.
    """
    import lpips

    device = torch.device(device)
    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def _call(clean: Tensor, other: Tensor) -> Tensor:
        # lpips expects [-1, 1] inputs.
        return model(clean.sub(0.5).mul(2.0), other.sub(0.5).mul(2.0)).mean()

    return _call


def _check_shapes(clean: Tensor, other: Tensor) -> None:
    if clean.shape != other.shape:
        msg = f"Shape mismatch: clean={tuple(clean.shape)} other={tuple(other.shape)}"
        raise ValueError(msg)
    if clean.dim() != 4 or clean.size(1) != 3:
        msg = f"Expected (N, 3, H, W), got {tuple(clean.shape)}."
        raise ValueError(msg)
