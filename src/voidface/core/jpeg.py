# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Differentiable JPEG re-encoding for the EOT distribution.
#
# Instagram, TikTok, X, and WhatsApp all JPEG-re-encode uploads at a
# quality between 60 and 90. A Voidface delta trained purely against
# raw RGB does not survive that pass. The R2 EOT distribution had
# resize + Gaussian blur; this module adds the last standard channel
# degradation the trained generator needs to be robust against.
#
# The implementation follows Reich et al 2024 "Differentiable JPEG":
# forward through DCT + quantize + dequantize + iDCT, with the round
# operation replaced by a differentiable soft-round using a
# straight-through estimator so gradients flow.
#
# See Documentation/training/eot.md.

"""Differentiable JPEG re-encoding for the EOT distribution."""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["differentiable_jpeg"]


# Standard JPEG luminance quantization table (Rec. T.81 Annex K).
_LUMA_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)

# Standard JPEG chrominance quantization table.
_CHROMA_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def differentiable_jpeg(image: Tensor, quality: int = 75) -> Tensor:
    """Re-encode ``image`` through a differentiable JPEG round-trip.

    Args:
        image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]`` RGB.
            H and W must be multiples of 8 (JPEG block size). The
            caller pads if needed; the EOT sampler handles this in
            practice.
        quality: JPEG quality in ``[1, 100]``. Higher = less
            information loss.

    Returns:
        A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``, RGB, with
        gradient flow preserved via a straight-through estimator on
        the quantization step.
    """
    if image.dim() != 4 or image.size(1) != 3:
        msg = f"Expected (N, 3, H, W), got {tuple(image.shape)}."
        raise ValueError(msg)
    if image.size(-1) % 8 != 0 or image.size(-2) % 8 != 0:
        msg = (
            f"H and W must be multiples of 8 for JPEG; got "
            f"{tuple(image.shape[-2:])}."
        )
        raise ValueError(msg)
    if not 1 <= quality <= 100:
        msg = f"quality must be in [1, 100], got {quality}."
        raise ValueError(msg)

    device = image.device
    dtype = image.dtype

    ycbcr = _rgb_to_ycbcr(image)
    scale = _quality_scale(quality)
    luma_q = (_LUMA_TABLE.to(device=device, dtype=dtype) * scale).clamp(min=1.0)
    chroma_q = (_CHROMA_TABLE.to(device=device, dtype=dtype) * scale).clamp(min=1.0)

    y = _channel_roundtrip(ycbcr[:, 0:1], luma_q)
    cb = _channel_roundtrip(ycbcr[:, 1:2], chroma_q)
    cr = _channel_roundtrip(ycbcr[:, 2:3], chroma_q)

    restored = torch.cat([y, cb, cr], dim=1)
    return _ycbcr_to_rgb(restored).clamp(0.0, 1.0)


def _channel_roundtrip(channel: Tensor, quant_table: Tensor) -> Tensor:
    """DCT -> quantize (with STE) -> dequantize -> iDCT on 8x8 blocks."""
    # channel: (N, 1, H, W) in a [-128, 127] range for JPEG semantics.
    # Convert to signed 8-bit centered value expected by the DCT.
    signed = channel * 255.0 - 128.0

    blocks = _to_blocks(signed)
    coefficients = _dct_2d(blocks)
    quantized = coefficients / quant_table
    # Soft-round with straight-through estimator so gradients pass.
    rounded = quantized + (quantized.round() - quantized).detach()
    dequantized = rounded * quant_table
    reconstructed_blocks = _idct_2d(dequantized)
    reconstructed = _from_blocks(reconstructed_blocks, channel.shape)
    return ((reconstructed + 128.0) / 255.0).clamp(0.0, 1.0)


def _to_blocks(image: Tensor) -> Tensor:
    """Split ``(N, 1, H, W)`` into ``(N, num_blocks, 8, 8)``."""
    n, _, h, w = image.shape
    blocks = image.reshape(n, 1, h // 8, 8, w // 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    return blocks.reshape(n, (h // 8) * (w // 8), 8, 8)


def _from_blocks(blocks: Tensor, target_shape: tuple[int, int, int, int]) -> Tensor:
    """Reverse of :func:`_to_blocks`."""
    n, _, h, w = target_shape
    reshaped = blocks.reshape(n, 1, h // 8, w // 8, 8, 8)
    reshaped = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    return reshaped.reshape(n, 1, h, w)


def _dct_2d(blocks: Tensor) -> Tensor:
    """2-D DCT-II of each 8x8 block via 1-D DCTs along both axes."""
    return _dct_1d(_dct_1d(blocks, axis=-1), axis=-2)


def _idct_2d(blocks: Tensor) -> Tensor:
    return _idct_1d(_idct_1d(blocks, axis=-1), axis=-2)


def _dct_1d(x: Tensor, axis: int) -> Tensor:
    """1-D DCT-II along ``axis``. Length must be 8."""
    n = 8
    device = x.device
    dtype = x.dtype
    k = torch.arange(n, device=device, dtype=dtype).view(1, 1, 1, n, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, 1, 1, 1, n)
    # Basis matrix.
    basis = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    scale = torch.full((n,), math.sqrt(2 / n), device=device, dtype=dtype)
    scale[0] = math.sqrt(1 / n)
    basis = basis * scale.view(1, 1, 1, n, 1)
    if axis == -1:
        return torch.matmul(x, basis.squeeze(-1).transpose(-1, -2))
    if axis == -2:
        return torch.matmul(basis.squeeze(-1), x)
    msg = f"axis must be -1 or -2, got {axis}."
    raise ValueError(msg)


def _idct_1d(x: Tensor, axis: int) -> Tensor:
    """1-D DCT-III (inverse of DCT-II) along ``axis``. Length must be 8."""
    n = 8
    device = x.device
    dtype = x.dtype
    k = torch.arange(n, device=device, dtype=dtype).view(1, 1, 1, n, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, 1, 1, 1, n)
    basis = torch.cos(math.pi * (2 * i + 1) * k / (2 * n))
    scale = torch.full((n,), math.sqrt(2 / n), device=device, dtype=dtype)
    scale[0] = math.sqrt(1 / n)
    basis = basis * scale.view(1, 1, 1, n, 1)
    if axis == -1:
        return torch.matmul(x, basis.squeeze(-1))
    if axis == -2:
        return torch.matmul(basis.squeeze(-1).transpose(-1, -2), x)
    msg = f"axis must be -1 or -2, got {axis}."
    raise ValueError(msg)


def _quality_scale(quality: int) -> float:
    """JPEG-standard scale factor from quality parameter."""
    if quality < 50:
        return 50.0 / quality
    return 2.0 - (quality * 2.0 / 100.0)


def _rgb_to_ycbcr(rgb: Tensor) -> Tensor:
    """Rec. 601 RGB -> YCbCr in ``[0, 1]``."""
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_to_rgb(ycbcr: Tensor) -> Tensor:
    """Rec. 601 YCbCr -> RGB in ``[0, 1]``."""
    y = ycbcr[:, 0:1]
    cb = ycbcr[:, 1:2] - 0.5
    cr = ycbcr[:, 2:3] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)
