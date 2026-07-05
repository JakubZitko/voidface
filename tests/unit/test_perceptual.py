# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PSNR / SSIM basic properties."""

from __future__ import annotations

import pytest
import torch

from voidface.eval.perceptual import psnr, ssim


def test_psnr_of_equal_images_is_infinite() -> None:
    x = torch.rand(1, 3, 16, 16)
    assert psnr(x, x) == float("inf")


def test_psnr_decreases_with_noise() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, 3, 16, 16)
    noise_small = x + 0.001 * torch.randn_like(x)
    noise_large = x + 0.05 * torch.randn_like(x)
    p_small = psnr(x, noise_small.clamp(0, 1))
    p_large = psnr(x, noise_large.clamp(0, 1))
    assert p_small > p_large


def test_ssim_of_equal_images_is_one() -> None:
    x = torch.rand(1, 3, 32, 32)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-4)


def test_ssim_shape_mismatch_raises() -> None:
    a = torch.rand(1, 3, 16, 16)
    b = torch.rand(1, 3, 24, 24)
    with pytest.raises(ValueError, match="Shape mismatch"):
        ssim(a, b)
