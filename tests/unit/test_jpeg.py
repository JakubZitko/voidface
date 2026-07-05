# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Differentiable JPEG tests."""

from __future__ import annotations

import pytest
import torch

from voidface.core.jpeg import differentiable_jpeg


def test_quality_100_is_near_identity() -> None:
    """At quality 100 the round-trip should be a near-identity."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    reconstructed = differentiable_jpeg(image, quality=100)
    # Not exact due to DCT round-trip precision and YCbCr conversion,
    # but should be very close.
    diff = (image - reconstructed).abs().mean().item()
    assert diff < 0.05


def test_low_quality_degrades_meaningfully() -> None:
    """At quality 20 the round-trip should visibly degrade."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    reconstructed = differentiable_jpeg(image, quality=20)
    diff = (image - reconstructed).abs().mean().item()
    # Low quality should show meaningful degradation.
    assert diff > 0.02


def test_gradient_flows_through_jpeg() -> None:
    """Gradients must flow through the STE quantization."""
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    reconstructed = differentiable_jpeg(image, quality=75)
    reconstructed.sum().backward()
    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0


def test_output_stays_in_range() -> None:
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 16)
    reconstructed = differentiable_jpeg(image, quality=50)
    assert reconstructed.min().item() >= 0.0
    assert reconstructed.max().item() <= 1.0


def test_non_multiple_of_8_raises() -> None:
    with pytest.raises(ValueError, match="multiples of 8"):
        differentiable_jpeg(torch.rand(1, 3, 30, 30), quality=75)


def test_invalid_quality_raises() -> None:
    with pytest.raises(ValueError, match=r"\[1, 100\]"):
        differentiable_jpeg(torch.rand(1, 3, 16, 16), quality=0)


def test_eot_with_jpeg_qualities() -> None:
    """EOT sampler with jpeg_qualities set applies JPEG to samples."""
    from voidface.core.eot import EotConfig, EotSampler

    config = EotConfig(samples=1, jpeg_qualities=(50,), seed=0)
    sampler = EotSampler(config)
    image = torch.rand(1, 3, 16, 16)
    out = sampler.apply(image)
    assert out.shape == (1, 3, 16, 16)
    # JPEG at quality 50 should visibly change the output.
    assert not torch.allclose(out, image, atol=1e-3)
