# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Semantic warp attack tests."""

from __future__ import annotations

import pytest
import torch

from voidface.attacks.semantic import SemanticWarp, apply_semantic_warp


def test_zero_field_is_near_identity() -> None:
    """A zero-displacement warp should produce a near-copy of the input."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 64, 64)
    field = torch.zeros(1, 2, 8, 8)
    warped = apply_semantic_warp(image, field, sigma_pixels=0.0)
    # grid_sample can introduce sub-pixel drift; tolerate 1e-3.
    torch.testing.assert_close(warped, image, atol=1e-3, rtol=1e-3)


def test_warp_shape_preserved() -> None:
    image = torch.rand(2, 3, 32, 48)
    field = torch.zeros(2, 2, 4, 6)
    warped = apply_semantic_warp(image, field)
    assert warped.shape == image.shape


def test_max_displacement_bound_respected() -> None:
    """After the epsilon clamp, no pixel should move by more than the bound."""
    torch.manual_seed(0)
    image = torch.zeros(1, 3, 64, 64)
    image[..., 32, 32] = 1.0
    huge_field = torch.full((1, 2, 8, 8), 100.0)
    warped = apply_semantic_warp(image, huge_field, sigma_pixels=0.0, max_displacement=3.0)
    # After clamping to max_displacement=3, the single lit pixel at
    # (32, 32) should NOT have moved to (0, 0). Check that a lit pixel
    # still exists within 6 pixels of the original position.
    lit = (warped > 0.4).nonzero(as_tuple=False)
    assert lit.numel() > 0
    for row in lit:
        _, _, y, x = row.tolist()
        assert abs(y - 32) <= 6, (y, x)
        assert abs(x - 32) <= 6, (y, x)


def test_gradient_flows_through_field() -> None:
    """Gradients must flow from a scalar downstream of the warp back into the field."""
    image = torch.rand(1, 3, 32, 32)
    field = torch.zeros(1, 2, 4, 4, requires_grad=True)
    # Introduce a non-zero starting field so the gradient signal is real.
    field.data.uniform_(-0.5, 0.5)
    warped = apply_semantic_warp(image, field)
    warped.sum().backward()
    assert field.grad is not None
    assert field.grad.abs().sum().item() > 0


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3, H, W\)"):
        apply_semantic_warp(torch.zeros(3, 8, 8), torch.zeros(1, 2, 2, 2))
    with pytest.raises(ValueError, match=r"\(N, 2, H_w, W_w\)"):
        apply_semantic_warp(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 2, 2))


def test_batch_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match image batch"):
        apply_semantic_warp(torch.zeros(2, 3, 8, 8), torch.zeros(1, 2, 2, 2))


def test_semantic_warp_class_project() -> None:
    warp = SemanticWarp(batch=1, height=32, width=32, max_displacement_pixels=1.5)
    warp.field.data.uniform_(-10.0, 10.0)
    warp.project()
    assert warp.field.abs().max().item() <= 1.5 + 1e-6


def test_semantic_warp_class_apply() -> None:
    torch.manual_seed(0)
    warp = SemanticWarp(batch=1, height=32, width=32)
    image = torch.rand(1, 3, 32, 32)
    warped = warp.apply(image)
    assert warped.shape == image.shape
