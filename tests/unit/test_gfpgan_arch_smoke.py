# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Vendored GFPGAN arch smoke tests.

These verify that the vendored files import cleanly, the basicsr shim
covers everything they need, and the top-level GFPGANv1Clean forwards
end-to-end at a small resolution with random weights. Real weight-load
+ face-restoration parity checks land in R4.5.2 alongside the actual
GfpganRestorer wrapper.
"""

from __future__ import annotations

import pytest
import torch


def test_shims_import() -> None:
    from voidface.models.restorers._gfpgan._basicsr_shims import (
        ARCH_REGISTRY,
        default_init_weights,
    )

    # The registry decorator is a pass-through.
    assert ARCH_REGISTRY.register is not None
    assert callable(default_init_weights)

    class _Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.lin = torch.nn.Linear(4, 4)

    module = _Dummy()
    # Should not raise. Explicit call to exercise both Conv2d and Linear paths.
    default_init_weights(module, scale=0.5, bias_fill=0.0, a=0.2, mode="fan_in", nonlinearity="leaky_relu")


def test_stylegan2_clean_instantiates() -> None:
    from voidface.models.restorers._gfpgan.stylegan2_clean import StyleGAN2GeneratorClean

    # Small out_size to keep the test fast.
    gen = StyleGAN2GeneratorClean(out_size=64, num_style_feat=64, num_mlp=2)
    assert gen is not None


@pytest.mark.slow
def test_gfpgan_clean_forwards_at_low_resolution() -> None:
    """GFPGANv1Clean.forward returns (image, out_rgbs)."""
    from voidface.models.restorers._gfpgan.gfpgan_clean import GFPGANv1Clean

    # out_size must be a power of 2; 64 is the smallest that keeps the
    # arch valid without needing all the channel entries above 64.
    model = GFPGANv1Clean(
        out_size=64,
        num_style_feat=64,
        channel_multiplier=1,
        narrow=1,
    ).eval()
    with torch.no_grad():
        image, out_rgbs = model(torch.randn(1, 3, 64, 64))
    assert image.shape == (1, 3, 64, 64)
    # Upstream returns intermediate rgb outputs per SFT stage; count is
    # ``log2(out_size) - 2`` for out_size == 64 -> 4 stages.
    assert isinstance(out_rgbs, list)
    assert len(out_rgbs) == 4
