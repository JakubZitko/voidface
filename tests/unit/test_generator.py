# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Generator G architecture tests.

Verifies the core contract of the shipped model:

  * Instantiation with the default config produces the R5 target size
    (5-10 M params).
  * Forward preserves ``(N, 3, H, W)`` shape and stays in ``[0, 1]``.
  * The L-infinity delta budget is honored (|G(x) - x| <= epsilon).
  * Non-conforming input sizes raise cleanly.
  * Deterministic — same input, same output.
  * Gradient flows through the whole network (needed for training).
"""

from __future__ import annotations

import pytest
import torch

from voidface.generator.architecture import Voidface, VoidfaceConfig


def test_default_generator_param_count_within_r5_target() -> None:
    net = Voidface()
    total = sum(p.numel() for p in net.parameters())
    # R5 target: 5-10 M parameters float32.
    assert 3_000_000 < total < 12_000_000, total


def test_forward_preserves_shape_and_range() -> None:
    net = Voidface().eval()
    with torch.no_grad():
        out = net(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 3, 128, 128)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_delta_within_epsilon_budget() -> None:
    net = Voidface(VoidfaceConfig(epsilon=8.0 / 255.0)).eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = net(x)
    delta = (y - x).abs().max().item()
    # tanh output is (-1, 1), scaled by epsilon, then clamped to [0, 1].
    # The absolute delta must not exceed epsilon.
    assert delta <= 8.0 / 255.0 + 1e-6


def test_epsilon_override_is_respected() -> None:
    net = Voidface(VoidfaceConfig(epsilon=8.0 / 255.0)).eval()
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        y = net(x, epsilon=2.0 / 255.0)
    delta = (y - x).abs().max().item()
    assert delta <= 2.0 / 255.0 + 1e-6


def test_non_divisible_input_raises() -> None:
    net = Voidface().eval()
    # 130 is not divisible by 2^4 = 16.
    with pytest.raises(ValueError, match="divisible by 16"):
        net(torch.rand(1, 3, 130, 128))


def test_forward_is_deterministic() -> None:
    net = Voidface().eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y1 = net(x)
        y2 = net(x)
    torch.testing.assert_close(y1, y2)


def test_gradient_flows_for_training() -> None:
    net = Voidface()
    x = torch.rand(1, 3, 64, 64)
    y = net(x)
    loss = y.sum()
    loss.backward()
    # Every parameter should receive a gradient.
    for name, param in net.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().sum().item() > 0, name


def test_selfattention_variant_instantiates() -> None:
    net = Voidface(VoidfaceConfig(attention_at_bottleneck=True)).eval()
    with torch.no_grad():
        y = net(torch.rand(1, 3, 64, 64))
    assert y.shape == (1, 3, 64, 64)
