# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PGD with iris-region L-infinity budget boost.

Verifies that when ``iris_mask`` is supplied, PGD's per-pixel L-inf
budget is scaled by ``iris_epsilon_ratio`` inside the mask and
stays at the standard epsilon everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.pgd import PgdConfig, run_pgd


@dataclass
class _StubBreakdown:
    """Minimal breakdown compatible with PgdStep dataclass."""

    total: float = 0.0
    per_target: dict[str, float] = None  # type: ignore[assignment]
    lpips: float = 0.0
    total_variation: float = 0.0
    restorer: str = "identity"

    def __post_init__(self) -> None:
        if self.per_target is None:
            self.per_target = {"stub": 0.0}


class _ConstantAdversarialLoss:
    """Composite-loss stub whose gradient drives delta uniformly positive.

    Loss = -mean(delta). This maximizes delta pixel-by-pixel, so
    every pixel hits its per-pixel L-inf ceiling.
    """

    def compute(
        self,
        clean_repeat: Tensor,
        eot_batch: Tensor,
        delta: Tensor,
        restorer: object | None = None,
    ) -> tuple[Tensor, _StubBreakdown]:
        loss = -delta.mean()
        return loss, _StubBreakdown()


def _iris_center_mask(height: int, width: int) -> Tensor:
    mask = torch.zeros(1, 1, height, width)
    y0, y1 = height // 4, height // 4 + height // 8
    x0, x1 = width // 4, width // 4 + width // 8
    mask[..., y0:y1, x0:x1] = 1.0
    return mask


def test_iris_mask_gives_higher_local_ceiling() -> None:
    """Inside the mask, |delta| can reach epsilon * iris_epsilon_ratio."""
    torch.manual_seed(0)
    clean = torch.full((1, 3, 32, 32), 0.5)
    eot = EotSampler(EotConfig(samples=1, seed=0))
    mask = _iris_center_mask(32, 32)

    result = run_pgd(
        clean=clean,
        composite_loss=_ConstantAdversarialLoss(),
        eot=eot,
        config=PgdConfig(epsilon=8 / 255.0, alpha=2 / 255.0, steps=20, log_every=0, seed=0),
        iris_mask=mask,
        iris_epsilon_ratio=2.0,
    )

    delta = result.delta
    inside = delta[mask.expand_as(delta) > 0.5]
    outside = delta[mask.expand_as(delta) < 0.5]

    inside_peak = inside.abs().max().item()
    outside_peak = outside.abs().max().item()

    assert inside_peak > outside_peak + 1e-6, (
        f"iris budget did not exceed outside budget: "
        f"inside={inside_peak:.5f} outside={outside_peak:.5f}"
    )
    assert inside_peak <= 2.0 * 8 / 255.0 + 1e-6
    assert outside_peak <= 8 / 255.0 + 1e-6


def test_iris_mask_ratio_1_matches_plain_pgd() -> None:
    """iris_epsilon_ratio=1.0 is identity — same as no mask at all."""
    torch.manual_seed(0)
    clean = torch.full((1, 3, 32, 32), 0.5)
    eot = EotSampler(EotConfig(samples=1, seed=0))
    mask = _iris_center_mask(32, 32)

    config = PgdConfig(epsilon=8 / 255.0, alpha=2 / 255.0, steps=15, log_every=0, seed=0)
    with_mask = run_pgd(
        clean=clean,
        composite_loss=_ConstantAdversarialLoss(),
        eot=eot,
        config=config,
        iris_mask=mask,
        iris_epsilon_ratio=1.0,
    )
    without = run_pgd(
        clean=clean,
        composite_loss=_ConstantAdversarialLoss(),
        eot=eot,
        config=config,
        iris_mask=None,
    )
    assert with_mask.delta.abs().max().item() == pytest.approx(
        without.delta.abs().max().item(), abs=1e-6
    )


def test_iris_mask_wrong_shape_raises() -> None:
    clean = torch.full((1, 3, 16, 16), 0.5)
    eot = EotSampler(EotConfig(samples=1, seed=0))
    # (H, W) instead of (N, 1, H, W)
    bad_mask = torch.zeros(16, 16)
    with pytest.raises(ValueError, match="iris_mask must be"):
        run_pgd(
            clean=clean,
            composite_loss=_ConstantAdversarialLoss(),
            eot=eot,
            config=PgdConfig(epsilon=8 / 255.0, alpha=2 / 255.0, steps=1, log_every=0, seed=0),
            iris_mask=bad_mask,
        )
