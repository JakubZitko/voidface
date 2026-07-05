# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PGD warm-start (initial_delta) tests."""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.loss import CompositeLoss, LossWeights
from voidface.core.pgd import PgdConfig, run_pgd
from voidface.models.base import TargetOutputs


class _MeanTarget(torch.nn.Module):
    spec = None

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _square_logits(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


def _composite() -> CompositeLoss:
    return CompositeLoss(
        weights=LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"mean": (_MeanTarget(), _square_logits)},
    )


def test_warm_start_delta_gives_lower_initial_loss() -> None:
    """A warm-start delta that already reduces the target should give a
    lower step-0 loss than uniform random init."""
    clean = torch.full((1, 3, 16, 16), 0.5)
    # A warm-start delta at maximum epsilon in the DIRECTION that
    # reduces mean-squared loss: subtract 8/255 to move mean toward
    # 0.
    warm_start = torch.full_like(clean, -8.0 / 255.0)

    def _run(initial):  # noqa: ANN001,ANN202
        return run_pgd(
            clean=clean,
            composite_loss=_composite(),
            eot=EotSampler(EotConfig(samples=1, seed=0)),
            config=PgdConfig(
                epsilon=32.0 / 255.0,
                alpha=1.0 / 255.0,
                steps=1,
                momentum=0.0,
                log_every=0,
                seed=0,
                initial_delta=initial,
            ),
        )

    cold = _run(None)
    warm = _run(warm_start)
    assert warm.history[0].total_loss <= cold.history[0].total_loss


def test_warm_start_delta_projected_into_epsilon_ball() -> None:
    """A warm-start delta larger than epsilon should be clamped down."""
    clean = torch.full((1, 3, 8, 8), 0.5)
    warm_start = torch.full_like(clean, 100.0)  # way outside any epsilon
    result = run_pgd(
        clean=clean,
        composite_loss=_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=PgdConfig(
            epsilon=8.0 / 255.0,
            alpha=1.0 / 255.0,
            steps=1,
            momentum=0.0,
            log_every=0,
            seed=0,
            initial_delta=warm_start,
        ),
    )
    assert result.delta.abs().max().item() <= 8.0 / 255.0 + 1e-6
