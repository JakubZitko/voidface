# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PGD with semantic warp composition test."""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.loss import CompositeLoss, LossWeights
from voidface.core.pgd import PgdConfig, run_pgd
from voidface.models.base import TargetOutputs


class _MeanTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = None  # type: ignore[assignment]

    def forward(self, image: Tensor) -> TargetOutputs:
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _square_logits(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


def _composite() -> CompositeLoss:
    return CompositeLoss(
        weights=LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"mean": (_MeanTarget(), _square_logits)},
    )


def test_pgd_with_semantic_warp_runs_and_converges() -> None:
    torch.manual_seed(0)
    result = run_pgd(
        clean=torch.full((1, 3, 32, 32), 0.5),
        composite_loss=_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=PgdConfig(
            epsilon=32.0 / 255.0,
            alpha=8.0 / 255.0,
            steps=20,
            momentum=0.9,
            log_every=0,
            seed=0,
        ),
        semantic_warp_max_pixels=2.0,
    )
    assert len(result.history) == 20
    initial = result.history[0].total_loss
    final = result.history[-1].total_loss
    assert final < initial, (initial, final)


def test_pgd_without_semantic_warp_still_works() -> None:
    """Ensure the R1-R6 pixel-only path is unchanged when the new flag is None."""
    torch.manual_seed(0)
    result = run_pgd(
        clean=torch.full((1, 3, 16, 16), 0.5),
        composite_loss=_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=PgdConfig(
            epsilon=32.0 / 255.0,
            alpha=8.0 / 255.0,
            steps=5,
            momentum=0.9,
            log_every=0,
            seed=0,
        ),
    )
    assert len(result.history) == 5
    assert result.adversarial.shape == (1, 3, 16, 16)
