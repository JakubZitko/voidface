# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PGD accepts a restorer sampler and records the sampled restorer per step."""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.loss import CompositeLoss, LossWeights
from voidface.core.pgd import PgdConfig, run_pgd
from voidface.models.base import TargetOutputs
from voidface.models.restorers.base import RestorerSpec
from voidface.models.restorers.identity import IdentityRestorer
from voidface.models.restorers.sampler import RestorerSampler, SamplerConfig


class _MeanTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = None  # type: ignore[assignment]

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _square_logits(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


class _AddRestorer:
    """Restorer that adds 0.1 to every pixel, then clamps.

    Deliberately not the identity so we can observe its effect in the
    loss history.
    """

    spec = RestorerSpec(name="add-0.1")

    def __call__(self, image: Tensor) -> Tensor:
        return image.add(0.1).clamp(0.0, 1.0)


def test_pgd_records_sampled_restorer_per_step() -> None:
    torch.manual_seed(0)
    target = _MeanTarget()
    composite = CompositeLoss(
        weights=LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"mean": (target, _square_logits)},
    )
    eot = EotSampler(EotConfig(samples=1, seed=0))
    sampler = RestorerSampler(
        [(IdentityRestorer(), 0.5), (_AddRestorer(), 0.5)],
        SamplerConfig(seed=0),
    )
    result = run_pgd(
        clean=torch.full((1, 3, 8, 8), 0.5),
        composite_loss=composite,
        eot=eot,
        config=PgdConfig(
            epsilon=32.0 / 255.0,
            alpha=8.0 / 255.0,
            steps=20,
            momentum=0.9,
            log_every=0,
            seed=0,
        ),
        restorer_sampler=sampler,
    )
    seen = {step.restorer for step in result.history}
    # Both restorers were drawn at least once across 20 steps.
    assert seen == {"identity", "add-0.1"}, seen


def test_pgd_without_sampler_records_identity() -> None:
    torch.manual_seed(0)
    target = _MeanTarget()
    composite = CompositeLoss(
        weights=LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"mean": (target, _square_logits)},
    )
    eot = EotSampler(EotConfig(samples=1, seed=0))
    result = run_pgd(
        clean=torch.full((1, 3, 8, 8), 0.5),
        composite_loss=composite,
        eot=eot,
        config=PgdConfig(
            epsilon=32.0 / 255.0,
            alpha=8.0 / 255.0,
            steps=5,
            momentum=0.9,
            log_every=0,
            seed=0,
        ),
    )
    assert all(step.restorer == "identity" for step in result.history)
