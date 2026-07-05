# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""G training loop tests.

Verifies that the training loop:

  * runs without crashing over a small batch iterator,
  * converges on a synthetic composite loss (the training loss
    decreases meaningfully across ~20 steps),
  * saves and can load checkpoints when a checkpoint directory is
    provided,
  * stops cleanly when the batch iterator runs out before steps
    reach the configured maximum.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.loss import CompositeLoss, LossWeights
from voidface.core.train import TrainConfig, train_generator
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.models.base import TargetOutputs


def _batch_iterator(size: int) -> Iterator[Tensor]:
    torch.manual_seed(0)
    for _ in range(size):
        yield torch.rand(1, 3, 32, 32)


class _MeanTarget(torch.nn.Module):
    """Loss target that penalizes the mean pixel intensity."""

    def __init__(self) -> None:
        super().__init__()
        self.spec = None  # type: ignore[assignment]

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _square_mean_loss(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


def _small_composite() -> CompositeLoss:
    return CompositeLoss(
        weights=LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"mean": (_MeanTarget(), _square_mean_loss)},
    )


def test_training_loop_runs_and_reduces_loss() -> None:
    torch.manual_seed(0)
    generator = Voidface(VoidfaceConfig(epsilon=32.0 / 255.0, base_channels=8))
    result = train_generator(
        generator=generator,
        batches=_batch_iterator(30),
        composite_loss=_small_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=TrainConfig(steps=30, learning_rate=1e-3, log_every=0, device="cpu"),
    )
    initial = sum(step.total_loss for step in result.history[:5]) / 5
    final = sum(step.total_loss for step in result.history[-5:]) / 5
    assert final < initial, (initial, final)


def test_training_stops_when_batches_exhaust() -> None:
    """If batches yields fewer than config.steps, training exits cleanly."""
    generator = Voidface(VoidfaceConfig(base_channels=8))
    result = train_generator(
        generator=generator,
        batches=_batch_iterator(3),
        composite_loss=_small_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=TrainConfig(steps=100, log_every=0, device="cpu"),
    )
    assert len(result.history) == 3


def test_checkpoint_written_and_loadable(tmp_path: Path) -> None:
    generator = Voidface(VoidfaceConfig(base_channels=8))
    result = train_generator(
        generator=generator,
        batches=_batch_iterator(4),
        composite_loss=_small_composite(),
        eot=EotSampler(EotConfig(samples=1, seed=0)),
        config=TrainConfig(
            steps=4,
            log_every=0,
            checkpoint_every=2,
            checkpoint_dir=tmp_path,
            device="cpu",
        ),
    )
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()

    # A fresh generator with the same config should accept the state.
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    fresh = Voidface(VoidfaceConfig(base_channels=8))
    fresh.load_state_dict(checkpoint["state_dict"])
