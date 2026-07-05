# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# G training loop.
#
# The training loop is a per-batch version of the PGD kernel with two
# essential changes:
#
#   * The gradient step updates G.parameters() (the shipped model)
#     instead of a per-image delta.
#   * The per-image loss is averaged over the batch before backprop.
#
# Everything else — the CompositeLoss, EOT wrapping, RestorerSampler,
# bilevel restorer forward — is REUSED VERBATIM from the R2/R3 stack.
# The whole R1-R4 build is what makes this loop trivial.
#
# See Documentation/training/overview.md.

"""G training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from voidface.util.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from voidface.core.eot import EotSampler
    from voidface.core.loss import CompositeLoss
    from voidface.generator.architecture import Voidface
    from voidface.models.restorers.sampler import RestorerSampler

__all__ = ["TrainConfig", "TrainResult", "TrainStep", "train_generator"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Static configuration for :func:`train_generator`."""

    steps: int = 10_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    log_every: int = 100
    checkpoint_every: int = 1_000
    checkpoint_dir: Path | None = None
    device: str = "cpu"
    seed: int = 0


@dataclass
class TrainStep:
    step: int
    total_loss: float
    per_target: dict[str, float]
    lpips: float
    total_variation: float
    restorer: str


@dataclass
class TrainResult:
    generator: Voidface
    history: list[TrainStep] = field(default_factory=list)
    checkpoint_path: Path | None = None


def train_generator(
    generator: Voidface,
    batches: Iterable[Tensor],
    composite_loss: CompositeLoss,
    eot: EotSampler,
    config: TrainConfig,
    restorer_sampler: RestorerSampler | None = None,
) -> TrainResult:
    """Train ``generator`` against the composite loss.

    Args:
        generator: A :class:`Voidface` instance. Weights are updated
            in place. Do NOT pre-freeze — the whole point is to train
            these parameters.
        batches: An iterable that yields ``(N, 3, H, W)`` float tensors
            in ``[0, 1]``. Consumed lazily — a dataset with an infinite
            iterator works. Iteration ends when ``batches`` runs out or
            when ``config.steps`` is reached, whichever comes first.
        composite_loss: The same CompositeLoss the PGD kernel uses.
        eot: The same EotSampler.
        config: Training schedule and I/O.
        restorer_sampler: Optional per-step restorer sampler for the
            bilevel objective.

    Returns:
        A :class:`TrainResult` with the trained generator, step
        history, and (if enabled) the final checkpoint path.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    generator = generator.to(device)
    generator.train()

    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[TrainStep] = []
    checkpoint_path: Path | None = None

    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for clean in batches:
        if step >= config.steps:
            break
        clean = clean.to(device)
        if clean.dim() != 4 or clean.size(1) != 3:
            msg = f"Expected (N, 3, H, W) batch, got {tuple(clean.shape)}."
            raise ValueError(msg)

        adversarial = generator(clean)
        delta = adversarial - clean

        eot_batch = eot.apply(adversarial)
        clean_repeat = clean.repeat(eot_batch.size(0) // clean.size(0), 1, 1, 1)
        restorer = restorer_sampler.sample() if restorer_sampler is not None else None

        loss_value, breakdown = composite_loss.compute(
            clean_repeat, eot_batch, delta, restorer=restorer
        )

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        history.append(
            TrainStep(
                step=step,
                total_loss=breakdown.total,
                per_target=dict(breakdown.per_target),
                lpips=breakdown.lpips,
                total_variation=breakdown.total_variation,
                restorer=breakdown.restorer,
            )
        )

        if config.log_every > 0 and (step + 1) % config.log_every == 0:
            _log.info(
                "train.step",
                step=step + 1,
                total=round(breakdown.total, 4),
                lpips=round(breakdown.lpips, 4),
                restorer=breakdown.restorer,
                per_target={k: round(v, 4) for k, v in breakdown.per_target.items()},
            )

        if (
            config.checkpoint_dir is not None
            and config.checkpoint_every > 0
            and (step + 1) % config.checkpoint_every == 0
        ):
            checkpoint_path = config.checkpoint_dir / f"voidface-step-{step + 1:06d}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "state_dict": generator.state_dict(),
                    "config": generator.config,
                },
                checkpoint_path,
            )
            _log.info("train.checkpoint", path=str(checkpoint_path))

        step += 1

    generator.eval()
    return TrainResult(
        generator=generator, history=history, checkpoint_path=checkpoint_path
    )
