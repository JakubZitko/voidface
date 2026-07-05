# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Reference per-image PGD kernel.
#
# This is the algorithm every academic paper in the space calls "PGD":
# projected gradient descent with a signed-gradient step and momentum
# (MI-FGSM style). It runs the composite loss under EOT and returns
# both the perturbed image and a step-by-step trace suitable for
# logging.
#
# The shipped Voidface generator amortizes this loop into a single
# forward pass. This kernel is used at evaluation time and in the R1
# CLI, not by end users.
#
# See Documentation/attacks/pixel.md and Documentation/training/overview.md.

"""Reference per-image PGD kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from voidface.util.log import get_logger

if TYPE_CHECKING:
    from voidface.core.eot import EotSampler
    from voidface.core.loss import CompositeLoss, LossBreakdown

__all__ = ["PgdConfig", "PgdResult", "PgdStep", "run_pgd"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class PgdConfig:
    """Configuration for :func:`run_pgd`.

    Attributes:
        epsilon: L-infinity perturbation budget, expressed as a raw
            fraction of ``[0, 1]`` (i.e. ``12 / 255`` == ``0.047``).
        alpha: Per-step signed-gradient magnitude, same units as
            ``epsilon``. A common choice is ``epsilon / 6``.
        steps: Number of PGD iterations.
        momentum: Momentum factor (0 disables momentum). MI-FGSM's
            recommended value is ``0.9``.
        log_every: Number of steps between log emissions. ``0`` silences.
        seed: If not ``None``, seed the delta initialization.
    """

    epsilon: float = 12.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 200
    momentum: float = 0.9
    log_every: int = 25
    seed: int | None = None


@dataclass
class PgdStep:
    """Per-step diagnostic record."""

    step: int
    total_loss: float
    per_target: dict[str, float]
    lpips: float
    total_variation: float


@dataclass
class PgdResult:
    """Return value of :func:`run_pgd`."""

    adversarial: Tensor
    delta: Tensor
    history: list[PgdStep] = field(default_factory=list)


def run_pgd(
    clean: Tensor,
    composite_loss: CompositeLoss,
    eot: EotSampler,
    config: PgdConfig,
) -> PgdResult:
    """Run per-image PGD against the composite loss.

    Args:
        clean: A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.
        composite_loss: Configured :class:`CompositeLoss` instance.
        eot: :class:`EotSampler` used to wrap the adversarial forward.
        config: :class:`PgdConfig` with the optimizer schedule.

    Returns:
        A :class:`PgdResult` containing the final perturbed image, the
        raw delta, and the per-step history.
    """
    if clean.dim() != 4 or clean.size(1) != 3:
        msg = f"Expected (N, 3, H, W) clean input, got {tuple(clean.shape)}."
        raise ValueError(msg)
    if not 0.0 < config.alpha <= config.epsilon:
        msg = "config.alpha must satisfy 0 < alpha <= epsilon."
        raise ValueError(msg)
    if config.steps <= 0:
        msg = "config.steps must be positive."
        raise ValueError(msg)

    generator = torch.Generator(device=clean.device)
    if config.seed is not None:
        generator.manual_seed(config.seed)

    delta = _init_delta(clean, config.epsilon, generator)
    delta.requires_grad_(True)
    momentum_buffer = torch.zeros_like(delta)
    history: list[PgdStep] = []

    for step in range(config.steps):
        eot_batch = eot.apply(_clip_image(clean + delta))
        clean_repeat = clean.repeat(eot_batch.size(0) // clean.size(0), 1, 1, 1)
        loss_value, breakdown = composite_loss.compute(clean_repeat, eot_batch, delta)

        if delta.grad is not None:
            delta.grad.zero_()
        loss_value.backward()
        if delta.grad is None:
            msg = "PGD backprop produced no gradient on delta."
            raise RuntimeError(msg)

        gradient = delta.grad.detach()
        momentum_buffer = config.momentum * momentum_buffer + gradient / _l1_norm(gradient)
        step_direction = momentum_buffer.sign()

        with torch.no_grad():
            delta.sub_(config.alpha * step_direction)
            delta.clamp_(-config.epsilon, +config.epsilon)
            # Ensure the projected image stays in [0, 1].
            delta.data = (clean + delta.data).clamp(0.0, 1.0) - clean

        history.append(
            PgdStep(
                step=step,
                total_loss=breakdown.total,
                per_target=dict(breakdown.per_target),
                lpips=breakdown.lpips,
                total_variation=breakdown.total_variation,
            )
        )

        if config.log_every > 0 and (step + 1) % config.log_every == 0:
            _log.info(
                "pgd.step",
                step=step + 1,
                total=round(breakdown.total, 4),
                lpips=round(breakdown.lpips, 4),
                per_target={k: round(v, 4) for k, v in breakdown.per_target.items()},
            )

    with torch.no_grad():
        adversarial = _clip_image(clean + delta.detach())

    return PgdResult(adversarial=adversarial, delta=delta.detach(), history=history)


def _init_delta(clean: Tensor, epsilon: float, generator: torch.Generator) -> Tensor:
    """Uniform init in ``[-epsilon / 2, +epsilon / 2]``."""
    delta = torch.empty_like(clean).uniform_(
        -epsilon / 2.0, +epsilon / 2.0, generator=generator
    )
    with torch.no_grad():
        delta.data = (clean + delta.data).clamp(0.0, 1.0) - clean
    return delta


def _clip_image(x: Tensor) -> Tensor:
    return x.clamp(0.0, 1.0)


def _l1_norm(x: Tensor) -> Tensor:
    """Per-sample L1 norm used to normalize gradients before momentum."""
    dims = tuple(range(1, x.dim()))
    return x.abs().sum(dim=dims, keepdim=True).clamp(min=1e-12)
