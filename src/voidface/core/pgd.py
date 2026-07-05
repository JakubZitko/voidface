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
    from voidface.core.loss import CompositeLoss
    from voidface.models.restorers.sampler import RestorerSampler

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
        initial_delta: When provided, initialize the PGD delta from
            this tensor instead of the uniform random init. Used by
            the R7.4 --refine-steps path where PGD refines a trained
            generator's output rather than starting from noise.
    """

    epsilon: float = 12.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 200
    momentum: float = 0.9
    log_every: int = 25
    seed: int | None = None
    initial_delta: Tensor | None = None


@dataclass
class PgdStep:
    """Per-step diagnostic record."""

    step: int
    total_loss: float
    per_target: dict[str, float]
    lpips: float
    total_variation: float
    restorer: str = "identity"


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
    restorer_sampler: RestorerSampler | None = None,
    semantic_warp_max_pixels: float | None = None,
    iris_mask: Tensor | None = None,
    iris_epsilon_ratio: float = 2.0,
) -> PgdResult:
    """Run per-image PGD against the composite loss.

    Args:
        clean: A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.
        composite_loss: Configured :class:`CompositeLoss` instance.
        eot: :class:`EotSampler` used to wrap the adversarial forward.
        config: :class:`PgdConfig` with the optimizer schedule.
        restorer_sampler: Optional :class:`RestorerSampler`. When
            provided, one restorer is drawn per PGD step and every
            ensemble target sees ``restorer(adversarial)``. When
            ``None`` the identity restorer is used every step (the
            same behavior as R1 and R2).
        semantic_warp_max_pixels: When set, jointly optimize a
            geometric warp field with maximum ``max_pixels`` sub-pixel
            displacement alongside the pixel delta. Adds
            :class:`voidface.attacks.semantic.SemanticWarp` on top of
            the standard pixel attack — the R7.1 semantic warp path.
        iris_mask: Optional ``(N, 1, H, W)`` soft binary mask (see
            :func:`voidface.attacks.iris.iris_region_mask`). When
            provided, pixels inside the mask get a locally higher
            L-infinity budget scaled by ``iris_epsilon_ratio``.
            Humans do not perceive sub-millimeter iris texture
            changes at ordinary viewing distance, so this is
            budget the eye pays back to the attacker without
            visual cost.
        iris_epsilon_ratio: When ``iris_mask`` is set, the effective
            epsilon inside the mask is ``config.epsilon *
            iris_epsilon_ratio``. Default ``2.0`` matches the
            design in ``Documentation/attacks/iris.md``.

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

    epsilon_map: Tensor | float = config.epsilon
    if iris_mask is not None:
        if iris_mask.dim() != 4 or iris_mask.shape[0] != clean.shape[0]:
            msg = (
                f"iris_mask must be (N, 1, H, W) matching clean batch; "
                f"got {tuple(iris_mask.shape)} vs clean {tuple(clean.shape)}"
            )
            raise ValueError(msg)
        boost = 1.0 + (float(iris_epsilon_ratio) - 1.0) * iris_mask.to(clean.device)
        epsilon_map = config.epsilon * boost

    if getattr(config, "initial_delta", None) is not None:
        delta = config.initial_delta.detach().clone()  # type: ignore[union-attr]
        # Project the provided delta into the L-inf ball before optimizing.
        if isinstance(epsilon_map, Tensor):
            delta = torch.clamp(delta, min=-epsilon_map, max=epsilon_map)
        else:
            delta = delta.clamp(-config.epsilon, +config.epsilon)
        delta = (clean + delta).clamp(0.0, 1.0) - clean
    else:
        delta = _init_delta(clean, config.epsilon, generator)
    delta.requires_grad_(True)
    momentum_buffer = torch.zeros_like(delta)
    history: list[PgdStep] = []

    warp = None
    warp_alpha = 0.0
    if semantic_warp_max_pixels is not None and semantic_warp_max_pixels > 0.0:
        from voidface.attacks.semantic import SemanticWarp, apply_semantic_warp

        n, _, h, w = clean.shape
        warp = SemanticWarp(
            batch=n,
            height=h,
            width=w,
            max_displacement_pixels=semantic_warp_max_pixels,
            device=clean.device,
        )
        # Warp field step size is scaled to the warp budget so 1
        # PGD step covers epsilon / 6 of the L-inf ball, same
        # relationship as the pixel delta uses.
        warp_alpha = semantic_warp_max_pixels / 6.0

    for step in range(config.steps):
        perturbed = _clip_image(clean + delta)
        if warp is not None:
            perturbed = apply_semantic_warp(
                perturbed,
                warp.field,
                sigma_pixels=4.0,
                max_displacement=semantic_warp_max_pixels,
            )
        eot_batch = eot.apply(perturbed)
        clean_repeat = clean.repeat(eot_batch.size(0) // clean.size(0), 1, 1, 1)
        restorer = restorer_sampler.sample() if restorer_sampler is not None else None
        loss_value, breakdown = composite_loss.compute(
            clean_repeat, eot_batch, delta, restorer=restorer
        )

        if delta.grad is not None:
            delta.grad.zero_()
        if warp is not None and warp.field.grad is not None:
            warp.field.grad.zero_()
        loss_value.backward()
        if delta.grad is None:
            msg = "PGD backprop produced no gradient on delta."
            raise RuntimeError(msg)

        gradient = delta.grad.detach()
        momentum_buffer = config.momentum * momentum_buffer + gradient / _l1_norm(gradient)
        step_direction = momentum_buffer.sign()

        with torch.no_grad():
            delta.sub_(config.alpha * step_direction)
            if isinstance(epsilon_map, Tensor):
                delta.data = torch.clamp(delta.data, min=-epsilon_map, max=epsilon_map)
            else:
                delta.clamp_(-config.epsilon, +config.epsilon)
            # Ensure the projected image stays in [0, 1].
            delta.data = (clean + delta.data).clamp(0.0, 1.0) - clean
            if warp is not None and warp.field.grad is not None:
                warp.field.data.sub_(warp_alpha * warp.field.grad.sign())
                warp.project()

        history.append(
            PgdStep(
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
                "pgd.step",
                step=step + 1,
                total=round(breakdown.total, 4),
                lpips=round(breakdown.lpips, 4),
                restorer=breakdown.restorer,
                per_target={k: round(v, 4) for k, v in breakdown.per_target.items()},
            )

    with torch.no_grad():
        adversarial = _clip_image(clean + delta.detach())
        if warp is not None:
            from voidface.attacks.semantic import apply_semantic_warp

            adversarial = apply_semantic_warp(
                adversarial,
                warp.field.detach(),
                sigma_pixels=4.0,
                max_displacement=semantic_warp_max_pixels,
            )

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
