# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Composite adversarial loss.
#
# The composite loss is a weighted sum of per-target adversarial terms
# and a perceptual budget on the perturbation. Each target contributes
# an atomic loss (see the target's docstring for the exact formula);
# this module composes them into the scalar that gets backpropagated.
#
# See Documentation/training/overview.md and Documentation/attacks/pixel.md.

"""Composite adversarial loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from voidface.models.base import EnsembleTarget, TargetOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "CompositeLoss",
    "LossBreakdown",
    "LossWeights",
    "arcface_identity_loss",
    "retinaface_suppression_loss",
]

_TAU_FACE_SUPPRESSION = 0.05


@dataclass(frozen=True)
class LossWeights:
    """Static weights for the composite loss."""

    targets: Mapping[str, float] = field(default_factory=dict)
    lpips: float = 0.10
    total_variation: float = 0.01


@dataclass
class LossBreakdown:
    """Per-term breakdown of the composite loss for a single step."""

    per_target: dict[str, float]
    lpips: float
    total_variation: float
    total: float


def retinaface_suppression_loss(outputs: TargetOutputs) -> Tensor:
    """Drive per-anchor face-present probability below ``tau``.

    The RetinaFace surrogate returns per-anchor softmax scores of shape
    ``(N, num_anchors, 2)``. The face-present channel is ``[..., 1]``.
    We penalize the squared amount by which each face-anchor exceeds the
    suppression threshold, averaged across anchors.

    Args:
        outputs: The output of the RetinaFace surrogate.

    Returns:
        A scalar tensor. Zero when every anchor is below threshold.
    """
    if outputs.logits is None:
        msg = "RetinaFace loss requires TargetOutputs.logits to be populated."
        raise ValueError(msg)
    face_score = outputs.logits[..., 1]
    excess = (face_score - _TAU_FACE_SUPPRESSION).clamp(min=0.0)
    return excess.pow(2).mean()


def arcface_identity_loss(perturbed: TargetOutputs, clean: TargetOutputs) -> Tensor:
    """Push the perturbed identity embedding away from the clean one.

    Both inputs must have ``embedding`` populated with L2-normalized
    tensors of shape ``(N, D)``. The loss is
    ``1 + cos_similarity`` — maximized when the cosine is +1 (embeddings
    agree), minimized when the cosine is -1 (opposite identity).

    Args:
        perturbed: Ensemble output for the perturbed image.
        clean: Ensemble output for the unperturbed reference image.

    Returns:
        A scalar tensor.
    """
    if perturbed.embedding is None or clean.embedding is None:
        msg = "ArcFace loss requires embedding on both inputs."
        raise ValueError(msg)
    cos = F.cosine_similarity(perturbed.embedding, clean.embedding, dim=-1)
    return cos.add(1.0).mean()


def total_variation(delta: Tensor) -> Tensor:
    """Anisotropic total variation of ``delta``.

    Encourages smooth perturbations that survive JPEG re-encoding.
    Argument shape: ``(N, C, H, W)``.
    """
    dh = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
    dw = (delta[..., :, 1:] - delta[..., :, :-1]).abs().mean()
    return dh + dw


class CompositeLoss:
    """Compose per-target adversarial losses with a perceptual budget.

    Instances are stateless with respect to model weights; they hold
    the weight configuration and a callable per target. This class does
    not own the surrogate models — pass them to :meth:`compute`.

    Example::

        loss = CompositeLoss(
            weights=LossWeights(targets={"retinaface": 0.5, "arcface": 0.5}),
            target_losses={
                "retinaface": (retinaface, retinaface_suppression_loss),
                "arcface": (arcface, arcface_identity_loss_pair),
            },
        )
        value, breakdown = loss.compute(clean, adversarial, delta)
    """

    def __init__(
        self,
        weights: LossWeights,
        target_losses: Mapping[str, tuple[EnsembleTarget, TargetLossFn]],
        lpips: LpipsFn | None = None,
    ) -> None:
        self._weights = weights
        self._targets = dict(target_losses)
        self._lpips = lpips

    def compute(
        self,
        clean: Tensor,
        adversarial: Tensor,
        delta: Tensor,
    ) -> tuple[Tensor, LossBreakdown]:
        """Compute the composite loss and a breakdown of its terms.

        Args:
            clean: The original image, ``(N, 3, H, W)`` in ``[0, 1]``.
            adversarial: The perturbed image, same shape.
            delta: The perturbation itself, same shape. Passed
                separately from ``adversarial`` so that TV and LPIPS
                regularizers can use it without a subtraction.

        Returns:
            A pair ``(total_loss, breakdown)`` where ``breakdown`` is
            a :class:`LossBreakdown` with float values suitable for
            logging.
        """
        per_target_scalars: dict[str, float] = {}
        weighted_target_loss = adversarial.new_zeros(())

        for name, (target, loss_fn) in self._targets.items():
            weight = self._weights.targets.get(name, 0.0)
            if weight == 0.0:
                continue
            adv_out = target(adversarial)
            clean_out = target(clean) if _requires_clean(loss_fn) else None
            value = loss_fn(adv_out, clean_out) if clean_out is not None else loss_fn(adv_out)
            weighted_target_loss = weighted_target_loss + weight * value
            per_target_scalars[name] = float(value.detach())

        lpips_value = adversarial.new_zeros(())
        if self._lpips is not None and self._weights.lpips > 0.0:
            lpips_value = self._lpips(clean, adversarial)

        tv_value = total_variation(delta)

        total = (
            weighted_target_loss
            + self._weights.lpips * lpips_value
            + self._weights.total_variation * tv_value
        )
        breakdown = LossBreakdown(
            per_target=per_target_scalars,
            lpips=float(lpips_value.detach()) if isinstance(lpips_value, Tensor) else 0.0,
            total_variation=float(tv_value.detach()),
            total=float(total.detach()),
        )
        return total, breakdown


# --- Type aliases ------------------------------------------------------------

TargetLossFn = object  # too polymorphic to hint precisely; see docstrings
LpipsFn = object


def _requires_clean(loss_fn: object) -> bool:
    """Heuristic: pair-style losses take two arguments.

    RetinaFace suppression is single-argument. ArcFace identity is a
    pair (perturbed, clean). This is checked by argument-count on the
    callable; it lets the composite loop pick the right call shape
    without an explicit tag.
    """
    import inspect

    signature = inspect.signature(loss_fn)  # type: ignore[arg-type]
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2
