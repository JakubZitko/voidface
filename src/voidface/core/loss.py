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
# Phase R2 adds the restorer-in-the-loop shape. When a Restorer is
# passed to :meth:`CompositeLoss.compute`, every target sees
# ``restorer(adversarial)`` instead of ``adversarial`` itself. Clean
# reference passes are always through the identity — the clean image
# is what the attacker downloads.
#
# See Documentation/training/overview.md and
# Documentation/training/bilevel-adversarial.md.

"""Composite adversarial loss."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from voidface.models.base import EnsembleTarget, TargetOutputs

if TYPE_CHECKING:
    from voidface.models.restorers.base import Restorer

__all__ = [
    "CompositeLoss",
    "LossBreakdown",
    "LossWeights",
    "arcface_identity_loss",
    "retinaface_suppression_loss",
    "total_variation",
    "vae_gray_latent_loss",
]

_TAU_FACE_SUPPRESSION = 0.05


@dataclass(frozen=True)
class LossWeights:
    """Static weights for the composite loss.

    Attributes:
        targets: Per-target weights for the ensemble adversarial
            losses.
        lpips: Weight on ``LPIPS(clean, adversarial)`` — the
            perceptual invisibility constraint. Minimized (the
            generator wants small perceptual distance to the clean
            reference so the perturbation is invisible).
        total_variation: Weight on TV(delta) — a smoothness prior
            that helps the perturbation survive JPEG.
        bilevel_lpips: Weight on ``LPIPS(restorer(clean),
            restorer(adversarial))`` — the CEO-critic-mandated
            "does the attack survive restoration" perceptual signal.
            Subtracted from the loss (we want restored versions to
            look different — MAXIMIZE this LPIPS). Only active when
            a restorer is passed to :meth:`CompositeLoss.compute`
            AND its spec.name is not ``"identity"``. Recommended
            small (e.g. 0.05) so it complements rather than dominates
            the ensemble adversarial signal.
    """

    targets: Mapping[str, float] = field(default_factory=dict)
    lpips: float = 0.10
    total_variation: float = 0.01
    bilevel_lpips: float = 0.0


@dataclass
class LossBreakdown:
    """Per-term breakdown of the composite loss for a single step."""

    per_target: dict[str, float]
    lpips: float
    total_variation: float
    total: float
    restorer: str = "identity"
    bilevel_lpips: float = 0.0


def retinaface_suppression_loss(outputs: TargetOutputs) -> Tensor:
    """Drive per-anchor face-present probability below ``tau``.

    Applicable to any detector surrogate whose ``TargetOutputs.logits``
    field encodes a per-anchor softmax score with the face-present
    channel at ``[..., 1]``. The name is historical (Phase R1's stand-in
    is P-Net; the R2/R3 replacement is real RetinaFace).
    """
    if outputs.logits is None:
        msg = "Detector loss requires TargetOutputs.logits to be populated."
        raise ValueError(msg)
    face_score = outputs.logits[..., 1]
    excess = (face_score - _TAU_FACE_SUPPRESSION).clamp(min=0.0)
    return excess.pow(2).mean()


def arcface_identity_loss(perturbed: TargetOutputs, clean: TargetOutputs) -> Tensor:
    """Push the perturbed identity embedding away from the clean one.

    Applicable to any identity encoder whose ``TargetOutputs.embedding``
    is an L2-normalized ``(N, D)`` tensor. Loss is ``1 + cos_similarity``:
    maximized when the two embeddings agree, minimized when opposite.
    """
    if perturbed.embedding is None or clean.embedding is None:
        msg = "Identity loss requires embedding on both inputs."
        raise ValueError(msg)
    cos = F.cosine_similarity(perturbed.embedding, clean.embedding, dim=-1)
    return cos.add(1.0).mean()


def vae_gray_latent_loss(
    perturbed: TargetOutputs,
    *,
    target_data: Tensor,
) -> Tensor:
    """Drive the VAE latent for the perturbed image toward a gray target.

    Applicable to any VAE whose ``TargetOutputs.latent`` is a
    ``(N, C, H, W)`` posterior mean. The loss is mean-squared distance
    to the fixed gray latent, normalized by the number of latent
    elements per sample so that different VAE families (SD 1.5 with 4
    channels, Flux with 16 channels) yield comparably-scaled loss
    values.

    Args:
        perturbed: Ensemble output for the perturbed image.
        target_data: The fixed gray latent, cached at training start
            via :meth:`Sd15Vae.encode_gray_target` (or its equivalent
            on other VAE families).

    Returns:
        A scalar tensor.
    """
    if perturbed.latent is None:
        msg = "VAE loss requires TargetOutputs.latent to be populated."
        raise ValueError(msg)
    latent = perturbed.latent
    diff = latent - target_data.expand_as(latent)
    return diff.pow(2).mean()


def total_variation(delta: Tensor) -> Tensor:
    """Anisotropic total variation of ``delta``.

    Encourages smooth perturbations that survive JPEG re-encoding.
    Argument shape: ``(N, C, H, W)``.
    """
    dh = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
    dw = (delta[..., :, 1:] - delta[..., :, :-1]).abs().mean()
    return dh + dw


TargetLossFn = Callable[..., Tensor]
LpipsFn = Callable[[Tensor, Tensor], Tensor]


class CompositeLoss:
    """Compose per-target adversarial losses with a perceptual budget.

    Each entry in ``target_losses`` maps a target name to a
    ``(target, loss_fn)`` pair. The signature of ``loss_fn`` determines
    how it is invoked:

    * one positional arg  -> ``loss_fn(target(x))``. Used for
      single-side losses like detector suppression.
    * two positional args -> ``loss_fn(target(x), target(clean))``.
      Used for identity losses that compare against the reference.
    * ``target_data`` kw   -> passed the value in
      ``target_static_data[name]``. Used for VAE gray-target losses
      that carry a precomputed constant tensor.
    """

    def __init__(
        self,
        weights: LossWeights,
        target_losses: Mapping[str, tuple[EnsembleTarget, TargetLossFn]],
        target_static_data: Mapping[str, Tensor] | None = None,
        lpips: LpipsFn | None = None,
    ) -> None:
        self._weights = weights
        self._targets = dict(target_losses)
        self._static = dict(target_static_data or {})
        self._lpips = lpips

    def compute(
        self,
        clean: Tensor,
        adversarial: Tensor,
        delta: Tensor,
        restorer: Restorer | None = None,
    ) -> tuple[Tensor, LossBreakdown]:
        """Compute the composite loss and a breakdown of its terms.

        Args:
            clean: The original image, ``(N, 3, H, W)`` in ``[0, 1]``.
            adversarial: The perturbed image, same shape.
            delta: The perturbation, same shape. Passed separately so
                TV and LPIPS regularizers can use it without a
                subtraction.
            restorer: Optional restorer applied to ``adversarial``
                before every target forward. Clean pass is never
                restored — the clean image is what the attacker starts
                from.

        Returns:
            A pair ``(total_loss, breakdown)`` where ``breakdown`` is
            a :class:`LossBreakdown` with float values suitable for
            logging.
        """
        adversarial_input = adversarial if restorer is None else restorer(adversarial)

        per_target_scalars: dict[str, float] = {}
        weighted_target_loss = adversarial.new_zeros(())

        for name, (target, loss_fn) in self._targets.items():
            weight = self._weights.targets.get(name, 0.0)
            if weight == 0.0:
                continue
            adv_out = target(adversarial_input)
            value = self._invoke(name, loss_fn, adv_out, target, clean)
            weighted_target_loss = weighted_target_loss + weight * value
            per_target_scalars[name] = float(value.detach())

        lpips_value = adversarial.new_zeros(())
        if self._lpips is not None and self._weights.lpips > 0.0:
            lpips_value = self._lpips(clean, adversarial)

        bilevel_lpips_value = adversarial.new_zeros(())
        active_restorer = restorer is not None and restorer.spec.name != "identity"
        if (
            self._lpips is not None
            and self._weights.bilevel_lpips > 0.0
            and active_restorer
        ):
            # Route the clean side through the same restorer so we
            # measure whether the delta survives restoration.
            assert restorer is not None  # for type checkers
            clean_restored = restorer(clean)
            bilevel_lpips_value = self._lpips(clean_restored, adversarial_input)

        tv_value = total_variation(delta)

        total = (
            weighted_target_loss
            + self._weights.lpips * lpips_value
            - self._weights.bilevel_lpips * bilevel_lpips_value
            + self._weights.total_variation * tv_value
        )
        breakdown = LossBreakdown(
            per_target=per_target_scalars,
            lpips=float(lpips_value.detach()) if isinstance(lpips_value, Tensor) else 0.0,
            total_variation=float(tv_value.detach()),
            total=float(total.detach()),
            restorer=restorer.spec.name if restorer is not None else "identity",
            bilevel_lpips=(
                float(bilevel_lpips_value.detach())
                if isinstance(bilevel_lpips_value, Tensor)
                else 0.0
            ),
        )
        return total, breakdown

    def _invoke(
        self,
        name: str,
        loss_fn: TargetLossFn,
        adv_out: TargetOutputs,
        target: EnsembleTarget,
        clean: Tensor,
    ) -> Tensor:
        signature = inspect.signature(loss_fn)
        kinds = {p.name for p in signature.parameters.values()}
        if "target_data" in kinds:
            static = self._static.get(name)
            if static is None:
                msg = (
                    f"target_losses[{name!r}] declares a target_data kwarg but no "
                    f"entry was provided in target_static_data."
                )
                raise ValueError(msg)
            return loss_fn(adv_out, target_data=static)

        positional = [
            p
            for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            clean_out = target(clean)
            return loss_fn(adv_out, clean_out)
        return loss_fn(adv_out)
