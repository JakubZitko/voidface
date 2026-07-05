# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Bilevel LPIPS term.

Verifies:

  * When bilevel_lpips weight is 0, the term contributes nothing and
    the composite loss is unchanged (backward-compat).
  * When a restorer is passed with spec.name == 'identity', the term
    is skipped (would degenerate to LPIPS(clean, adversarial), same
    as the regular LPIPS term).
  * When bilevel_lpips weight > 0 AND a non-identity restorer is
    passed, the term is subtracted from the total (we want to MAXIMIZE
    it, so it enters the loss with a negative sign).
  * LossBreakdown reports the bilevel_lpips scalar.
"""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.loss import CompositeLoss, LossWeights
from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.restorers.base import RestorerSpec
from voidface.models.restorers.identity import IdentityRestorer


class _MeanTarget(torch.nn.Module):
    spec = TargetSpec(name="mean", family="detectors")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _single_arg_loss(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


class _NonIdentityRestorer:
    """Restorer that halves pixel intensities. Not the identity."""

    spec = RestorerSpec(name="halve")

    def __call__(self, image: Tensor) -> Tensor:
        return image * 0.5


def _fake_lpips(clean: Tensor, adversarial: Tensor) -> Tensor:
    return (clean - adversarial).abs().mean()


def _base_composite(bilevel_weight: float) -> CompositeLoss:
    return CompositeLoss(
        weights=LossWeights(
            targets={"mean": 1.0},
            lpips=0.0,
            total_variation=0.0,
            bilevel_lpips=bilevel_weight,
        ),
        target_losses={"mean": (_MeanTarget(), _single_arg_loss)},
        lpips=_fake_lpips,
    )


def test_bilevel_term_off_by_default() -> None:
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean
    total, breakdown = _base_composite(bilevel_weight=0.0).compute(clean, adversarial, delta)
    assert breakdown.bilevel_lpips == 0.0


def test_bilevel_term_skipped_for_identity_restorer() -> None:
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean
    _, breakdown = _base_composite(bilevel_weight=0.5).compute(
        clean, adversarial, delta, restorer=IdentityRestorer()
    )
    assert breakdown.bilevel_lpips == 0.0


def test_bilevel_term_active_for_non_identity_restorer() -> None:
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean
    _, breakdown = _base_composite(bilevel_weight=0.5).compute(
        clean, adversarial, delta, restorer=_NonIdentityRestorer()
    )
    assert breakdown.bilevel_lpips > 0.0


def test_bilevel_term_subtracts_from_total() -> None:
    """With everything else equal, a positive bilevel LPIPS weight
    reduces the total loss vs weight=0."""
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean
    _, off = _base_composite(bilevel_weight=0.0).compute(
        clean, adversarial, delta, restorer=_NonIdentityRestorer()
    )
    _, on = _base_composite(bilevel_weight=0.5).compute(
        clean, adversarial, delta, restorer=_NonIdentityRestorer()
    )
    # bilevel term enters with a negative sign, so total should drop.
    assert on.total < off.total
