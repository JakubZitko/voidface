# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Per-target normalization in CompositeLoss.

Verifies that when weights.normalize_per_target=True, a large-
magnitude target (e.g. VAE ~50) doesn't drown out a small-magnitude
target (e.g. detector ~0.05) at equal weight.
"""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.loss import CompositeLoss, LossWeights
from voidface.models.base import TargetOutputs, TargetSpec


class _BigTarget(torch.nn.Module):
    """Emits a large-magnitude loss (~50)."""

    spec = TargetSpec(name="big", family="vaes")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


class _SmallTarget(torch.nn.Module):
    """Emits a small-magnitude loss (~0.05)."""

    spec = TargetSpec(name="small", family="detectors")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _big_loss(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean() * 50.0


def _small_loss(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean() * 0.05


def _composite(normalize: bool) -> CompositeLoss:
    return CompositeLoss(
        weights=LossWeights(
            targets={"big": 0.5, "small": 0.5},
            lpips=0.0,
            total_variation=0.0,
            normalize_per_target=normalize,
            normalization_ema_decay=0.0,  # instantaneous EMA -> current value
        ),
        target_losses={
            "big": (_BigTarget(), _big_loss),
            "small": (_SmallTarget(), _small_loss),
        },
    )


def test_without_normalization_big_dominates() -> None:
    """Without normalization the big target's contribution dominates the sum."""
    clean = torch.full((1, 3, 8, 8), 0.5)
    adversarial = torch.full((1, 3, 8, 8), 0.6)
    delta = adversarial - clean
    _, breakdown = _composite(normalize=False).compute(clean, adversarial, delta)
    # Big loss magnitude is ~1000x the small one.
    assert breakdown.per_target["big"] > 100 * breakdown.per_target["small"]


def test_with_normalization_terms_balance_out() -> None:
    """With normalization the per-target reported values are unchanged
    (they report the raw pre-normalization scalar), but the WEIGHTED
    contribution to total is balanced.

    A cheap check: the ratio (total_loss / max(per_target)) shrinks
    when normalization is enabled.
    """
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean

    _, bare = _composite(normalize=False).compute(clean, adversarial, delta)
    _, normalized = _composite(normalize=True).compute(clean, adversarial, delta)

    # The reported per-target values are the raw (pre-normalization)
    # magnitudes, so they should be similar between the two cases.
    assert abs(bare.per_target["big"] - normalized.per_target["big"]) < 1e-3
    # But the total loss should be much closer to a unit magnitude
    # under normalization.
    max_bare = max(abs(bare.per_target["big"]), abs(bare.per_target["small"]))
    max_norm = max(abs(normalized.per_target["big"]), abs(normalized.per_target["small"]))
    assert bare.total > 0.9 * max_bare * 0.5  # big term dominates ~50% weight
    assert normalized.total < 0.9 * max_norm  # normalized total < scale of raw values
