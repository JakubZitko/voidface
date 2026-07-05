# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""PGD kernel test against a synthetic differentiable target.

The real face-detector and identity-encoder tests live in
``tests/integration/``. This unit test exercises the PGD loop against a
tiny hand-crafted loss so it runs in milliseconds and requires no model
downloads.
"""

from __future__ import annotations

import torch

from voidface.core.eot import EotConfig, EotSampler
from voidface.core.loss import CompositeLoss, LossWeights
from voidface.core.pgd import PgdConfig, run_pgd
from voidface.models.base import TargetOutputs


class _MeanGap(torch.nn.Module):
    """Tiny 'ensemble target' whose logits are the sample mean.

    An adversarial loss that squares this value drives the perturbed image
    toward zero mean — an easy target to verify the PGD loop converges.
    """

    def __init__(self) -> None:
        super().__init__()
        self.spec = None  # type: ignore[assignment]

    def forward(self, image: torch.Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


def _square_logits(outputs: TargetOutputs) -> torch.Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


def test_pgd_reduces_the_target_loss() -> None:
    torch.manual_seed(0)
    target = _MeanGap()
    weights = LossWeights(targets={"mean": 1.0}, lpips=0.0, total_variation=0.0)
    composite = CompositeLoss(
        weights=weights,
        target_losses={"mean": (target, _square_logits)},
    )
    eot = EotSampler(EotConfig(samples=1, seed=0))
    clean = torch.full((1, 3, 16, 16), 0.5)
    config = PgdConfig(
        epsilon=32.0 / 255.0,
        alpha=8.0 / 255.0,
        steps=30,
        momentum=0.9,
        log_every=0,
        seed=0,
    )

    result = run_pgd(clean=clean, composite_loss=composite, eot=eot, config=config)

    initial = result.history[0].total_loss
    final = result.history[-1].total_loss
    assert final < initial, (initial, final)
    # Perturbed image should stay within [0, 1].
    assert result.adversarial.min().item() >= 0.0
    assert result.adversarial.max().item() <= 1.0
    # Perturbation stays within eps budget.
    assert result.delta.abs().max().item() <= config.epsilon + 1e-6
