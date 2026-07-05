# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""CompositeLoss dispatches to single-arg, pair-arg, and target_data-kwarg
losses correctly, and applies a restorer wrapping when provided."""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.core.loss import CompositeLoss, LossWeights, total_variation, vae_gray_latent_loss
from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.restorers.identity import IdentityRestorer


class _EchoLogits(torch.nn.Module):
    """Echo the mean value of the image in TargetOutputs.logits."""

    spec = TargetSpec(name="echo-logits", family="detectors")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=image.mean(dim=(1, 2, 3), keepdim=True))


class _EchoEmbedding(torch.nn.Module):
    """Embed the image by mean-pool into a 4-D vector."""

    spec = TargetSpec(name="echo-embed", family="recognizers")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        vec = image.mean(dim=(2, 3))  # (N, 3)
        vec = torch.cat([vec, image.std(dim=(2, 3))], dim=-1)  # (N, 6)
        return TargetOutputs(embedding=vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8))


class _EchoLatent(torch.nn.Module):
    """Emit a fixed-shape latent proportional to the input."""

    spec = TargetSpec(name="echo-latent", family="vaes")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        # Down-sample by 4 for a "latent"-shaped output.
        latent = torch.nn.functional.avg_pool2d(image, 4).sum(dim=1, keepdim=True)
        return TargetOutputs(latent=latent)


def _single_arg_loss(outputs: TargetOutputs) -> Tensor:
    assert outputs.logits is not None
    return outputs.logits.pow(2).mean()


def _pair_arg_loss(perturbed: TargetOutputs, clean: TargetOutputs) -> Tensor:
    assert perturbed.embedding is not None and clean.embedding is not None
    return (perturbed.embedding - clean.embedding).pow(2).mean()


def test_dispatches_by_signature() -> None:
    clean = torch.full((1, 3, 8, 8), 0.5)
    adversarial = torch.full((1, 3, 8, 8), 0.6)
    delta = adversarial - clean
    gray_latent = torch.zeros(1, 1, 2, 2)

    composite = CompositeLoss(
        weights=LossWeights(
            targets={"single": 0.4, "pair": 0.4, "vae": 0.2},
            lpips=0.0,
            total_variation=0.0,
        ),
        target_losses={
            "single": (_EchoLogits(), _single_arg_loss),
            "pair": (_EchoEmbedding(), _pair_arg_loss),
            "vae": (_EchoLatent(), vae_gray_latent_loss),
        },
        target_static_data={"vae": gray_latent},
    )
    total, breakdown = composite.compute(clean, adversarial, delta)
    assert set(breakdown.per_target) == {"single", "pair", "vae"}
    assert total.item() > 0
    assert breakdown.restorer == "identity"


def test_identity_restorer_no_op_matches_no_restorer() -> None:
    torch.manual_seed(0)
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean

    composite = CompositeLoss(
        weights=LossWeights(targets={"single": 1.0}, lpips=0.0, total_variation=0.0),
        target_losses={"single": (_EchoLogits(), _single_arg_loss)},
    )
    total_bare, _ = composite.compute(clean, adversarial, delta)
    total_with_id, breakdown = composite.compute(
        clean, adversarial, delta, restorer=IdentityRestorer()
    )
    assert torch.allclose(total_bare, total_with_id)
    assert breakdown.restorer == "identity"


def test_zero_weight_target_is_skipped() -> None:
    clean = torch.rand(1, 3, 8, 8)
    adversarial = torch.rand(1, 3, 8, 8)
    delta = adversarial - clean
    composite = CompositeLoss(
        weights=LossWeights(
            targets={"used": 1.0, "skipped": 0.0}, lpips=0.0, total_variation=0.0
        ),
        target_losses={
            "used": (_EchoLogits(), _single_arg_loss),
            "skipped": (_EchoLogits(), _single_arg_loss),
        },
    )
    _, breakdown = composite.compute(clean, adversarial, delta)
    assert "used" in breakdown.per_target
    assert "skipped" not in breakdown.per_target
