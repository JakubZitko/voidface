# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Gradient-checkpointing wiring for the GFPGAN restorer.

The real 348 MB GFPGAN v1.4 forward on a 512x512 crop is a memory hog
in the bilevel training loop — its StyleGAN2 decoder in particular is
what makes the outer step OOM on a 16 GB Kaggle P100. This test
exercises the checkpointing wiring on a tiny stand-in ``GFPGANv1Clean``
so the CPU test suite stays fast, and verifies:

  * :class:`GfpganRestorer` accepts a ``gradient_checkpointing`` flag
    and propagates it onto both the vendored ``GFPGANv1Clean`` and its
    inner StyleGAN2 decoder.
  * The vendored architecture produces the same forward output with
    checkpointing on vs off (up to floating-point tolerance) when
    ``randomize_noise=False`` — matching the semantics
    :class:`GfpganRestorer` uses at call time.
  * Gradients still flow to both the input tensor and to encoder
    parameters through a checkpointed forward — the bilevel LPIPS term
    (``CompositeLoss``'s negated-weight LPIPS on restorer(clean) vs
    restorer(adversarial)) depends on this being intact.
"""

from __future__ import annotations

import pytest
import torch

from voidface.models.base import TargetOutputs
from voidface.models.restorers._gfpgan.gfpgan_clean import GFPGANv1Clean


def _make_small_gfpgan(seed: int = 1234) -> GFPGANv1Clean:
    """Small deterministically-initialised GFPGANv1Clean for CPU tests.

    ``out_size=32`` gives ``log_size=5`` → 3 encoder ResBlocks + 3
    decoder ResBlocks + 3 StyleGAN2 super-blocks, enough to exercise
    both checkpoint call sites while keeping the whole network under a
    few MB of parameters.
    """
    torch.manual_seed(seed)
    net = GFPGANv1Clean(
        out_size=32,
        num_style_feat=64,
        channel_multiplier=1,
        fix_decoder=True,
        num_mlp=2,
        input_is_latent=True,
        different_w=True,
        narrow=0.5,
        sft_half=True,
    ).eval()
    return net


class _StubDetector:
    """Stub RetinaFace stand-in — the restorer's landmark path is not
    exercised in this test but ``GfpganRestorer`` requires a detector
    at construction time."""

    def __call__(self, image: torch.Tensor) -> TargetOutputs:
        n = image.size(0)
        return TargetOutputs(
            logits=torch.zeros(n, 8, 2),
            aux={
                "landmarks": torch.zeros(n, 8, 10),
                "bbox_regressions": torch.zeros(n, 8, 4),
            },
        )


def _install_gfpgan_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Divert the 348 MB weight download to a small in-memory net."""

    def _fake_loader(_id: str, _device: torch.device) -> GFPGANv1Clean:
        return _make_small_gfpgan()

    monkeypatch.setattr(
        "voidface.models.restorers.gfpgan._load_gfpgan_v1_4_weights",
        _fake_loader,
        raising=True,
    )


def test_gfpgan_restorer_defaults_to_no_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compatibility: default keeps checkpointing off."""
    _install_gfpgan_stub(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(detector=_StubDetector(), device=torch.device("cpu"))
    assert restorer._gradient_checkpointing is False
    assert restorer._net.gradient_checkpointing is False
    assert restorer._net.stylegan_decoder.gradient_checkpointing is False


def test_gfpgan_restorer_propagates_gradient_checkpointing_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag reaches the vendored architecture + inner decoder."""
    _install_gfpgan_stub(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(
        detector=_StubDetector(),
        device=torch.device("cpu"),
        gradient_checkpointing=True,
    )
    assert restorer._gradient_checkpointing is True
    assert restorer._net.gradient_checkpointing is True
    assert restorer._net.stylegan_decoder.gradient_checkpointing is True


def test_gfpgan_forward_output_matches_with_and_without_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpointing must not change the numerical output."""
    _install_gfpgan_stub(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    off = GfpganRestorer(
        detector=_StubDetector(),
        device=torch.device("cpu"),
        gradient_checkpointing=False,
    )
    on = GfpganRestorer(
        detector=_StubDetector(),
        device=torch.device("cpu"),
        gradient_checkpointing=True,
    )
    # Sanity: both restorers came from the same deterministic factory
    # so their weights match bit-for-bit — any output difference is
    # purely from the checkpoint wiring.
    for name, p in off._net.named_parameters():
        assert torch.equal(p, dict(on._net.named_parameters())[name])

    torch.manual_seed(0)
    x = (torch.rand(1, 3, 32, 32) * 2.0 - 1.0)

    with torch.no_grad():
        out_off, _ = off._net(x.clone(), randomize_noise=False)
        out_on, _ = on._net(x.clone(), randomize_noise=False)

    torch.testing.assert_close(out_off, out_on, atol=1e-5, rtol=1e-4)


def test_gfpgan_gradient_flows_through_checkpointed_restorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gradients must reach both the input tensor and encoder params.

    This is the load-bearing invariant for the bilevel LPIPS term —
    ``CompositeLoss`` calls the restorer twice (on ``clean`` and on
    ``adversarial``) and backprops through both. If checkpointing
    silently drops gradients on the adversarial branch, the negated
    LPIPS signal is gone and the delta stops learning to survive
    restoration.
    """
    _install_gfpgan_stub(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(
        detector=_StubDetector(),
        device=torch.device("cpu"),
        gradient_checkpointing=True,
    )
    # The vendored ``_load_gfpgan_v1_4_weights`` freezes every net
    # parameter — that is correct for training (GFPGAN is the frozen
    # oracle, not the thing being optimised). Un-freeze one encoder
    # conv so we can confirm gradients reach the checkpointed encoder
    # block; the input tensor's ``.grad`` is what actually matters for
    # the bilevel LPIPS backprop.
    for p in restorer._net.parameters():
        p.requires_grad_(False)
    restorer._net.conv_body_first.weight.requires_grad_(True)

    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    out, _ = restorer._net(x, randomize_noise=False)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "input gradient must flow through checkpointed restorer"
    assert x.grad.abs().sum().item() > 0, "input gradient should be nonzero"
    encoder_grad = restorer._net.conv_body_first.weight.grad
    assert encoder_grad is not None, "encoder param gradient must be populated"
    assert encoder_grad.abs().sum().item() > 0, "encoder param gradient should be nonzero"
