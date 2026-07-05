# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""GfpganRestorer wiring smoke test.

Uses a stub RetinaFace + stub GFPGANv1Clean so we can validate:

  * the Restorer protocol is satisfied,
  * input shape is preserved through the restorer's forward
    (paste-back correctly composites the aligned crop back to the
    input resolution),
  * gradients flow through the whole pipeline (image -> landmarks
    -> align -> gfpgan -> unalign -> loss).

The real 348 MB weight-loading + FFHQ-parity check is an on-demand
integration test tagged ``network``, not part of the default unit
suite.
"""

from __future__ import annotations

import pytest
import torch

from voidface.models.base import TargetOutputs


class _StubDetector:
    """Stub RetinaFace that returns constant landmarks + high face score."""

    def __call__(self, image: torch.Tensor) -> TargetOutputs:
        n, _, h, w = image.shape
        num_anchors = 8
        # Face-present logits high at the first anchor, low elsewhere.
        logits = torch.full((n, num_anchors, 2), -10.0)
        logits[:, 0, 1] = 10.0
        # Landmarks positioned near image center in RetinaFace's 640-space.
        landmarks = torch.zeros(n, num_anchors, 10)
        # 5 (x, y) pairs at 640-resolution reference frame.
        # left eye, right eye, nose, left mouth, right mouth
        landmarks[:, 0] = torch.tensor(
            [
                220.0, 260.0,
                420.0, 260.0,
                320.0, 340.0,
                240.0, 420.0,
                400.0, 420.0,
            ]
        )
        return TargetOutputs(
            logits=logits,
            aux={"landmarks": landmarks, "bbox_regressions": torch.zeros(n, num_anchors, 4)},
        )


class _StubGfpgan(torch.nn.Module):
    """Stub GFPGANv1Clean forward that just doubles pixel intensities.

    Enough to prove the restorer wiring: gradient flows through, shape
    is preserved, values move.
    """

    def forward(
        self, x: torch.Tensor, *, randomize_noise: bool = True
    ) -> tuple[torch.Tensor, list]:
        # Upstream returns (image, out_rgbs). Values in [-1, 1] output.
        return x * 0.5, []


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_loader(_id: str, _device: torch.device) -> _StubGfpgan:
        return _StubGfpgan().eval()

    monkeypatch.setattr(
        "voidface.models.restorers.gfpgan._load_gfpgan_v1_4_weights",
        _fake_loader,
        raising=True,
    )


def test_gfpgan_restorer_satisfies_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stubs(monkeypatch)
    from voidface.models.restorers.base import Restorer
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(detector=_StubDetector(), device=torch.device("cpu"))
    assert isinstance(restorer, Restorer)
    assert restorer.spec.name == "gfpgan-v1.4"


def test_gfpgan_restorer_preserves_input_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stubs(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(detector=_StubDetector(), device=torch.device("cpu"))
    image = torch.rand(1, 3, 320, 320)
    out = restorer(image)
    assert out.shape == image.shape


def test_gfpgan_restorer_gradient_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stubs(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    restorer = GfpganRestorer(detector=_StubDetector(), device=torch.device("cpu"))
    image = torch.rand(1, 3, 320, 320, requires_grad=True)
    out = restorer(image)
    loss = out.mean()
    loss.backward()
    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0


def test_gfpgan_restorer_falls_back_when_no_face_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the detector reports low confidence, the restorer should
    return the input unchanged."""
    _install_stubs(monkeypatch)
    from voidface.models.restorers.gfpgan import GfpganRestorer

    class _NoFaceDetector:
        def __call__(self, image: torch.Tensor) -> TargetOutputs:
            n = image.size(0)
            logits = torch.full((n, 8, 2), 0.0)  # both classes equal -> softmax = 0.5
            return TargetOutputs(
                logits=logits,
                aux={"landmarks": torch.zeros(n, 8, 10), "bbox_regressions": torch.zeros(n, 8, 4)},
            )

    restorer = GfpganRestorer(
        detector=_NoFaceDetector(),
        device=torch.device("cpu"),
        detector_score_threshold=0.9,   # softmax(0, 0)[1] = 0.5 < 0.9 -> fallback
    )
    image = torch.rand(1, 3, 128, 128)
    out = restorer(image)
    torch.testing.assert_close(out, image)
