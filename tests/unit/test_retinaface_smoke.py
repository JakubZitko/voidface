# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""RetinaFace smoke tests — imports, spec, arch instantiation, wrapper
wiring, and the pre-softmax logits contract (a specific R4 correctness
critic ask). Real 109 MB weight download is separated into an on-demand
network integration test."""

from __future__ import annotations

import pytest
import torch


def test_retinaface_module_imports() -> None:
    from voidface.models.detectors.retinaface import RetinaFace

    assert RetinaFace.spec.name == "retinaface-r50"
    assert RetinaFace.spec.family == "detectors"


def test_retinaface_arch_instantiates_and_forwards() -> None:
    """The vendored architecture must construct and forward from
    random weights without touching the network."""
    from voidface.models.detectors._retinaface_arch import RetinaFaceR50Arch

    net = RetinaFaceR50Arch().eval()
    with torch.no_grad():
        bbox, cls, ldm = net(torch.randn(1, 3, 320, 320))
    # All three tensors share their anchor dimension, which depends on
    # input resolution but must be equal across the three heads.
    assert bbox.dim() == 3 and cls.dim() == 3 and ldm.dim() == 3
    assert bbox.shape[0] == cls.shape[0] == ldm.shape[0] == 1
    assert bbox.shape[1] == cls.shape[1] == ldm.shape[1]
    assert bbox.shape[2] == 4
    assert cls.shape[2] == 2
    assert ldm.shape[2] == 10


def test_retinaface_arch_returns_raw_logits_not_softmax() -> None:
    """R4 correctness critic requirement: raw pre-softmax logits.

    A softmax'd output would sum to 1 across the class dim; the raw
    logits distribution should NOT sum to 1 on average.
    """
    from voidface.models.detectors._retinaface_arch import RetinaFaceR50Arch

    torch.manual_seed(0)
    net = RetinaFaceR50Arch().eval()
    with torch.no_grad():
        _, cls, _ = net(torch.randn(1, 3, 320, 320))
    row_sums = cls.sum(dim=-1)
    # If a softmax were applied, this would be uniformly ~1. Assert
    # that it isn't.
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05)


def test_retinaface_wrapper_with_stub_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the wrapper's forward: antialias resize to 640, BGR flip,
    x255, per-channel mean subtraction, and TargetOutputs population."""

    captured: dict[str, torch.Tensor] = {}
    from voidface.models.base import TargetOutputs

    class _StubArch(torch.nn.Module):
        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            captured["input"] = x.detach().clone()
            n = x.size(0)
            return (torch.zeros(n, 100, 4), torch.zeros(n, 100, 2), torch.zeros(n, 100, 10))

        def eval(self) -> "_StubArch":  # type: ignore[override]
            return self

        def parameters(self):
            return iter([])

    def _fake_loader(_id: str, _device: torch.device) -> _StubArch:
        return _StubArch()

    monkeypatch.setattr(
        "voidface.models.detectors.retinaface._load_retinaface_r50_with_weights",
        _fake_loader,
        raising=True,
    )

    from voidface.models.detectors.retinaface import RetinaFace

    detector = RetinaFace(device=torch.device("cpu"))
    # RGB channel 0=1.0, channel 1=0.5, channel 2=0.0 constant.
    image = torch.zeros(1, 3, 256, 256)
    image[:, 0] = 1.0
    image[:, 1] = 0.5
    image[:, 2] = 0.0

    out = detector(image)
    seen = captured["input"]

    # Post-resize shape.
    assert seen.shape == (1, 3, 640, 640)
    # After BGR flip: channel 0 <- 0.0 (was channel 2), channel 2 <- 1.0.
    # After *255: 0.0, 127.5, 255.0.
    # After mean subtraction (104, 117, 123): -104.0, 10.5, 132.0.
    torch.testing.assert_close(seen[:, 0].mean(), torch.tensor(-104.0), atol=0.5, rtol=1e-2)
    torch.testing.assert_close(seen[:, 1].mean(), torch.tensor(10.5), atol=0.5, rtol=1e-2)
    torch.testing.assert_close(seen[:, 2].mean(), torch.tensor(132.0), atol=0.5, rtol=1e-2)

    # TargetOutputs contract.
    assert out.logits is not None
    assert out.logits.shape == (1, 100, 2)
    assert out.aux is not None
    assert "bbox_regressions" in out.aux
    assert "landmarks" in out.aux
