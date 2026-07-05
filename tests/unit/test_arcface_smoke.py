# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Arcface smoke tests — imports, spec, wrapper wiring against a
monkey-patched IResNet so CI doesn't pull the 249 MB checkpoint per run.

The real weight-download + forward-parity integration test is separate
and marked with ``@pytest.mark.network``.
"""

from __future__ import annotations

import pytest
import torch


def test_arcface_module_imports() -> None:
    from voidface.models.recognizers.arcface import Arcface

    assert Arcface.spec.name == "arcface"
    assert Arcface.spec.family == "recognizers"
    assert Arcface.spec.input_resolution == 112


def test_iresnet100_instantiates_without_weights() -> None:
    """The vendored arch must at least construct on CPU without the checkpoint."""
    from voidface.models.recognizers._iresnet import iresnet100

    net = iresnet100().eval()
    # Total param count should be roughly 65 million (upstream R100).
    total = sum(p.numel() for p in net.parameters())
    assert 60_000_000 < total < 70_000_000, total


def test_iresnet100_forward_shape() -> None:
    """Forward through untrained weights must at least give (N, 512)."""
    from voidface.models.recognizers._iresnet import iresnet100

    net = iresnet100().eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 112, 112))
    assert out.shape == (2, 512)


def test_arcface_wrapper_with_stub_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the wrapper's forward: BGR flip + [-1, 1] normalize + L2 norm."""

    captured: dict[str, torch.Tensor] = {}

    class _StubNet(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            captured["input"] = x.detach().clone()
            # Fake 512-D output with a fixed direction so we can check
            # F.normalize turns it into a unit vector.
            return torch.ones(x.size(0), 512) * 2.0

        def eval(self) -> "_StubNet":  # type: ignore[override]
            return self

        def parameters(self):
            return iter([])

    def _fake_loader(_model_id: str, _device: torch.device) -> _StubNet:
        return _StubNet()

    monkeypatch.setattr(
        "voidface.models.recognizers.arcface._load_iresnet100_with_weights",
        _fake_loader,
        raising=True,
    )

    from voidface.models.recognizers.arcface import Arcface

    encoder = Arcface(device=torch.device("cpu"))
    # Craft an input where the channel order is easy to inspect:
    # channel 0 = 1.0, channel 1 = 0.5, channel 2 = 0.0.
    image = torch.zeros(1, 3, 128, 128)
    image[:, 0] = 1.0
    image[:, 1] = 0.5
    image[:, 2] = 0.0

    out = encoder(image)

    # After resize to 112 the values are preserved (constant per channel),
    # then BGR flip swaps channel 0 <-> channel 2, then [-1, 1] normalize.
    #   Original RGB channel means: 1.0, 0.5, 0.0
    #   After BGR flip:             0.0, 0.5, 1.0
    #   After (x-0.5)*2:            -1.0, 0.0, 1.0
    seen = captured["input"]
    assert seen.shape == (1, 3, 112, 112)
    torch.testing.assert_close(seen[:, 0].mean(), torch.tensor(-1.0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(seen[:, 1].mean(), torch.tensor(0.0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(seen[:, 2].mean(), torch.tensor(1.0), atol=1e-4, rtol=1e-4)

    # Output should be L2-normalized (unit norm) 512-D.
    assert out.embedding is not None
    assert out.embedding.shape == (1, 512)
    norms = out.embedding.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(1), atol=1e-5, rtol=1e-5)
