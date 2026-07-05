# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""SdxlVae smoke test — the module imports, exposes the right spec, and
its forward signature matches the ensemble contract.

The real weight download and forward-pass test is an integration test
that runs on-demand (marked ``network``), because the SDXL VAE weights
are 334 MB and the download hurts CI. The smoke test here uses monkey
patching to inject a stub AutoencoderKL so we can verify the wrapper
behaves correctly without touching the network.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch


def test_sdxl_vae_module_imports() -> None:
    from voidface.models.vaes.sdxl import SdxlVae

    assert SdxlVae is not None
    assert SdxlVae.spec.name == "sdxl-vae"
    assert SdxlVae.spec.family == "vaes"


def test_sdxl_vae_forward_shape_and_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub AutoencoderKL and verify the wrapper wiring end-to-end."""

    class _StubEncoder:
        def __init__(self, latent_channels: int = 4) -> None:
            self.latent_channels = latent_channels

        def __call__(self, x: torch.Tensor) -> Any:
            n, _, h, w = x.shape
            mean = torch.zeros(n, self.latent_channels, h // 8, w // 8)
            return type("LatentDist", (), {"mean": mean})()

    class _StubVae:
        def __init__(self) -> None:
            self._encoder = _StubEncoder()

        def encode(self, x: torch.Tensor) -> Any:
            return type("EncOut", (), {"latent_dist": self._encoder(x)})()

        def to(self, _device: torch.device) -> "_StubVae":
            return self

        def eval(self) -> "_StubVae":
            return self

        def parameters(self):
            return iter([])

    def _fake_loader(_model_id: str, _device: torch.device) -> _StubVae:
        return _StubVae()

    monkeypatch.setattr(
        "voidface.models.vaes.sdxl.load_autoencoder_kl", _fake_loader, raising=True
    )

    from voidface.models.vaes.sdxl import SdxlVae

    vae = SdxlVae(device=torch.device("cpu"))
    image = torch.rand(1, 3, 256, 256)
    out = vae(image)
    assert out.latent is not None
    assert out.latent.shape == (1, 4, 32, 32)
    # Reject wrong shapes at the boundary.
    with pytest.raises(ValueError, match=r"\(N, 3, H, W\)"):
        vae(torch.rand(3, 256, 256))
