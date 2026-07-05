# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""OpenClip smoke tests — import, spec, wrapper wiring via a
monkey-patched CLIPVisionModel so CI doesn't pull 150 MB per run."""

from __future__ import annotations

from typing import Any

import pytest
import torch


def test_openclip_module_imports() -> None:
    from voidface.models.clip.openclip import OpenClip

    assert OpenClip.spec.name == "openclip-vit-b-32"
    assert OpenClip.spec.family == "clip"
    assert OpenClip.spec.input_resolution == 224


def test_openclip_wrapper_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the wrapper: antialias 224 resize, CLIP mean/std normalize,
    output L2-normalized."""

    captured: dict[str, torch.Tensor] = {}

    class _StubOutput:
        def __init__(self, pooler: torch.Tensor) -> None:
            self.pooler_output = pooler

    class _StubClipVision:
        def __call__(self, *, pixel_values: torch.Tensor) -> _StubOutput:  # noqa: ANN001
            captured["pixel_values"] = pixel_values.detach().clone()
            return _StubOutput(torch.ones(pixel_values.size(0), 768) * 3.0)

        def to(self, _device: torch.device) -> "_StubClipVision":
            return self

        def eval(self) -> "_StubClipVision":
            return self

        def parameters(self):  # noqa: ANN201
            return iter([])

        @classmethod
        def from_pretrained(cls, _id: str) -> "_StubClipVision":
            return cls()

    monkeypatch.setattr(
        "transformers.CLIPVisionModel", _StubClipVision, raising=True
    )

    from voidface.models.clip.openclip import OpenClip

    encoder = OpenClip(device=torch.device("cpu"))
    image = torch.zeros(1, 3, 256, 256)
    image[:, 0] = 0.48145466  # mean of channel 0
    image[:, 1] = 0.4578275
    image[:, 2] = 0.40821073

    out = encoder(image)

    # After resize to 224 (constant per channel preserved), then
    # (x - mean) / std yields ~0 in every channel.
    seen = captured["pixel_values"]
    assert seen.shape == (1, 3, 224, 224)
    torch.testing.assert_close(
        seen[:, 0].mean(), torch.tensor(0.0), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        seen[:, 1].mean(), torch.tensor(0.0), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        seen[:, 2].mean(), torch.tensor(0.0), atol=1e-4, rtol=1e-4
    )

    # Output shape and L2 norm.
    assert out.embedding is not None
    assert out.embedding.shape == (1, 768)
    norms = out.embedding.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(1), atol=1e-5, rtol=1e-5)
