# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Round-trip tests for the canonical tensor image I/O layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from voidface.util.image import load_image, pil_to_tensor, save_image, tensor_to_pil


def test_pil_tensor_roundtrip_is_stable() -> None:
    pil = Image.new("RGB", (32, 32), (255, 0, 128))
    tensor = pil_to_tensor(pil)
    back = tensor_to_pil(tensor)
    assert back.mode == "RGB"
    assert back.size == pil.size
    assert list(back.getdata())[0] == (255, 0, 128)


def test_pil_to_tensor_shape_and_range() -> None:
    pil = Image.new("RGB", (10, 20), (128, 128, 128))
    tensor = pil_to_tensor(pil)
    assert tensor.shape == (3, 20, 10)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert torch.allclose(tensor.mean(), torch.tensor(128 / 255))


def test_save_and_load_lossless(tmp_path: Path) -> None:
    tensor = torch.zeros(3, 8, 8)
    tensor[0] = 1.0
    save_image(tensor, tmp_path / "red.png")
    reloaded = load_image(tmp_path / "red.png")
    assert torch.allclose(reloaded, tensor)


def test_tensor_to_pil_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        tensor_to_pil(torch.zeros(2, 8, 8))
