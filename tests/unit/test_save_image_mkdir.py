# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""save_image creates parent directories as needed."""

from __future__ import annotations

from pathlib import Path

import torch

from voidface.util.image import save_image


def test_save_image_creates_parent_dir(tmp_path: Path) -> None:
    tensor = torch.zeros(3, 8, 8)
    out = tmp_path / "does" / "not" / "exist" / "yet" / "out.png"
    save_image(tensor, out)
    assert out.exists()


def test_save_image_no_parent_creation_when_dir_exists(tmp_path: Path) -> None:
    tensor = torch.zeros(3, 8, 8)
    out = tmp_path / "out.png"  # tmp_path itself exists
    save_image(tensor, out)
    assert out.exists()
