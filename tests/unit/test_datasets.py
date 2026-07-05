# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""FolderImageDataset — walks a directory, resizes, augments, batches
correctly, and integrates with torch DataLoader for the training
loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from voidface.data.datasets import FolderImageDataset, collect_image_paths


def _write_synthetic_faces(root: Path, count: int, size: int = 128) -> None:
    """Emit ``count`` random-color PNGs, half with ``.png`` half with ``.jpg``."""
    for idx in range(count):
        colour = (
            (idx * 37) % 256,
            (idx * 71) % 256,
            (idx * 113) % 256,
        )
        image = Image.new("RGB", (size, size), colour)
        suffix = ".png" if idx % 2 == 0 else ".jpg"
        image.save(root / f"face_{idx:04d}{suffix}")


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FolderImageDataset(tmp_path)


def test_non_directory_raises(tmp_path: Path) -> None:
    file = tmp_path / "not_a_dir.png"
    file.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        collect_image_paths(file)


def test_collects_and_iterates_stable_order(tmp_path: Path) -> None:
    _write_synthetic_faces(tmp_path, count=4)
    a = FolderImageDataset(tmp_path, resolution=64, augment=False)
    b = FolderImageDataset(tmp_path, resolution=64, augment=False)
    assert len(a) == 4
    # Ordering must be stable across dataset instances so training
    # runs are reproducible.
    for i in range(len(a)):
        torch.testing.assert_close(a[i], b[i])


def test_output_shape_and_range(tmp_path: Path) -> None:
    _write_synthetic_faces(tmp_path, count=3)
    dataset = FolderImageDataset(tmp_path, resolution=32, augment=False)
    item = dataset[0]
    assert item.shape == (3, 32, 32)
    assert item.dtype == torch.float32
    assert item.min().item() >= 0.0
    assert item.max().item() <= 1.0


def test_dataloader_yields_expected_batch(tmp_path: Path) -> None:
    _write_synthetic_faces(tmp_path, count=6)
    dataset = FolderImageDataset(tmp_path, resolution=32, augment=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch.shape == (2, 3, 32, 32)


def test_augmentation_can_flip(tmp_path: Path) -> None:
    """With augment=True and a fixed seed we should see at least one
    flipped image across many samples of the same underlying image.

    Uses a horizontally-asymmetric checker pattern so a flip is
    observable at the tensor level (a solid-color image would be
    flip-invariant).
    """
    import numpy as np

    # Half-black half-white — flipping swaps the halves, definitely
    # changes the tensor.
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, :32] = 255
    Image.fromarray(array).save(tmp_path / "asym.png")

    dataset = FolderImageDataset(tmp_path, resolution=64, augment=True, seed=0)
    ref = dataset[0]
    flipped_seen = 0
    for _ in range(20):
        item = dataset[0]
        if not torch.equal(item, ref):
            flipped_seen += 1
    assert flipped_seen > 0
