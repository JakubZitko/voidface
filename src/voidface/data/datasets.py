# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Face-image dataset for the R5 training loop.
#
# Voidface does NOT bundle a dataset. Users point the training loop at
# their own directory of face crops (FFHQ, CelebA-HQ, or any collection
# they have rights to). The dataset class here is the interface piece
# that lets train_generator() consume such a directory.
#
# Everything is deliberately simple. Any preprocessing beyond
# resize + horizontal-flip belongs in a subsystem-specific pipeline
# built on top of this one.
#
# See Documentation/training/overview.md.

"""Face-image dataset that walks a directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from voidface.util.image import pil_to_tensor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

__all__ = ["FolderImageDataset", "collect_image_paths"]


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def collect_image_paths(directory: Path, recursive: bool = True) -> list[Path]:
    """Walk ``directory`` and return every image path in stable order.

    Args:
        directory: A directory containing face crops.
        recursive: When True, recurse into subdirectories.

    Returns:
        A sorted ``list[Path]`` of every supported image file.
    """
    if not directory.is_dir():
        msg = f"Not a directory: {directory}"
        raise NotADirectoryError(msg)
    iterator: Iterable[Path] = (
        directory.rglob("*") if recursive else directory.iterdir()
    )
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


class FolderImageDataset(Dataset):
    """A minimal folder-of-faces dataset for :func:`train_generator`.

    Each item is a ``(3, H, W)`` float tensor in ``[0, 1]``. Any image
    aspect ratio is antialias-resized to ``resolution x resolution``.
    Random horizontal flip is applied when ``augment`` is True.

    Args:
        directory: The directory to walk.
        resolution: Side length in pixels for every returned tensor.
            Must be divisible by the generator's ``2^num_stages``
            (16 by default). The training loop asserts this at first
            forward.
        augment: When True, apply random horizontal flip during
            training. Default True. Disable for validation.
        recursive: When True, recurse into subdirectories.
        seed: Optional RNG seed for the flip.

    Raises:
        NotADirectoryError: When ``directory`` is not a directory.
        FileNotFoundError: When no supported images are found.
    """

    def __init__(
        self,
        directory: Path,
        resolution: int = 256,
        augment: bool = True,
        recursive: bool = True,
        seed: int | None = None,
    ) -> None:
        self._paths = collect_image_paths(directory, recursive=recursive)
        if not self._paths:
            msg = f"No supported image files under {directory}"
            raise FileNotFoundError(msg)
        self._resolution = resolution
        self._augment = augment
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self._paths[index]
        with Image.open(path) as image:
            tensor = pil_to_tensor(image)
        if tensor.shape[-2:] != (self._resolution, self._resolution):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self._resolution, self._resolution),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        if self._augment and torch.rand((), generator=self._rng).item() > 0.5:
            tensor = tensor.flip(dims=(-1,))
        return tensor.contiguous()
