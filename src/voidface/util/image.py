# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Image I/O and tensor conversion. Voidface uses a single canonical
# tensor layout everywhere:
#
#     shape:  (N, 3, H, W)     or   (3, H, W)
#     dtype:  float32
#     range:  [0.0, 1.0]
#     order:  RGB
#
# All other layouts (PIL, uint8, BGR, HWC) are converted at the I/O
# boundary and are not passed into subsystem code.

"""Image I/O and canonical tensor conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch import Tensor

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "load_image",
    "pil_to_tensor",
    "save_image",
    "tensor_to_pil",
]


def load_image(path: Path) -> Tensor:
    """Load an image from disk into the canonical tensor form.

    Args:
        path: A filesystem path to a PNG, JPEG, WEBP, or any other
            format Pillow can decode.

    Returns:
        A float tensor of shape ``(3, H, W)`` with values in
        ``[0.0, 1.0]``.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        return pil_to_tensor(image)


def save_image(tensor: Tensor, path: Path) -> None:
    """Save a canonical tensor to disk.

    The output format is inferred from the file extension. PNG is
    recommended for lossless storage; JPEG at Q=95 or higher is
    acceptable for size at the cost of a single JPEG pass.

    Args:
        tensor: A tensor of shape ``(3, H, W)`` or ``(1, 3, H, W)``.
        path: The output filesystem path.
    """
    tensor_to_pil(tensor).save(path)


def pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image to the canonical tensor form."""
    # np.array copies the PIL buffer; np.asarray sometimes returns a
    # read-only view which torch.from_numpy warns about because it
    # cannot enforce PyTorch's mutability invariants. We copy on the
    # way in so downstream ops are safe.
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert a canonical tensor to a PIL image.

    Accepts either ``(3, H, W)`` or ``(1, 3, H, W)`` inputs. Values are
    clamped to ``[0, 1]`` before conversion.
    """
    if tensor.dim() == 4:
        if tensor.size(0) != 1:
            msg = f"Expected batch size 1, got {tensor.size(0)}."
            raise ValueError(msg)
        tensor = tensor.squeeze(0)
    if tensor.dim() != 3 or tensor.size(0) != 3:
        msg = f"Expected (3, H, W) tensor, got shape {tuple(tensor.shape)}."
        raise ValueError(msg)
    array = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode="RGB")
