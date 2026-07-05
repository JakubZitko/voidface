# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Iris-region binary mask helper.

Face recognizers assign heavy weight to iris texture. Humans do not
perceive sub-millimeter iris changes at ordinary viewing distance.
This asymmetry is the largest single perceptibility-vs-signal
opportunity in a face image.

This module gives the pixel PGD loop a way to concentrate its budget
inside the iris region. See :mod:`voidface.attacks.pixel` for the
outer PGD loop; the iris mask is intended to be multiplied into the
delta before the epsilon clamp, letting the iris region take a
higher local ceiling (see ``iris_epsilon_ratio`` below).

The design in ``Documentation/attacks/iris.md`` allows the iris
budget to be up to 2x the global epsilon; that choice is left to
the caller. This module ships the mask; composing it with PGD is
the caller's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["iris_region_mask"]


def iris_region_mask(
    landmarks: Tensor,
    height: int,
    width: int,
    radius_frac: float = 0.028,
    softness_px: float = 1.5,
) -> Tensor:
    """Produce a soft binary mask covering the iris region.

    Args:
        landmarks: ``(N, 5, 2)`` tensor of 5-point landmarks in pixel
            coordinates matching (``height``, ``width``). Order:
            left eye, right eye, nose, left mouth, right mouth (same
            as :data:`voidface.data.align.FFHQ_LANDMARKS_512`).
        height: Image height in pixels.
        width: Image width in pixels.
        radius_frac: Iris radius as a fraction of the inter-ocular
            distance. Default 0.028 matches the mean human iris
            diameter of ~11.7 mm relative to a 65 mm inter-pupillary
            distance.
        softness_px: Gaussian falloff width at the mask edge, in
            pixels. Softens the border so PGD's epsilon boundary
            doesn't create a sharp transition line at the iris
            perimeter.

    Returns:
        ``(N, 1, height, width)`` float tensor in ``[0, 1]``. Uses
        the same dtype and device as ``landmarks``.
    """
    if landmarks.dim() != 3 or landmarks.size(1) < 2 or landmarks.size(2) != 2:
        msg = (
            f"landmarks must be (N, >=2, 2); got {tuple(landmarks.shape)}"
        )
        raise ValueError(msg)

    n = landmarks.size(0)
    device = landmarks.device
    dtype = landmarks.dtype

    left_eye = landmarks[:, 0, :]
    right_eye = landmarks[:, 1, :]
    interocular = torch.linalg.norm(right_eye - left_eye, dim=-1)
    radius = interocular * float(radius_frac)

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)

    mask = torch.zeros((n, 1, height, width), device=device, dtype=dtype)
    for eye in (left_eye, right_eye):
        center = eye[:, None, None, :]
        dist = torch.linalg.norm(grid[None, ...] - center, dim=-1)
        r_expanded = radius[:, None, None]
        soft_edge = (r_expanded - dist).clamp(min=-softness_px) / softness_px
        contribution = soft_edge.clamp(0.0, 1.0).unsqueeze(1)
        mask = torch.maximum(mask, contribution)
    return mask
