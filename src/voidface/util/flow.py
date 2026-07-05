# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Optical-flow-based temporal warping helpers.
#
# Voidface's video protection path applies the generator to each frame
# independently by default. The result is visually correct but has a
# characteristic "boiling" texture at the perturbation because
# consecutive frames get delta patterns that are independent samples
# from the generator's per-pixel output.
#
# This module warps the previous frame's delta by dense optical flow
# so subsequent frames share most of their perturbation with their
# predecessor — the boiling collapses into a smooth temporal signal.
# Uses Farnebäck flow from OpenCV — no external ML deps, fast enough
# to run alongside G on a per-frame basis.

"""Optical-flow-based temporal warping for video protection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["farneback_flow", "warp_forward"]


def farneback_flow(prev: Tensor, curr: Tensor) -> np.ndarray:
    """Dense optical flow from ``prev`` to ``curr``.

    Args:
        prev: A ``(3, H, W)`` float tensor in ``[0, 1]``.
        curr: Same shape and range as ``prev``.

    Returns:
        A ``(H, W, 2)`` float32 numpy array of ``(dx, dy)`` per pixel.
    """
    import cv2

    if prev.shape != curr.shape:
        msg = f"Shape mismatch: prev={tuple(prev.shape)} curr={tuple(curr.shape)}"
        raise ValueError(msg)
    if prev.dim() != 3 or prev.size(0) != 3:
        msg = f"Expected (3, H, W), got {tuple(prev.shape)}"
        raise ValueError(msg)

    def _luminance(t: Tensor) -> np.ndarray:
        # Rec. 709 luma weights on RGB channels.
        rgb = t.detach().cpu().numpy()
        luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        return (luma * 255.0).clip(0, 255).astype(np.uint8)

    return cv2.calcOpticalFlowFarneback(
        _luminance(prev), _luminance(curr),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )


def warp_forward(source: Tensor, flow: np.ndarray) -> Tensor:
    """Warp ``source`` by the given flow field.

    Args:
        source: A ``(C, H, W)`` float tensor. Typically the previous
            frame's delta.
        flow: ``(H, W, 2)`` optical flow, produced by
            :func:`farneback_flow`.

    Returns:
        A ``(C, H, W)`` warped copy of ``source``.
    """
    import cv2

    _, h, w = source.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    channels = source.detach().cpu().numpy()
    warped = np.zeros_like(channels)
    for c in range(channels.shape[0]):
        warped[c] = cv2.remap(
            channels[c], map_x, map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
    return torch.from_numpy(warped).to(source.dtype)
