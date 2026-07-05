# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Deploy-time face-region mask.
#
# The trained generator G outputs a delta over the whole image. That
# is efficient for training (loss covers every ensemble target), but
# at deploy time we usually want the perturbation to land ONLY on the
# face region — smooth backgrounds (walls, sky, plain skin outside
# the face) render adversarial noise visibly, and there is no attack
# benefit to perturbing the background at all.
#
# This module uses OpenCV's shipped Haar cascade (frontal_face) to
# find the face bbox, expands it, and returns a feathered ``[0, 1]``
# alpha mask the caller multiplies onto the delta. When no face is
# found the mask is all ones — the caller can then decide whether to
# fail loudly or ship the un-masked delta.

"""Face-region mask for the deploy path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["face_region_mask"]


def face_region_mask(
    image: Tensor,
    *,
    expand: float = 0.30,
    feather_pixels: int = 24,
) -> Tensor:
    """Return a soft ``[0, 1]`` face-region mask for ``image``.

    Args:
        image: A ``(3, H, W)`` or ``(1, 3, H, W)`` float tensor in
            ``[0, 1]``.
        expand: Fractional expansion of the detected bbox before
            feathering. ``0.30`` grows a 100-px bbox by 30% on every
            side. Covers ears, hairline, and jawline that the Haar
            cascade under-tightens.
        feather_pixels: Softness of the mask's edge, in pixels. The
            mask fades from 1 inside the expanded bbox to 0 over
            this many pixels.

    Returns:
        A ``(1, H, W)`` float tensor in ``[0, 1]``. When no face is
        detected, returns a tensor of ones (the caller decides how to
        handle that).
    """
    import cv2

    if image.dim() == 4:
        if image.size(0) != 1:
            msg = "face_region_mask expects a single image."
            raise ValueError(msg)
        image = image.squeeze(0)
    if image.dim() != 3 or image.size(0) != 3:
        msg = f"Expected (3, H, W), got {tuple(image.shape)}."
        raise ValueError(msg)

    _, height, width = image.shape
    array = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4)
    if len(faces) == 0:
        return torch.ones(1, height, width, dtype=image.dtype, device=image.device)

    # Take the largest face.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    ex = int(w * expand)
    ey = int(h * expand)
    x0 = max(0, x - ex)
    y0 = max(0, y - ey)
    x1 = min(width, x + w + ex)
    y1 = min(height, y + h + ey)

    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    if feather_pixels > 0:
        kernel_size = max(3, 2 * feather_pixels + 1)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=feather_pixels)

    return torch.from_numpy(mask).to(device=image.device, dtype=image.dtype).unsqueeze(0)
