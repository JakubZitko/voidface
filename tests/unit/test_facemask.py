# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Face-region mask tests.

The Haar cascade in OpenCV doesn't reliably fire on synthetic
solid-colour blocks, so we cannot test the actual "face detected" path
without shipping a face image (which we deliberately don't). What we
CAN verify structurally: the mask function returns the right shape,
the right range, and the no-face fallback path.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cv2")


def test_no_face_falls_back_to_all_ones() -> None:
    from voidface.util.facemask import face_region_mask

    # Solid grey — the Haar cascade will not find a face here.
    image = torch.full((3, 64, 64), 0.5)
    mask = face_region_mask(image)
    assert mask.shape == (1, 64, 64)
    torch.testing.assert_close(mask, torch.ones_like(mask))


def test_shape_mismatch_raises() -> None:
    from voidface.util.facemask import face_region_mask

    with pytest.raises(ValueError, match=r"\(3, H, W\)"):
        face_region_mask(torch.zeros(1, 64, 64))


def test_batch_greater_than_one_raises() -> None:
    from voidface.util.facemask import face_region_mask

    with pytest.raises(ValueError, match="single image"):
        face_region_mask(torch.zeros(2, 3, 64, 64))


def test_squeezes_batch_dim_of_one() -> None:
    from voidface.util.facemask import face_region_mask

    mask = face_region_mask(torch.full((1, 3, 32, 32), 0.5))
    assert mask.shape == (1, 32, 32)
    assert 0.0 <= mask.min() <= mask.max() <= 1.0 + 1e-6
