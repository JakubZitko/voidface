# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Iris region mask correctness tests."""

from __future__ import annotations

import pytest
import torch

from voidface.attacks.iris import iris_region_mask


def _canonical_landmarks(n: int = 1) -> torch.Tensor:
    """5-point FFHQ landmark set as a (n, 5, 2) tensor."""
    from voidface.data.align import FFHQ_LANDMARKS_512

    single = torch.tensor(FFHQ_LANDMARKS_512, dtype=torch.float32)
    return single.unsqueeze(0).expand(n, -1, -1).contiguous()


def test_mask_shape_and_dtype_match_input() -> None:
    lm = _canonical_landmarks(2)
    mask = iris_region_mask(lm, height=512, width=512)
    assert mask.shape == (2, 1, 512, 512)
    assert mask.dtype == lm.dtype


def test_mask_range_zero_to_one() -> None:
    lm = _canonical_landmarks()
    mask = iris_region_mask(lm, height=512, width=512)
    assert mask.min().item() >= 0.0
    assert mask.max().item() <= 1.0 + 1e-6


def test_mask_center_of_iris_is_ones() -> None:
    """The exact center of each iris (eye landmark) should be 1.0."""
    lm = _canonical_landmarks()
    mask = iris_region_mask(lm, height=512, width=512)

    left_eye = lm[0, 0].round().long()
    right_eye = lm[0, 1].round().long()
    assert mask[0, 0, left_eye[1], left_eye[0]].item() == pytest.approx(1.0)
    assert mask[0, 0, right_eye[1], right_eye[0]].item() == pytest.approx(1.0)


def test_mask_outside_face_is_zero() -> None:
    """Far corner of the image should be well outside the iris mask."""
    lm = _canonical_landmarks()
    mask = iris_region_mask(lm, height=512, width=512)
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 500, 500].item() == 0.0


def test_mask_covers_reasonable_area() -> None:
    """Coverage should be small but non-zero — humans have small irises."""
    lm = _canonical_landmarks()
    mask = iris_region_mask(lm, height=512, width=512)
    total = 512 * 512
    covered = (mask > 0.5).sum().item()
    fraction = covered / total
    assert 0.00005 < fraction < 0.02


def test_mask_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="landmarks must be"):
        iris_region_mask(torch.zeros(1, 5), height=64, width=64)


def test_mask_batch_dim_preserved() -> None:
    lm = _canonical_landmarks(4)
    mask = iris_region_mask(lm, height=256, width=256)
    assert mask.shape[0] == 4


def test_radius_parameter_scales_coverage() -> None:
    lm = _canonical_landmarks()
    small = iris_region_mask(lm, height=512, width=512, radius_frac=0.02)
    large = iris_region_mask(lm, height=512, width=512, radius_frac=0.05)
    assert (large > 0.5).sum().item() > (small > 0.5).sum().item()
