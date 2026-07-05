# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Optical flow warp tests."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cv2")


def test_farneback_flow_of_identical_frames_is_near_zero() -> None:
    from voidface.util.flow import farneback_flow

    torch.manual_seed(0)
    frame = torch.rand(3, 64, 64)
    flow = farneback_flow(frame, frame)
    assert flow.shape == (64, 64, 2)
    # Identical frames should yield near-zero flow everywhere.
    max_magnitude = float(abs(flow).max())
    assert max_magnitude < 0.5, max_magnitude


def test_warp_forward_preserves_shape() -> None:
    import numpy as np

    from voidface.util.flow import warp_forward

    delta = torch.rand(3, 32, 48) * 0.05
    flow = np.zeros((32, 48, 2), dtype=np.float32)
    warped = warp_forward(delta, flow)
    assert warped.shape == delta.shape


def test_warp_forward_identity_flow_is_near_no_op() -> None:
    import numpy as np

    from voidface.util.flow import warp_forward

    delta = torch.rand(3, 32, 48) * 0.05
    flow = np.zeros((32, 48, 2), dtype=np.float32)
    warped = warp_forward(delta, flow)
    torch.testing.assert_close(warped, delta, atol=1e-5, rtol=1e-5)


def test_farneback_shape_mismatch_raises() -> None:
    from voidface.util.flow import farneback_flow

    prev = torch.rand(3, 32, 32)
    curr = torch.rand(3, 40, 40)
    with pytest.raises(ValueError, match="Shape mismatch"):
        farneback_flow(prev, curr)
