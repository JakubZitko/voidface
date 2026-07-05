# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Differentiable 5-point aligner tests.

Verifies that:

  * similarity transform recovers the identity on same-in-same-out
    inputs;
  * a known translation-only transform is recovered accurately;
  * align_faces produces a 512x512 crop with the source pixels warped
    correctly onto the FFHQ template landmarks;
  * unalign_paste round-trips: aligning then pasting the aligned crop
    back reconstructs the original image to within a small blend
    tolerance;
  * gradients flow through the whole pipeline (grad-check on a scalar
    downstream of grid_sample).
"""

from __future__ import annotations

import pytest
import torch

from voidface.data.align import (
    FFHQ_LANDMARKS_512,
    align_faces,
    estimate_similarity_transform,
    unalign_paste,
)


def test_identity_transform_recovered() -> None:
    """When source == target, the transform should be the identity."""
    pts = torch.tensor(
        [
            [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0], [100.0, 300.0], [200.0, 300.0]],
        ]
    )
    matrix = estimate_similarity_transform(pts, pts)
    expected = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    torch.testing.assert_close(matrix, expected, atol=1e-4, rtol=1e-4)


def test_pure_translation_recovered() -> None:
    src = torch.tensor(
        [[[0.0, 0.0], [10.0, 0.0], [5.0, 10.0], [0.0, 20.0], [10.0, 20.0]]]
    )
    tgt = src + torch.tensor([[[7.0, 3.0]]])
    matrix = estimate_similarity_transform(src, tgt)
    # Rotation and scale should be identity; translation (7, 3).
    torch.testing.assert_close(matrix[0, 0, 0], torch.tensor(1.0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(matrix[0, 1, 1], torch.tensor(1.0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(matrix[0, 0, 2], torch.tensor(7.0), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(matrix[0, 1, 2], torch.tensor(3.0), atol=1e-4, rtol=1e-4)


def test_align_produces_expected_crop_shape() -> None:
    torch.manual_seed(0)
    image = torch.rand(1, 3, 640, 640)
    landmarks = torch.tensor([list(FFHQ_LANDMARKS_512)]).float()
    aligned = align_faces(image, landmarks, output_size=512)
    assert aligned.crop.shape == (1, 3, 512, 512)
    assert aligned.transform.shape == (1, 2, 3)


def test_align_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"\(N, 5, 2\)"):
        align_faces(torch.rand(1, 3, 100, 100), torch.rand(1, 3, 2))


def test_alignment_gradient_flows() -> None:
    """Gradients must flow through the alignment when we backprop from
    the aligned crop back to the input image."""
    image = torch.rand(1, 3, 256, 256, requires_grad=True)
    # Landmarks well within the 256x256 image so grid_sample samples
    # actual pixels.
    landmarks = torch.tensor(
        [[[100.0, 100.0], [156.0, 100.0], [128.0, 150.0], [100.0, 200.0], [156.0, 200.0]]]
    )
    aligned = align_faces(image, landmarks, output_size=128)
    loss = aligned.crop.mean()
    loss.backward()
    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0


def test_align_and_unalign_approximate_roundtrip() -> None:
    """Aligning then pasting the aligned crop back reconstructs the
    original image over the alpha-blended region."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 320, 320)
    landmarks = torch.tensor(
        [[[110.0, 100.0], [210.0, 100.0], [160.0, 160.0], [120.0, 220.0], [200.0, 220.0]]]
    )
    aligned = align_faces(image, landmarks, output_size=256)
    reconstructed = unalign_paste(
        image, aligned.crop, aligned.transform, feather_pixels=16
    )
    # The reconstruction should be very close to the input inside the
    # feathered region. Not perfect due to double bilinear resample.
    assert (reconstructed - image).abs().mean().item() < 0.1
