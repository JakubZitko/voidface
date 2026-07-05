# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Direct tests for pick_top_landmarks — the public wrapper that
decodes RetinaFace's per-anchor landmark regressions into a single
5-point landmark tensor per batch element.

These tests use a hand-authored TargetOutputs mock so they run
without network access or the real RetinaFace weights.
"""

from __future__ import annotations

import pytest
import torch

from voidface.models.base import TargetOutputs
from voidface.models.restorers.gfpgan import pick_top_landmarks


def _make_outputs(
    face_scores: torch.Tensor,
    landmarks_640: torch.Tensor,
) -> TargetOutputs:
    """Build a TargetOutputs with (N, A, 2) logits and (N, A, 10) landmarks.

    face_scores: (N, A) desired face-present probabilities.
    landmarks_640: (N, A, 10) landmark regressions in 640x640 pixel space.
    """
    # Reconstruct logits so softmax yields (1 - face_scores, face_scores).
    eps = 1e-6
    bg = torch.log((1.0 - face_scores).clamp(min=eps))
    fg = torch.log(face_scores.clamp(min=eps))
    logits = torch.stack([bg, fg], dim=-1)  # (N, A, 2)
    return TargetOutputs(logits=logits, aux={"landmarks": landmarks_640})


def test_returns_none_when_no_confident_face() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.3]])
    landmarks = torch.zeros(1, 3, 10)
    outputs = _make_outputs(scores, landmarks)
    image = torch.zeros(1, 3, 320, 320)
    assert pick_top_landmarks(outputs, image, threshold=0.5) is None


def test_returns_top_anchor_landmarks() -> None:
    scores = torch.tensor([[0.1, 0.9, 0.3]])
    # Anchor index 1 wins; give it distinctive landmark coordinates.
    landmarks_640 = torch.zeros(1, 3, 10)
    landmarks_640[0, 1] = torch.tensor(
        [193.0, 240.0, 319.0, 240.0, 256.0, 314.0, 201.0, 371.0, 313.0, 371.0]
    )
    outputs = _make_outputs(scores, landmarks_640)
    image = torch.zeros(1, 3, 640, 640)

    result = pick_top_landmarks(outputs, image, threshold=0.5)
    assert result is not None
    assert result.shape == (1, 5, 2)
    # 640x640 input scales 1:1 with the 640-native landmarks.
    assert result[0, 0].tolist() == pytest.approx([193.0, 240.0])
    assert result[0, 4].tolist() == pytest.approx([313.0, 371.0])


def test_scales_to_input_image_size() -> None:
    """Landmarks are returned in the input image's pixel space, not 640."""
    scores = torch.tensor([[0.9]])
    landmarks_640 = torch.tensor([[[320.0, 320.0, 320.0, 320.0, 320.0, 320.0,
                                    320.0, 320.0, 320.0, 320.0]]])
    outputs = _make_outputs(scores, landmarks_640)

    image = torch.zeros(1, 3, 320, 160)  # H=320, W=160
    result = pick_top_landmarks(outputs, image, threshold=0.5)
    assert result is not None
    # x scale = 160/640 = 0.25 -> 80. y scale = 320/640 = 0.5 -> 160.
    assert result[0, 0, 0].item() == pytest.approx(80.0)
    assert result[0, 0, 1].item() == pytest.approx(160.0)


def test_returns_none_when_outputs_missing_landmarks() -> None:
    outputs = TargetOutputs(logits=torch.zeros(1, 1, 2), aux=None)
    image = torch.zeros(1, 3, 64, 64)
    assert pick_top_landmarks(outputs, image, threshold=0.5) is None


def test_returns_none_when_outputs_missing_logits() -> None:
    outputs = TargetOutputs(logits=None, aux={"landmarks": torch.zeros(1, 1, 10)})
    image = torch.zeros(1, 3, 64, 64)
    assert pick_top_landmarks(outputs, image, threshold=0.5) is None


def test_batch_dim_preserved() -> None:
    scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    landmarks = torch.arange(2 * 2 * 10, dtype=torch.float32).view(2, 2, 10)
    outputs = _make_outputs(scores, landmarks)
    image = torch.zeros(2, 3, 640, 640)
    result = pick_top_landmarks(outputs, image, threshold=0.5)
    assert result is not None
    assert result.shape == (2, 5, 2)
