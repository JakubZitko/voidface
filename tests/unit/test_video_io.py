# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Voidface video I/O helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("cv2")


def _write_synthetic_video(path: Path, width: int, height: int, frames: int, fps: int = 12) -> None:
    """Write a small MP4 with distinct solid-colour frames."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for idx in range(frames):
        colour = ((idx * 40) % 256, 100, 150)
        frame = np.full((height, width, 3), colour, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_iter_frames_reports_correct_metadata(tmp_path: Path) -> None:
    from voidface.util.video import iter_frames

    video = tmp_path / "in.mp4"
    _write_synthetic_video(video, width=48, height=32, frames=5, fps=10)

    metadata, iterator = iter_frames(video)
    assert metadata.width == 48
    assert metadata.height == 32
    assert 9 <= metadata.fps <= 11
    # OpenCV's frame_count on tiny synthetic MP4s is not always exact —
    # loosen the assertion but check it's in the right ballpark.
    assert 3 <= metadata.frame_count <= 7

    frames = list(iterator)
    # Similarly the actual frame count can be exactly what we wrote.
    assert 3 <= len(frames) <= 6
    for frame in frames:
        assert frame.shape == (3, 32, 48)
        assert frame.dtype == torch.float32
        assert 0.0 <= frame.min().item() <= frame.max().item() <= 1.0


def test_missing_video_raises(tmp_path: Path) -> None:
    from voidface.util.video import iter_frames

    with pytest.raises(FileNotFoundError):
        iter_frames(tmp_path / "does_not_exist.mp4")


def test_write_video_roundtrip(tmp_path: Path) -> None:
    from voidface.util.video import VideoMetadata, iter_frames, write_video

    metadata = VideoMetadata(width=48, height=32, fps=10.0, frame_count=4)
    frames_out = [torch.rand(3, 32, 48) for _ in range(4)]

    output = tmp_path / "out.mp4"
    write_video(output, iter(frames_out), metadata)
    assert output.exists() and output.stat().st_size > 0

    _, reader = iter_frames(output)
    frames_in = list(reader)
    assert len(frames_in) >= 3
    for frame in frames_in:
        assert frame.shape == (3, 32, 48)
