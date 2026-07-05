# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect-video end-to-end CLI test.

Writes a small synthetic MP4, runs the protect-video subcommand with a
tiny generator checkpoint, and verifies the output video reads back
with the expected frame count and dimensions. Covers both temporal-
blend modes (0 = per-frame independent, >0 = flow-warped).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("cv2")


def _write_synthetic_video(path: Path, width: int, height: int, frames: int) -> None:
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 12, (width, height))
    for idx in range(frames):
        colour = ((idx * 40) % 256, 100, 150)
        writer.write(np.full((height, width, 3), colour, dtype=np.uint8))
    writer.release()


def _write_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def test_protect_video_per_frame(tmp_path: Path) -> None:
    from voidface.util.video import iter_frames
    from voidface_cli.main import main

    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    ckpt = tmp_path / "gen.pt"
    _write_synthetic_video(src, width=48, height=32, frames=6)
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect-video",
            str(src),
            str(dst),
            "--use-generator",
            str(ckpt),
            "--device",
            "cpu",
            "--epsilon",
            "12",
            "--temporal-blend",
            "0.0",  # per-frame independent
        ]
    )
    assert rc == 0
    assert dst.exists() and dst.stat().st_size > 0

    metadata, frames = iter_frames(dst)
    assert metadata.width == 48
    assert metadata.height == 32
    frame_list = list(frames)
    assert 4 <= len(frame_list) <= 7


def test_protect_video_temporal_blend(tmp_path: Path) -> None:
    from voidface.util.video import iter_frames
    from voidface_cli.main import main

    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    ckpt = tmp_path / "gen.pt"
    _write_synthetic_video(src, width=32, height=32, frames=5)
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect-video",
            str(src),
            str(dst),
            "--use-generator",
            str(ckpt),
            "--device",
            "cpu",
            "--epsilon",
            "12",
            "--temporal-blend",
            "0.7",  # coherence on
        ]
    )
    assert rc == 0
    assert dst.exists()

    _, frames = iter_frames(dst)
    frame_list = list(frames)
    assert 3 <= len(frame_list) <= 6
