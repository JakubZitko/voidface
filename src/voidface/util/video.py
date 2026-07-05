# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Video I/O helpers built on OpenCV.
#
# Voidface's per-frame video protection runs the generator on every
# frame independently. That is what R6.7a ships. Optical-flow-based
# temporal coherence — warping delta forward between keyframes to
# avoid the "boiling" texture that per-frame independent perturbation
# produces — is R6.7b (planned).

"""OpenCV-based video I/O for the video protection paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

__all__ = ["VideoMetadata", "iter_frames", "write_video"]


@dataclass(frozen=True)
class VideoMetadata:
    """Descriptor for a source video."""

    width: int
    height: int
    fps: float
    frame_count: int


def iter_frames(path: Path) -> tuple[VideoMetadata, Iterator[Tensor]]:
    """Read a video into an iterator of ``(3, H, W)`` float tensors.

    Args:
        path: Path to any container OpenCV can read (MP4, MOV, MKV,
            WEBM, ...). Codec support depends on the FFmpeg
            OpenCV was built against.

    Returns:
        A pair ``(metadata, iterator)``. Consuming the iterator
        streams frames one at a time — memory footprint stays small
        for long videos.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        msg = f"OpenCV could not open the video: {path}"
        raise FileNotFoundError(msg)

    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    def _iterator() -> Iterator[Tensor]:
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                # OpenCV yields BGR; canonical tensor is RGB.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                yield tensor
        finally:
            capture.release()

    return metadata, _iterator()


def write_video(
    path: Path,
    frames: Iterator[Tensor],
    metadata: VideoMetadata,
    *,
    codec: str = "mp4v",
) -> None:
    """Write a stream of ``(3, H, W)`` float tensors as a video.

    Args:
        path: Output video path. The container is inferred from the
            extension.
        frames: Iterator of ``(3, H, W)`` float tensors in ``[0, 1]``,
            RGB order.
        metadata: The metadata to use for width/height/fps. When any
            of these differs from the actual frames, OpenCV's writer
            will produce a corrupted file — validate upstream.
        codec: A four-character codec FourCC. ``"mp4v"`` is the
            broadly-compatible default; ``"avc1"`` (H.264) may not be
            available depending on OpenCV build.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        msg = f"OpenCV could not open the writer for: {path}"
        raise RuntimeError(msg)

    try:
        for tensor in frames:
            array = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().byte()
            array = array.permute(1, 2, 0).cpu().numpy()  # -> H W 3 RGB
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()
