# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface protect-video` — per-frame G with optical-flow temporal blending."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from voidface.util.facemask import face_region_mask
from voidface.util.flow import farneback_flow, warp_forward
from voidface.util.log import configure_logging, get_logger
from voidface.util.video import iter_frames, write_video
from voidface_cli.common import load_generator_checkpoint, resolve_device


def run(args: argparse.Namespace) -> int:
    """Video protection via the deploy fast path with optional temporal blending."""
    configure_logging(level="INFO")
    log = get_logger("voidface.cli.protect-video")
    device = resolve_device(args.device)

    if not args.use_generator.exists():
        log.error(
            "generator.checkpoint.not_found",
            path=str(args.use_generator),
            hint="produce one with `voidface train cfg.toml` or download a release .pt",
        )
        return 2
    if not args.input.exists():
        log.error(
            "video.input.not_found",
            path=str(args.input),
            hint="point --input at a real video file",
        )
        return 2
    if args.epsilon < 1 or args.epsilon > 255:
        log.error(
            "epsilon.out_of_range",
            got=args.epsilon,
            hint="--epsilon is expressed as N/255; use 1-255 (typical 8-16)",
        )
        return 2
    if not 0.0 <= args.temporal_blend <= 1.0:
        log.error(
            "temporal_blend.out_of_range",
            got=args.temporal_blend,
            hint="--temporal-blend is a mixing ratio in [0.0, 1.0]",
        )
        return 2

    generator, config = load_generator_checkpoint(args.use_generator, device, log)
    metadata, frames = iter_frames(args.input)
    log.info(
        "video.opened",
        path=str(args.input),
        width=metadata.width,
        height=metadata.height,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        temporal_blend=args.temporal_blend,
    )

    divisor = 1 << config.num_stages
    padded_h = (metadata.height + divisor - 1) // divisor * divisor
    padded_w = (metadata.width + divisor - 1) // divisor * divisor
    pad_bottom = padded_h - metadata.height
    pad_right = padded_w - metadata.width
    epsilon = args.epsilon / 255.0
    alpha = float(args.temporal_blend)

    from collections.abc import Iterator

    def _protected_frames() -> Iterator[torch.Tensor]:
        prev_frame_cpu: torch.Tensor | None = None
        prev_delta_cpu: torch.Tensor | None = None
        with torch.no_grad():
            for index, frame in enumerate(frames):
                frame_batched = frame.unsqueeze(0).to(device)
                if pad_bottom > 0 or pad_right > 0:
                    frame_padded = F.pad(
                        frame_batched,
                        (0, pad_right, 0, pad_bottom),
                        mode="reflect",
                    )
                else:
                    frame_padded = frame_batched
                fresh = generator(frame_padded, epsilon=epsilon)
                fresh = fresh[..., : metadata.height, : metadata.width]
                fresh_delta = (fresh - frame_batched).squeeze(0).cpu()

                if prev_delta_cpu is None or alpha <= 0.0:
                    delta = fresh_delta
                else:
                    flow = farneback_flow(prev_frame_cpu, frame)  # type: ignore[arg-type]
                    warped_prev = warp_forward(prev_delta_cpu, flow)
                    delta = alpha * warped_prev + (1.0 - alpha) * fresh_delta

                if args.face_mask:
                    mask = face_region_mask(frame)
                    delta = delta * mask

                delta = delta.clamp(-epsilon, +epsilon)
                protected_frame = (frame + delta).clamp(0.0, 1.0)

                if (index + 1) % 30 == 0:
                    log.info(
                        "video.progress",
                        frame=index + 1,
                        total=metadata.frame_count,
                    )

                prev_frame_cpu = frame.detach()
                prev_delta_cpu = delta.detach()
                yield protected_frame

    write_video(args.output, _protected_frames(), metadata, codec=args.codec)
    log.info("video.written", path=str(args.output))
    return 0
