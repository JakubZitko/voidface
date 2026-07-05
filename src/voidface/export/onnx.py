# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# ONNX export for the shipped generator.
#
# ONNX is the cross-platform shipping format:
#   * Windows / Linux inference via ONNX Runtime.
#   * Browser demo via ONNX Runtime Web with the WebGPU execution
#     provider (converted to .ort in a follow-up commit).
#
# CoreML export for Apple Silicon lives in coreml.py — a separate
# module because the coremltools dep is macOS-only.
#
# See Documentation/deployment/onnx.md.

"""ONNX export for the shipped generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from voidface.generator.architecture import Voidface

__all__ = ["export_generator_to_onnx"]


def export_generator_to_onnx(
    generator: Voidface,
    output_path: Path,
    *,
    example_resolution: int = 256,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> None:
    """Export the generator to ONNX.

    Args:
        generator: A trained (or freshly-instantiated) Voidface. Put
            in eval mode by this function.
        output_path: Where to write the .onnx file. Parent directory
            must exist.
        example_resolution: Side length in pixels of the tracing
            example. Must be divisible by the generator's stride
            (16 by default). Runtime shape can differ when
            ``dynamic_axes`` is True.
        opset: ONNX opset version. 17 is the R5 target — broadly
            supported by ONNX Runtime, includes GridSample for future
            R5.5 experiments with the semantic-warp attack.
        dynamic_axes: When True, mark height and width as dynamic so a
            single ONNX file serves multiple input resolutions. When
            False, the exported graph is locked to
            ``example_resolution x example_resolution``.
    """
    generator = generator.eval()
    example_input = torch.rand(
        1, 3, example_resolution, example_resolution, device="cpu"
    )
    # Move generator to CPU for tracing so we do not accidentally bake
    # a CUDA / MPS device tag into the exported graph.
    generator_cpu = generator.to("cpu")

    if not output_path.parent.exists():
        msg = f"Output parent directory does not exist: {output_path.parent}"
        raise FileNotFoundError(msg)

    dynamic_axes_arg: dict[str, dict[int, str]] | None = None
    if dynamic_axes:
        dynamic_axes_arg = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        generator_cpu,
        example_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes=dynamic_axes_arg,
        do_constant_folding=True,
    )
