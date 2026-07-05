# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# CoreML export for the shipped generator.
#
# CoreML runs the generator on Apple Silicon's Neural Engine (M1/M2/
# M3/M4 Macs, iPad, iPhone) at native tensor-op speed. On Intel Macs
# it falls back to CPU. The mlpackage produced here is what the
# desktop app (tools/desktop/) and any future iOS wrapper embed.
#
# coremltools is Apple Silicon-only in modern releases; the pyproject
# dep is gated on ``platform_system == 'Darwin' and platform_machine
# == 'arm64'``. This module raises a clear error on non-Apple-Silicon
# platforms.
#
# See Documentation/deployment/coreml.md.

"""CoreML export for the shipped generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from voidface.generator.architecture import Voidface

__all__ = ["CoreMlExportError", "export_generator_to_coreml"]


class CoreMlExportError(RuntimeError):
    """Raised when CoreML export cannot proceed on the current platform."""


def export_generator_to_coreml(
    generator: Voidface,
    output_path: Path,
    *,
    example_resolution: int = 256,
    minimum_deployment_target: str = "iOS17",
    quantize_weights: bool = True,
) -> None:
    """Export the generator to a CoreML ``.mlpackage`` directory.

    Args:
        generator: A trained (or freshly-instantiated) Voidface.
        output_path: The ``.mlpackage`` path. Coremltools writes a
            directory here; parent must exist.
        example_resolution: Side length in pixels for the tracing
            example. Runtime input is declared flexible so users can
            supply larger inputs.
        minimum_deployment_target: The CoreML deployment target
            string. ``"iOS17"`` gives access to fp16 activation on
            the Neural Engine.
        quantize_weights: When True, apply 8-bit weight quantization
            before writing. Reduces the shipped mlpackage size by
            ~4x with minimal drift.

    Raises:
        CoreMlExportError: When coremltools is not installed (Intel
            Mac, Linux, Windows).
    """
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover -- platform-gated
        msg = (
            "coremltools is not available on this platform. CoreML "
            "export requires Apple Silicon macOS with coremltools>=8.0. "
            "Use `voidface export --quantize int8` for a portable ONNX "
            "artifact instead."
        )
        raise CoreMlExportError(msg) from exc

    generator = generator.to("cpu").eval()
    example_input = torch.rand(1, 3, example_resolution, example_resolution)
    traced = torch.jit.trace(generator, example_input)

    if not output_path.parent.exists():
        msg = f"Output parent directory does not exist: {output_path.parent}"
        raise FileNotFoundError(msg)

    input_spec = ct.TensorType(
        name="input",
        shape=ct.Shape(
            shape=(
                1,
                3,
                ct.RangeDim(lower_bound=64, upper_bound=2048, default=example_resolution),
                ct.RangeDim(lower_bound=64, upper_bound=2048, default=example_resolution),
            )
        ),
    )
    mlmodel = ct.convert(
        traced,
        inputs=[input_spec],
        minimum_deployment_target=getattr(
            ct.target, minimum_deployment_target, ct.target.iOS17
        ),
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )

    if quantize_weights:
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=1024)
        )
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    mlmodel.save(str(output_path))
