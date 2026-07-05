# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Convert an ONNX file to the ORT format that ONNX Runtime Web
# consumes directly. The .ort format bundles the model with the
# operators the runtime needs, cutting startup latency and download
# size for the browser demo.
#
# See Documentation/deployment/onnx.md.

"""ORT format conversion for the browser demo."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["OrtConversionError", "convert_onnx_to_ort"]


class OrtConversionError(RuntimeError):
    """Raised when the onnxruntime.tools converter fails or is missing."""


def convert_onnx_to_ort(
    input_path: Path,
    output_dir: Path,
    *,
    optimization_style: str = "Fixed",
) -> Path:
    """Convert ``input_path`` (.onnx) into an .ort file under ``output_dir``.

    Args:
        input_path: An .onnx file. Both fp32 and int8-quantized inputs
            work.
        output_dir: Directory where the .ort file is written. Created
            if missing.
        optimization_style: ORT optimization style — ``"Fixed"``
            (default; bakes optimizations at conversion time) or
            ``"Runtime"`` (defers to runtime).

    Returns:
        The path to the produced .ort file.

    Raises:
        OrtConversionError: When the ``onnxruntime.tools``
            subcommand fails or is not available.
    """
    if not input_path.exists():
        msg = f"Input ONNX file does not exist: {input_path}"
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the input into the output directory because the converter
    # writes the .ort file alongside the source and does not accept a
    # separate output location.
    staged_input = output_dir / input_path.name
    if staged_input.resolve() != input_path.resolve():
        shutil.copy2(input_path, staged_input)

    completed = subprocess.run(  # noqa: S603 -- fully-controlled args
        [
            sys.executable,
            "-m",
            "onnxruntime.tools.convert_onnx_models_to_ort",
            str(staged_input),
            "--optimization_style",
            optimization_style,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        msg = (
            f"onnxruntime.tools converter exited {completed.returncode}. "
            f"stdout: {completed.stdout.strip()!r} "
            f"stderr: {completed.stderr.strip()!r}"
        )
        raise OrtConversionError(msg)

    ort_path = staged_input.with_suffix(".ort")
    if not ort_path.exists():
        # Some ORT versions emit .with_runtime_opt.ort — try that.
        candidate = staged_input.with_suffix(".with_runtime_opt.ort")
        if candidate.exists():
            return candidate
        msg = (
            f"onnxruntime.tools converter returned success but no .ort file "
            f"was produced next to {staged_input}. Contents of {output_dir}: "
            f"{sorted(p.name for p in output_dir.iterdir())}"
        )
        raise OrtConversionError(msg)
    return ort_path
