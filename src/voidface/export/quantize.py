# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Post-training dynamic int8 quantization of the exported ONNX
# generator. Reduces the shipped artifact from ~21 MB fp32 to ~5-6 MB
# int8 with a small numerical drift on the output. Static (calibrated)
# quantization lands in a follow-up when we have a calibration corpus.
#
# See Documentation/deployment/onnx.md.

"""Post-training int8 quantization of the shipped generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["quantize_onnx_generator"]


def quantize_onnx_generator(
    input_path: Path,
    output_path: Path,
    *,
    weight_type: str = "int8",
) -> None:
    """Dynamically quantize the ONNX generator to reduce shipped size.

    Args:
        input_path: A .onnx file produced by
            :func:`voidface.export.onnx.export_generator_to_onnx`.
        output_path: Where to write the quantized model.
        weight_type: ``"int8"`` (default) or ``"uint8"``. Static-shape
            deployment prefers int8; the browser demo may prefer
            uint8 for slightly faster WebAssembly kernels.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weight_map = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    if weight_type not in weight_map:
        msg = f"weight_type must be 'int8' or 'uint8', got {weight_type!r}."
        raise ValueError(msg)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=weight_map[weight_type],
    )
