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
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

__all__ = ["quantize_onnx_generator", "quantize_onnx_generator_static"]


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


def quantize_onnx_generator_static(
    input_path: Path,
    output_path: Path,
    calibration_inputs: Iterator[np.ndarray],
    *,
    activation_type: str = "int8",
    weight_type: str = "int8",
) -> None:
    """Statically quantize the ONNX generator using a calibration corpus.

    Unlike dynamic quantization, static quant records activation
    ranges from real inputs so runtime parity is much tighter. Uses
    ORT's CalibrationDataReader interface.

    Args:
        input_path: A .onnx file produced by
            :func:`voidface.export.onnx.export_generator_to_onnx`.
        output_path: Where to write the quantized model.
        calibration_inputs: Iterator of ``(1, 3, H, W)`` float32
            numpy arrays in ``[0, 1]``. Typically 32-256 real face
            crops that resemble the deployment distribution.
        activation_type: ``"int8"`` (default) or ``"uint8"``.
        weight_type: ``"int8"`` (default) or ``"uint8"``.

    Raises:
        ValueError: On unsupported dtype combinations.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantType,
        quantize_static,
    )

    activation_map = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    weight_map = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    if activation_type not in activation_map:
        msg = f"activation_type must be 'int8' or 'uint8', got {activation_type!r}."
        raise ValueError(msg)
    if weight_type not in weight_map:
        msg = f"weight_type must be 'int8' or 'uint8', got {weight_type!r}."
        raise ValueError(msg)

    class _Reader(CalibrationDataReader):
        def __init__(self, iterator: Iterator[np.ndarray]) -> None:
            self._iter = iter(iterator)

        def get_next(self) -> dict[str, np.ndarray] | None:
            try:
                array = next(self._iter)
            except StopIteration:
                return None
            return {"input": array}

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=_Reader(calibration_inputs),
        activation_type=activation_map[activation_type],
        weight_type=weight_map[weight_type],
    )
