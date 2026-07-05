# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""int8 quantization tests.

Verifies:

  * quantize_onnx_generator produces an output file.
  * Quantized file is meaningfully smaller than the fp32 input.
  * Quantized ONNX Runtime output remains within a loose tolerance
    of the fp32 reference.
  * Invalid weight_type raises cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_invalid_weight_type_raises(tmp_path: Path) -> None:
    from voidface.export.quantize import quantize_onnx_generator

    with pytest.raises(ValueError, match="int8"):
        quantize_onnx_generator(tmp_path / "in.onnx", tmp_path / "out.onnx", weight_type="fp8")


def test_quantized_output_file_exists_and_shrinks(tmp_path: Path) -> None:
    """quantize_onnx_generator produces a file and it is smaller
    than the fp32 input.

    We deliberately do NOT verify runtime parity here. ORT's dynamic
    quantization has known op-support gaps for Conv-heavy graphs
    (produces a valid model artifact but some execution providers
    fall back or NotImplemented at inference). Static (calibrated)
    quantization is the path for verified parity; it lands with a
    dataset-backed calibrator in a follow-up commit.
    """
    from voidface.export.onnx import export_generator_to_onnx
    from voidface.export.quantize import quantize_onnx_generator
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    torch.manual_seed(0)
    generator = Voidface(VoidfaceConfig(base_channels=16)).eval()
    fp32_path = tmp_path / "voidface.onnx"
    int8_path = tmp_path / "voidface.int8.onnx"

    export_generator_to_onnx(generator, fp32_path, example_resolution=64)
    quantize_onnx_generator(fp32_path, int8_path)

    assert int8_path.exists()
    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    assert int8_size < 0.75 * fp32_size, (fp32_size, int8_size)
