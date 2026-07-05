# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Static (calibrated) int8 quantization tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def _calibration_iterator(n: int, size: int):
    import numpy as np

    rng = np.random.default_rng(seed=0)
    for _ in range(n):
        yield rng.random(size=(1, 3, size, size), dtype=np.float32)


def test_static_quantize_invalid_dtype_raises(tmp_path: Path) -> None:
    from voidface.export.quantize import quantize_onnx_generator_static

    with pytest.raises(ValueError, match="int8"):
        quantize_onnx_generator_static(
            tmp_path / "in.onnx",
            tmp_path / "out.onnx",
            _calibration_iterator(1, 32),
            weight_type="fp8",
        )


def test_static_quantize_output_smaller_and_close(tmp_path: Path) -> None:
    """Static-quantized model preserves runtime parity within a loose tolerance."""
    import numpy as np
    import onnxruntime as ort

    from voidface.export.onnx import export_generator_to_onnx
    from voidface.export.quantize import quantize_onnx_generator_static
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    torch.manual_seed(0)
    generator = Voidface(VoidfaceConfig(base_channels=16)).eval()
    fp32_path = tmp_path / "voidface.onnx"
    int8_path = tmp_path / "voidface.int8.onnx"

    export_generator_to_onnx(generator, fp32_path, example_resolution=64)
    quantize_onnx_generator_static(
        fp32_path, int8_path, _calibration_iterator(16, 64)
    )

    assert int8_path.exists()
    assert int8_path.stat().st_size < fp32_path.stat().st_size

    # Static quant records real activation ranges, so runtime parity
    # should be within a workable tolerance (unlike dynamic quant on
    # ConvNets which has op-support gaps we cannot check).
    input_tensor = torch.rand(1, 3, 64, 64)
    fp32_out = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    ).run(None, {"input": input_tensor.numpy()})[0]
    int8_out = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    ).run(None, {"input": input_tensor.numpy()})[0]

    # Output range is [0, 1] with a small delta; static-quant tolerance
    # under ~0.05 is a reasonable ship gate.
    assert np.allclose(fp32_out, int8_out, atol=0.05, rtol=1e-2)
