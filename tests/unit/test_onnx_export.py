# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""ONNX export smoke tests.

Verifies:

  * The generator exports to a real .onnx file that ONNX Runtime can
    load.
  * The ONNX output matches the PyTorch reference to within a small
    numerical tolerance.
  * dynamic_axes=True actually produces a dynamic-shape graph
    (loading with a different H/W than the tracing example still
    works).

These tests require ``onnxruntime``, which is in the [export] optional
dep group. Skip when not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_export_produces_valid_onnx_file(tmp_path: Path) -> None:
    from voidface.export.onnx import export_generator_to_onnx
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    generator = Voidface(VoidfaceConfig(base_channels=8))
    out_path = tmp_path / "voidface.onnx"
    export_generator_to_onnx(generator, out_path, example_resolution=32)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_export_output_matches_pytorch(tmp_path: Path) -> None:
    import onnxruntime as ort

    from voidface.export.onnx import export_generator_to_onnx
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    torch.manual_seed(0)
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    out_path = tmp_path / "voidface.onnx"
    export_generator_to_onnx(generator, out_path, example_resolution=32)

    input_tensor = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        py_out = generator(input_tensor).cpu().numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": input_tensor.cpu().numpy()})[0]

    # Small numerical drift is expected from op fusion + fp32 rounding.
    import numpy as np

    assert np.allclose(py_out, onnx_out, atol=1e-4, rtol=1e-4)


def test_dynamic_axes_accept_different_resolution(tmp_path: Path) -> None:
    import onnxruntime as ort

    from voidface.export.onnx import export_generator_to_onnx
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    out_path = tmp_path / "voidface.onnx"
    export_generator_to_onnx(
        generator, out_path, example_resolution=32, dynamic_axes=True
    )

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    # Trace at 32, run at 64.
    input_tensor = torch.rand(1, 3, 64, 64)
    onnx_out = session.run(None, {"input": input_tensor.cpu().numpy()})[0]
    assert onnx_out.shape == (1, 3, 64, 64)
