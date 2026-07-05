# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""ORT-Web .ort format conversion tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def test_missing_input_raises(tmp_path: Path) -> None:
    from voidface.export.ort import convert_onnx_to_ort

    with pytest.raises(FileNotFoundError, match="does not exist"):
        convert_onnx_to_ort(tmp_path / "missing.onnx", tmp_path)


def test_ort_conversion_produces_ort_file(tmp_path: Path) -> None:
    from voidface.export.onnx import export_generator_to_onnx
    from voidface.export.ort import convert_onnx_to_ort
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    torch.manual_seed(0)
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    onnx_path = tmp_path / "voidface.onnx"
    export_generator_to_onnx(generator, onnx_path, example_resolution=32)

    ort_path = convert_onnx_to_ort(onnx_path, output_dir=tmp_path / "ort")
    assert ort_path.exists()
    assert ort_path.suffix == ".ort"
