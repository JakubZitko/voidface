# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface export --quantize-static-dir end-to-end test."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def _write_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def _write_calibration_images(directory: Path, count: int, size: int = 32) -> None:
    for idx in range(count):
        Image.new("RGB", (size, size), (idx * 40 % 256, 100, 150)).save(
            directory / f"cal_{idx:04d}.png"
        )


def test_export_with_quantize_static_writes_extra_file(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    onnx = tmp_path / "voidface.onnx"
    calibration = tmp_path / "cal"
    calibration.mkdir()

    _write_checkpoint(ckpt)
    _write_calibration_images(calibration, count=8)

    rc = main(
        [
            "export",
            str(ckpt),
            str(onnx),
            "--example-resolution",
            "32",
            "--quantize-static-dir",
            str(calibration),
            "--quantize-static-samples",
            "4",
        ]
    )
    assert rc == 0
    assert onnx.exists()
    assert onnx.with_suffix(".static-int8.onnx").exists()
    assert onnx.with_suffix(".static-int8.onnx").stat().st_size < onnx.stat().st_size
