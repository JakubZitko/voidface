# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface package subcommand tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def _write_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 42, "state_dict": generator.state_dict(), "config": config}, path)


def test_package_minimal(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    out_dir = tmp_path / "release"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "package",
            str(ckpt),
            str(out_dir),
            "--example-resolution",
            "32",
        ]
    )
    assert rc == 0
    assert (out_dir / "voidface.onnx").exists()
    assert (out_dir / "voidface.int8.onnx").exists()
    assert (out_dir / "CHECKSUMS.sha256").exists()
    assert (out_dir / "MANIFEST.json").exists()
    assert (out_dir / "README").exists()

    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    assert manifest["training_step"] == 42
    assert "onnx" in manifest["artifacts"]
    assert "onnx" in manifest["checksums"]


def test_package_with_calibration_dir(tmp_path: Path) -> None:
    from PIL import Image

    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    out_dir = tmp_path / "release"
    calibration = tmp_path / "cal"
    calibration.mkdir()
    for i in range(4):
        Image.new("RGB", (32, 32), (i * 40, 100, 150)).save(calibration / f"c{i}.png")

    _write_checkpoint(ckpt)

    rc = main(
        [
            "package",
            str(ckpt),
            str(out_dir),
            "--example-resolution",
            "32",
            "--calibration-dir",
            str(calibration),
        ]
    )
    assert rc == 0
    assert (out_dir / "voidface.static-int8.onnx").exists()
    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    assert "static-int8" in manifest["artifacts"]
