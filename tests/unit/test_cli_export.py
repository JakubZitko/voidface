# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface export CLI subcommand tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def _write_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def test_export_produces_onnx(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    out = tmp_path / "voidface.onnx"
    _write_checkpoint(ckpt)
    rc = main(["export", str(ckpt), str(out), "--example-resolution", "32"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


def test_export_with_quantize_produces_second_file(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    out = tmp_path / "voidface.onnx"
    _write_checkpoint(ckpt)
    rc = main(
        [
            "export",
            str(ckpt),
            str(out),
            "--example-resolution",
            "32",
            "--quantize",
            "int8",
        ]
    )
    assert rc == 0
    quantized = out.with_suffix(".int8.onnx")
    assert quantized.exists()
    assert quantized.stat().st_size < out.stat().st_size
