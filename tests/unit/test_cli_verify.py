# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface verify subcommand tests."""

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


def test_verify_matches_produced_bundle(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    bundle = tmp_path / "release"
    _write_checkpoint(ckpt)

    rc = main(["package", str(ckpt), str(bundle), "--example-resolution", "32"])
    assert rc == 0

    rc = main(["verify", str(bundle)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_verify_missing_checksums_errors(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    rc = main(["verify", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a voidface bundle" in err or "not found" in err


def test_verify_detects_tampered_file(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    bundle = tmp_path / "release"
    _write_checkpoint(ckpt)

    rc = main(["package", str(ckpt), str(bundle), "--example-resolution", "32"])
    assert rc == 0

    # Tamper with the fp32 ONNX file.
    onnx = bundle / "voidface.onnx"
    original = onnx.read_bytes()
    onnx.write_bytes(original + b"\x00tampered\x00")

    rc = main(["verify", str(bundle)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out or "FAIL" in out
