# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface package --dry-run test."""

from __future__ import annotations

from pathlib import Path

import torch

from voidface.generator.architecture import Voidface, VoidfaceConfig


def _write_checkpoint(path: Path) -> None:
    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save(
        {"step": 0, "state_dict": generator.state_dict(), "config": config},
        path,
    )


def test_dry_run_prints_planned_artifacts_without_writing(
    tmp_path: Path, capsys
) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    out_dir = tmp_path / "release"
    rc = main(
        [
            "package", str(ckpt), str(out_dir), "--name", "test-model",
            "--dry-run",
        ]
    )
    assert rc == 0

    output = capsys.readouterr().out
    assert "--- package dry run ---" in output
    assert "test-model" in output
    assert "onnx (fp32)" in output
    assert "int8 (dynamic)" in output
    assert "static-int8" not in output  # --calibration-dir not passed
    assert "coreml" not in output       # --coreml not passed

    # Real assertion: nothing was written.
    assert not out_dir.exists()


def test_dry_run_includes_coreml_when_flagged(
    tmp_path: Path, capsys
) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "package", str(ckpt), str(tmp_path / "r"),
            "--coreml", "--dry-run",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "coreml (.mlpackage" in output


def test_dry_run_includes_static_int8_with_calibration_dir(
    tmp_path: Path, capsys
) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    cal = tmp_path / "cal"
    cal.mkdir()

    rc = main(
        [
            "package", str(ckpt), str(tmp_path / "r"),
            "--calibration-dir", str(cal), "--dry-run",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    assert "static-int8" in output
