# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface report subcommand tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _write_solid(path: Path, colour: tuple[int, int, int], size: int = 64) -> None:
    Image.new("RGB", (size, size), colour).save(path)


def test_report_prints_psnr_and_ssim(tmp_path: Path, capsys) -> None:
    from voidface_cli.main import main

    original = tmp_path / "orig.png"
    protected = tmp_path / "prot.png"
    _write_solid(original, (128, 128, 128))
    _write_solid(protected, (129, 128, 128))

    rc = main(["report", str(original), str(protected)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PSNR" in out
    assert "SSIM" in out
    assert "L-inf" in out


def test_report_identity_gives_infinite_psnr(tmp_path: Path, capsys) -> None:
    from voidface_cli.main import main

    original = tmp_path / "orig.png"
    _write_solid(original, (200, 100, 50))

    rc = main(["report", str(original), str(original)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PSNR" in out
    # PSNR of an image with itself is infinite.
    assert "inf" in out.lower()


def test_report_shape_mismatch_errors(tmp_path: Path, capsys) -> None:
    from voidface_cli.main import main

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_solid(a, (0, 0, 0), size=64)
    _write_solid(b, (0, 0, 0), size=96)

    rc = main(["report", str(a), str(b)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "shape mismatch" in err
