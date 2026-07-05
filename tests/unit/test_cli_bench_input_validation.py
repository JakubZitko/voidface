# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface bench flag validation."""

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


def test_zero_resolution_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench", str(ckpt), str(images),
            "--resolution", "0",
        ]
    )
    assert rc == 2


def test_negative_limit_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench", str(ckpt), str(images),
            "--limit", "-1",
        ]
    )
    assert rc == 2


def test_detection_threshold_above_one_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench", str(ckpt), str(images),
            "--detection-threshold", "1.5",
        ]
    )
    assert rc == 2


def test_detection_threshold_negative_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench", str(ckpt), str(images),
            "--detection-threshold", "-0.5",
        ]
    )
    assert rc == 2
