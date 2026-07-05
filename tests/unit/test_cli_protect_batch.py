# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect batch mode tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _write_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def _write_images(directory: Path, count: int, size: int = 96) -> None:
    for idx in range(count):
        Image.new("RGB", (size, size), (idx * 40 % 256, 100, 100)).save(
            directory / f"face_{idx:04d}.png"
        )


def test_batch_missing_output_dir_errors(tmp_path: Path) -> None:
    from voidface_cli.main import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    _write_images(input_dir, count=2)
    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect",
            str(input_dir),
            "--use-generator",
            str(ckpt),
            "--device",
            "cpu",
        ]
    )
    assert rc == 2


def test_batch_missing_generator_errors(tmp_path: Path) -> None:
    from voidface_cli.main import main

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_images(input_dir, count=2)

    rc = main(
        [
            "protect",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    )
    assert rc == 2


def test_batch_processes_every_image(tmp_path: Path) -> None:
    from voidface_cli.main import main

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    _write_images(input_dir, count=3)
    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--use-generator",
            str(ckpt),
            "--device",
            "cpu",
            "--epsilon",
            "12",
        ]
    )
    assert rc == 0
    protected = sorted(output_dir.glob("*.protected.png"))
    assert len(protected) == 3
