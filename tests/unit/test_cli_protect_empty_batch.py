# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Batch mode with an empty input directory returns exit code 2."""

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


def test_empty_batch_dir_returns_exit_code_2(tmp_path: Path) -> None:
    from voidface_cli.main import main

    empty = tmp_path / "empty"
    empty.mkdir()
    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect",
            str(empty),
            "--output-dir",
            str(tmp_path / "out"),
            "--use-generator",
            str(ckpt),
        ]
    )
    assert rc == 2


def test_batch_dir_with_only_non_image_files_returns_exit_code_2(tmp_path: Path) -> None:
    from voidface_cli.main import main

    weird = tmp_path / "weird"
    weird.mkdir()
    # A .txt file is not a supported image extension.
    (weird / "notes.txt").write_text("hi")

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)

    rc = main(
        [
            "protect",
            str(weird),
            "--output-dir",
            str(tmp_path / "out"),
            "--use-generator",
            str(ckpt),
        ]
    )
    assert rc == 2
