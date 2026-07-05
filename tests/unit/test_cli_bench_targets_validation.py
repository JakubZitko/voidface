# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface bench --targets validates the requested subset."""

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


def test_unknown_bench_target_returns_exit_code_2(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench",
            str(ckpt),
            str(images),
            "--targets",
            "vae",  # bench only accepts detector,recognizer
        ]
    )
    assert rc == 2


def test_typo_in_bench_targets_returns_exit_code_2(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    images = tmp_path / "images"
    images.mkdir()

    rc = main(
        [
            "bench",
            str(ckpt),
            str(images),
            "--targets",
            "detektor,recognizer",  # typo
        ]
    )
    assert rc == 2
