# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect-video flag validation."""

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


def test_epsilon_out_of_range_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake mp4 body")

    rc = main(
        [
            "protect-video",
            str(inp), str(tmp_path / "out.mp4"),
            "--use-generator", str(ckpt),
            "--epsilon", "0",
        ]
    )
    assert rc == 2


def test_temporal_blend_above_one_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake mp4 body")

    rc = main(
        [
            "protect-video",
            str(inp), str(tmp_path / "out.mp4"),
            "--use-generator", str(ckpt),
            "--temporal-blend", "1.5",
        ]
    )
    assert rc == 2


def test_temporal_blend_negative_rejected(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"fake mp4 body")

    rc = main(
        [
            "protect-video",
            str(inp), str(tmp_path / "out.mp4"),
            "--use-generator", str(ckpt),
            "--temporal-blend", "-0.1",
        ]
    )
    assert rc == 2
