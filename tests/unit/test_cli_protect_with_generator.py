# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect --use-generator fast-path tests."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def _write_synthetic_checkpoint(path: Path) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def _write_synthetic_image(path: Path, size: int = 96) -> None:
    Image.new("RGB", (size, size), (192, 128, 96)).save(path)


def test_protect_with_generator_runs_end_to_end(tmp_path: Path) -> None:
    from voidface_cli.main import main

    image_path = tmp_path / "input.png"
    ckpt_path = tmp_path / "gen.pt"
    output_path = tmp_path / "protected.png"

    _write_synthetic_image(image_path)
    _write_synthetic_checkpoint(ckpt_path)

    rc = main(
        [
            "protect",
            str(image_path),
            "-o",
            str(output_path),
            "--use-generator",
            str(ckpt_path),
            "--device",
            "cpu",
            "--epsilon",
            "12",
        ]
    )
    assert rc == 0
    assert output_path.exists()


def test_protect_with_generator_reflects_epsilon_budget(tmp_path: Path) -> None:
    """The protected image should stay within L-inf budget of the input."""
    from voidface.util.image import load_image
    from voidface_cli.main import main

    image_path = tmp_path / "input.png"
    ckpt_path = tmp_path / "gen.pt"
    output_path = tmp_path / "protected.png"

    _write_synthetic_image(image_path, size=96)
    _write_synthetic_checkpoint(ckpt_path)

    rc = main(
        [
            "protect",
            str(image_path),
            "-o",
            str(output_path),
            "--use-generator",
            str(ckpt_path),
            "--device",
            "cpu",
            "--epsilon",
            "8",
        ]
    )
    assert rc == 0

    original = load_image(image_path)
    protected = load_image(output_path)
    linf = (original - protected).abs().max().item()
    # Allow one integer unit of JPEG/rounding slack (1/255 ~= 0.004).
    assert linf <= 8.0 / 255.0 + 0.005
