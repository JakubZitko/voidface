# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect --iris-boost / --iris-ratio parser tests."""

from __future__ import annotations

import argparse

import pytest

from voidface_cli.main import _build_parser


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    return _build_parser()


def test_iris_boost_defaults_off(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["protect", "input.jpg"])
    assert args.iris_boost is False


def test_iris_boost_enables(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["protect", "input.jpg", "--iris-boost"])
    assert args.iris_boost is True


def test_iris_ratio_defaults_to_two(parser: argparse.ArgumentParser) -> None:
    """Default matches Documentation/attacks/iris.md's `2 · epsilon`."""
    args = parser.parse_args(["protect", "input.jpg"])
    assert args.iris_ratio == 2.0


def test_iris_ratio_overridable(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(
        ["protect", "input.jpg", "--iris-boost", "--iris-ratio", "3.0"]
    )
    assert args.iris_boost is True
    assert args.iris_ratio == 3.0


def test_dump_iris_mask_defaults_none(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(["protect", "input.jpg"])
    assert args.dump_iris_mask is None


def test_dump_iris_mask_accepts_path(parser: argparse.ArgumentParser) -> None:
    from pathlib import Path

    args = parser.parse_args(
        ["protect", "input.jpg", "--iris-boost", "--dump-iris-mask", "out/mask.png"]
    )
    assert args.dump_iris_mask == Path("out/mask.png")


def test_iris_ratio_below_one_rejected(tmp_path) -> None:  # noqa: ANN001
    """`--iris-ratio 0.5` is nonsensical (would shrink budget); return 2."""
    from voidface_cli.main import main

    img = tmp_path / "x.png"
    img.write_bytes(b"not a real image")  # File exists so we reach validation.
    rc = main(
        [
            "protect", str(img), "--iris-boost", "--iris-ratio", "0.5",
        ]
    )
    assert rc == 2


def test_negative_semantic_warp_rejected(tmp_path) -> None:  # noqa: ANN001
    """`--semantic-warp -1.0` is nonsensical (max-displacement must be positive)."""
    from voidface_cli.main import main

    img = tmp_path / "x.png"
    img.write_bytes(b"not a real image")
    rc = main(
        [
            "protect", str(img), "--semantic-warp", "-1.0",
        ]
    )
    assert rc == 2


def test_epsilon_out_of_range_rejected(tmp_path) -> None:  # noqa: ANN001
    """--epsilon is N/255; values below 1 or above 255 are rejected."""
    from voidface_cli.main import main

    img = tmp_path / "x.png"
    img.write_bytes(b"not a real image")
    # 0 is degenerate (no perturbation).
    assert main(["protect", str(img), "--epsilon", "0"]) == 2
    # 256 exceeds byte range.
    assert main(["protect", str(img), "--epsilon", "256"]) == 2
    # Negative is nonsensical.
    assert main(["protect", str(img), "--epsilon", "-5"]) == 2


def test_zero_steps_rejected(tmp_path) -> None:  # noqa: ANN001
    """--steps 0 means no PGD iterations — degenerate."""
    from voidface_cli.main import main

    img = tmp_path / "x.png"
    img.write_bytes(b"not a real image")
    assert main(["protect", str(img), "--steps", "0"]) == 2


def test_negative_refine_steps_rejected(tmp_path) -> None:  # noqa: ANN001
    """--refine-steps -3 is nonsensical."""
    from voidface_cli.main import main

    img = tmp_path / "x.png"
    img.write_bytes(b"not a real image")
    assert main(["protect", str(img), "--refine-steps", "-3"]) == 2
