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
