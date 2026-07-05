# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""bench --strict flag is wired and parseable.

The full end-to-end ship-gate check requires a real trained
checkpoint and heavyweight target models, so this test verifies
only that the flag is accepted and the defaults match the values
in Documentation/process/release.md.
"""

from __future__ import annotations

import argparse

import pytest
from voidface_cli.main import _build_parser


@pytest.fixture
def parser() -> argparse.ArgumentParser:
    return _build_parser()


def test_bench_strict_flag_accepted(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(
        [
            "bench",
            "checkpoint.pt",
            "images/",
            "--strict",
        ]
    )
    assert args.strict is True


def test_bench_strict_defaults_match_release_doc(parser: argparse.ArgumentParser) -> None:
    """Ship-gate defaults must stay pinned to Documentation/process/release.md.

    If any of these change, update release.md in the same commit so
    the runbook stays truthful.
    """
    args = parser.parse_args(["bench", "checkpoint.pt", "images/"])
    assert args.strict is False
    assert args.strict_detection_asr == 0.60
    assert args.strict_identity_cos == 0.20
    assert args.strict_psnr == 30.0
    assert args.strict_ssim == 0.92


def test_bench_strict_thresholds_overridable(parser: argparse.ArgumentParser) -> None:
    args = parser.parse_args(
        [
            "bench",
            "checkpoint.pt",
            "images/",
            "--strict",
            "--strict-detection-asr",
            "0.75",
            "--strict-psnr",
            "35.0",
        ]
    )
    assert args.strict_detection_asr == 0.75
    assert args.strict_psnr == 35.0
    assert args.strict_identity_cos == 0.20
    assert args.strict_ssim == 0.92
