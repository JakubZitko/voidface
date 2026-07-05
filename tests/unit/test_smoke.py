# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Import-time smoke tests. Last-line-of-defense sanity checks that keep
the tree assembling into a real Python package."""

from __future__ import annotations


def test_package_importable() -> None:
    import voidface

    assert voidface.__version__


def test_ensemble_target_protocol_importable() -> None:
    from voidface.models.base import EnsembleTarget, TargetOutputs, TargetSpec

    assert EnsembleTarget is not None
    assert TargetOutputs is not None
    assert TargetSpec is not None


def test_cli_help_returns_zero() -> None:
    from voidface_cli.main import main

    try:
        main(["--help"])
    except SystemExit as exit_error:
        assert exit_error.code in (0, None)


def test_cli_version_returns_zero() -> None:
    from voidface_cli.main import main

    try:
        main(["--version"])
    except SystemExit as exit_error:
        assert exit_error.code in (0, None)
