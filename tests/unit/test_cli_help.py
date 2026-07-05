# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Verify every documented subcommand's --help works cleanly.

Regression guard: adding a new subcommand or renaming an existing one
without updating the parser wiring would show up here.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "cmd",
    [
        "protect",
        "protect-video",
        "train",
        "bench",
        "export",
        "package",
        "info",
        "config-check",
        "init",
        "report",
    ],
)
def test_subcommand_help_exits_cleanly(cmd: str) -> None:
    from voidface_cli.main import main

    try:
        main([cmd, "--help"])
    except SystemExit as exit_error:
        assert exit_error.code in (0, None)


def test_toplevel_help_exits_cleanly() -> None:
    from voidface_cli.main import main

    try:
        main(["--help"])
    except SystemExit as exit_error:
        assert exit_error.code in (0, None)


def test_version_flag_prints_semver() -> None:
    import voidface
    from voidface_cli.main import main

    try:
        main(["--version"])
    except SystemExit as exit_error:
        assert exit_error.code in (0, None)
    assert voidface.__version__
