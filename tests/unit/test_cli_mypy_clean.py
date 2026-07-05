# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Regression guard — voidface_cli must remain mypy-clean.

Every file under tools/cli/voidface_cli/ (common.py and all 11
subcommand modules) passes strict mypy after R7.102-R7.121. If a
future refactor accidentally reintroduces untyped locals or
object-typed device params, this test catches it in CI.

Skipped in environments without mypy installed to avoid slowing
down a fresh checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _cli_dir() -> Path:
    # tests/unit/<this-file>.py  →  <repo-root>/tools/cli/voidface_cli/
    return Path(__file__).parent.parent.parent / "tools" / "cli" / "voidface_cli"


def test_cli_common_and_commands_are_mypy_clean() -> None:
    if shutil.which("mypy") is None:
        pytest.skip("mypy not installed in this environment")

    cli_dir = _cli_dir()
    targets = [str(cli_dir / "common.py"), str(cli_dir / "commands")]

    result = subprocess.run(
        [sys.executable, "-m", "mypy", *targets],
        cwd=str(cli_dir.parent.parent.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"mypy reported errors:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
