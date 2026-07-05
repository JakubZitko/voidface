# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface protect on a non-existent image returns a clean error.

Regression guard: R7.86 replaced the raw PIL FileNotFoundError
traceback with a structured error log + exit code 2.
"""

from __future__ import annotations

from pathlib import Path


def test_missing_image_returns_exit_code_2(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ghost = tmp_path / "does_not_exist.png"
    rc = main(["protect", str(ghost)])
    assert rc == 2


def test_missing_batch_dir_and_file_returns_exit_code_2(tmp_path: Path) -> None:
    """A path that is neither a file nor a directory still returns 2."""
    from voidface_cli.main import main

    ghost = tmp_path / "does_not_exist"
    rc = main(["protect", str(ghost)])
    assert rc == 2
