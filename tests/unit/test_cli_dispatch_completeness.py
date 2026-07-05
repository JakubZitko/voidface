# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Guard against dispatch table drift.

Every subparser wired into `_build_parser` must have a matching
handler in `main()`'s dispatch dict. If someone adds a subcommand
to the parser but forgets to wire it into main(), running that
subcommand would silently print --help and return 2. This test
catches the omission at import time.
"""

from __future__ import annotations

import inspect
import re


def test_every_subparser_has_a_dispatch_entry() -> None:
    from voidface_cli.main import _build_parser, main

    parser = _build_parser()

    subparsers_action = None
    for action in parser._actions:  # noqa: SLF001
        if action.__class__.__name__ == "_SubParsersAction":
            subparsers_action = action
            break

    assert subparsers_action is not None, "expected argparse subparsers action"
    parser_names = set(subparsers_action.choices)

    source = inspect.getsource(main)
    # Extract the dispatch dict keys — every `"foo":` line inside main().
    dispatch_keys = set(re.findall(r'"([^"]+)":\s*_cmd_', source))

    missing_from_dispatch = parser_names - dispatch_keys
    missing_from_parser = dispatch_keys - parser_names

    assert not missing_from_dispatch, (
        f"subparsers without a dispatch entry: {sorted(missing_from_dispatch)}"
    )
    assert not missing_from_parser, (
        f"dispatch entries without a subparser: {sorted(missing_from_parser)}"
    )


def test_dispatch_covers_all_documented_subcommands() -> None:
    """The set of subcommands must match Documentation/status.md's promise.

    Guards the human-facing status doc against silent CLI drift.
    """
    from voidface_cli.main import _build_parser

    parser = _build_parser()
    subparsers = None
    for action in parser._actions:  # noqa: SLF001
        if action.__class__.__name__ == "_SubParsersAction":
            subparsers = action
            break
    assert subparsers is not None
    live = set(subparsers.choices)

    expected = {
        "protect", "protect-video", "report", "train", "bench",
        "export", "package", "verify", "info", "config-check", "init",
    }
    assert live == expected, f"CLI drift: live={sorted(live)} expected={sorted(expected)}"
