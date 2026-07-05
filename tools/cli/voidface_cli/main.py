# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Voidface command-line entry point.
#
# This file is a scaffold. Subcommands and argument parsing are added
# incrementally as the underlying subsystems land. See
# Documentation/deployment/ for the runtime wrapper this eventually
# invokes.

"""Voidface CLI entry point."""

from __future__ import annotations

import sys

import voidface

__all__ = ["app", "main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m voidface_cli.main`` and the console script.

    Args:
        argv: Argument vector without the program name. Defaults to
            :data:`sys.argv[1:]`.

    Returns:
        The exit code to hand back to the shell.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args in ([], ["-h"], ["--help"]):
        _print_usage()
        return 0
    if args == ["--version"]:
        print(f"voidface {voidface.__version__}")
        return 0

    print(
        f"voidface {voidface.__version__}: CLI is scaffolded but not "
        f"implemented yet. See Documentation/architecture.md.",
        file=sys.stderr,
    )
    return 2


def app() -> None:
    """Console-script wrapper. Exits the process with :func:`main`'s code."""
    raise SystemExit(main())


def _print_usage() -> None:
    print(
        "voidface — adversarial face-region blindfold\n"
        "\n"
        "Usage:\n"
        "  voidface [--version | --help]\n"
        "  voidface protect <image> [-o <out>]                (TBD)\n"
        "  voidface protect-video <clip> [-o <out>]           (TBD)\n"
        "  voidface report <image>                            (TBD)\n"
        "\n"
        "See Documentation/ for the full design."
    )


if __name__ == "__main__":
    raise SystemExit(main())
