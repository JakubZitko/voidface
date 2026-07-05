# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface init` — write a starter TOML for a common training scenario.

Preset content lives under ``voidface_cli/init_presets/<name>.toml``
so the presets are trivially reviewable and diffable without
scrolling through a huge Python string constant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    """Write (or print) the preset TOML the user requested."""
    preset_path = (
        Path(__file__).parent.parent / "init_presets" / f"{args.preset}.toml"
    )
    if not preset_path.exists():
        print(f"error: preset {args.preset!r} not found", file=sys.stderr)
        return 2
    text = preset_path.read_text()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0
