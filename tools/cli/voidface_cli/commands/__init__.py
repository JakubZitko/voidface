# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Subcommand handler modules.

Each module holds one subcommand's runtime logic (the `_cmd_*`
handler and any helpers it privately owns). `main.py` still owns the
argparse wiring and dispatch table; this split keeps main.py small
enough to reason about and lets each subcommand be tested in
isolation.
"""
