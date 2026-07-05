# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Config loader.

Voidface config is TOML. See ``samples/configs/`` for reference. The
loader is deliberately minimal — it returns a plain nested dict and
does not enforce any schema. Subsystems that consume config should
parse the shape they need into a dataclass locally.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["load_config"]


def load_config(path: Path) -> dict[str, Any]:
    """Read a TOML config file into a nested dict."""
    with path.open("rb") as handle:
        return tomllib.load(handle)
