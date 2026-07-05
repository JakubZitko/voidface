# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Structured logging setup.

Voidface uses structlog for structured logs. Every subsystem obtains a
logger via :func:`get_logger`. Applications call :func:`configure_logging`
once at startup; libraries never touch the root config.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(level: str = "INFO", *, use_colors: bool | None = None) -> None:
    """Configure the root logger and structlog rendering.

    Applications call this once at startup. Libraries must not.

    Args:
        level: A logging level name, e.g. ``"DEBUG"``, ``"INFO"``.
        use_colors: If ``None``, colors are enabled when stderr is a TTY.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        msg = f"Unknown log level: {level!r}."
        raise ValueError(msg)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
    )

    colorize = sys.stderr.isatty() if use_colors is None else use_colors
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(colors=colorize),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)
