# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Voidface — adversarial face-region blindfold.

This is the top-level package. Real code lives inside subsystem
packages under :mod:`voidface.core`, :mod:`voidface.models`,
:mod:`voidface.attacks`, :mod:`voidface.generator`, :mod:`voidface.data`,
:mod:`voidface.eval`, :mod:`voidface.export`, and :mod:`voidface.util`.

The only names re-exported here are ``__version__`` and a small handful of
public types that constitute the stable API. Everything else must be
imported from its subsystem module directly.
"""

from __future__ import annotations

__all__ = ["__version__"]

# Session 2026-07-05 walked from 0.0.1 through R7.33 with 80+ commits.
# The version pinned here is the currently-shipping semver.
__version__ = "0.1.0"
