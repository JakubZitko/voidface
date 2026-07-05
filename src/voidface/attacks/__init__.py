# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Attack techniques.

Modules:

    :mod:`voidface.attacks.pixel`     pixel-space PGD (baseline).
    :mod:`voidface.attacks.semantic`  sub-pixel geometric warp (phase R3).
    :mod:`voidface.attacks.iris`      iris region mask helper (phase R3).

See ``Documentation/attacks/`` for the design of each.
"""

from __future__ import annotations

from voidface.attacks.iris import iris_region_mask
from voidface.attacks.semantic import SemanticWarp, apply_semantic_warp

__all__ = ["SemanticWarp", "apply_semantic_warp", "iris_region_mask"]
