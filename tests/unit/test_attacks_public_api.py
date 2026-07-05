# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Lock the voidface.attacks public API surface.

The `attacks` subpackage exports a small set of composable attack
primitives. This test guarantees the names stay stable — anyone
importing `from voidface.attacks import ...` must not have their
code break silently when we reorganize modules underneath.
"""

from __future__ import annotations


def test_iris_region_mask_reexported() -> None:
    from voidface.attacks import iris_region_mask
    from voidface.attacks.iris import iris_region_mask as canonical

    assert iris_region_mask is canonical


def test_semantic_warp_class_reexported() -> None:
    from voidface.attacks import SemanticWarp
    from voidface.attacks.semantic import SemanticWarp as canonical

    assert SemanticWarp is canonical


def test_apply_semantic_warp_reexported() -> None:
    from voidface.attacks import apply_semantic_warp
    from voidface.attacks.semantic import apply_semantic_warp as canonical

    assert apply_semantic_warp is canonical


def test_public_api_matches_declared_all() -> None:
    """__all__ is authoritative — anything importable must be in it."""
    import voidface.attacks as attacks_pkg

    assert set(attacks_pkg.__all__) == {
        "SemanticWarp",
        "apply_semantic_warp",
        "iris_region_mask",
    }
