# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Import-time smoke tests. These are the last-line-of-defense sanity
checks that keep the tree assembling into a real Python package."""

from __future__ import annotations


def test_package_importable() -> None:
    import voidface

    assert voidface.__version__


def test_ensemble_target_protocol_importable() -> None:
    from voidface.models.base import EnsembleTarget, TargetOutputs, TargetSpec

    assert EnsembleTarget is not None
    assert TargetOutputs is not None
    assert TargetSpec is not None
