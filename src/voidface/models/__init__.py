# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Differentiable surrogates of the attacker's models.

Sub-packages:

    :mod:`voidface.models.detectors`    face detectors.
    :mod:`voidface.models.recognizers`  identity encoders.
    :mod:`voidface.models.vaes`         diffusion VAEs.
    :mod:`voidface.models.clip`         CLIP-family image encoders.
    :mod:`voidface.models.restorers`    face restorers (bilevel target).

All targets implement the :class:`voidface.models.base.EnsembleTarget`
protocol.
"""

from __future__ import annotations

from voidface.models.base import EnsembleTarget, TargetOutputs, TargetSpec

__all__ = [
    "EnsembleTarget",
    "TargetOutputs",
    "TargetSpec",
]
