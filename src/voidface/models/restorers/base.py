# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Restorer interface — the bilevel-training abstraction.
#
# A Restorer is a differentiable image-to-image transform interposed
# between the adversarial image and the ensemble targets. The training
# loop samples one Restorer per step; each target then sees
# ``restorer(x + delta)`` instead of the raw perturbed image. This is
# what makes Voidface's perturbation survive an attacker's post-hoc
# GFPGAN / CodeFormer / Real-ESRGAN pass.
#
# The interface is intentionally minimal so multiple concrete restorers
# (identity, GFPGAN, CodeFormer, Real-ESRGAN) satisfy it uniformly.
#
# See Documentation/training/bilevel-adversarial.md and
# Documentation/models/restorers.md.

"""Restorer interface for the bilevel objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Restorer", "RestorerSpec"]


@dataclass(frozen=True, slots=True)
class RestorerSpec:
    """Static specification of a restorer.

    Attributes:
        name: Short identifier used in logs and configs
            (e.g. ``"identity"``, ``"gfpgan"``, ``"codeformer"``).
        family: Currently always ``"restorers"``. Reserved for future
            distinctions (super-resolution vs face-restore vs generic
            image-restore).
        expects_face_crop: If True, the restorer assumes the input has
            been face-cropped and aligned to its native resolution. If
            False, it operates on full images.
        weight_url: Where the shipped weights live, if downloadable.
    """

    name: str
    family: str = "restorers"
    expects_face_crop: bool = False
    weight_url: str | None = None


@runtime_checkable
class Restorer(Protocol):
    """Differentiable image-to-image restoration transform.

    Implementations preserve the canonical layout: a ``(N, 3, H, W)``
    float tensor in ``[0.0, 1.0]`` goes in; the same shape and range
    come out. Preserving spatial dimensions is required — downstream
    targets assume it.
    """

    spec: RestorerSpec

    def __call__(self, image: Tensor) -> Tensor:
        """Apply the restorer to ``image`` and return the restored tensor."""
        ...
