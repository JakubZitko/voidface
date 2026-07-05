# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Common interface for every adversarial ensemble target.
#
# See Documentation/architecture.md for the role this file plays and
# Documentation/models/*.md for per-subsystem detail.

"""Interfaces shared by every adversarial ensemble target.

Every wrapped attacker model — face detectors, identity recognizers,
diffusion VAEs, CLIP-family encoders, face restorers — implements the
:class:`EnsembleTarget` protocol. The training loop in
:mod:`voidface.core.train` consumes only this protocol; it never depends
on the concrete model classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

__all__ = [
    "EnsembleTarget",
    "TargetOutputs",
    "TargetSpec",
]


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Static specification of an ensemble target.

    Attributes:
        name: A short identifier used in logs and configs
            (e.g. ``"retinaface"``, ``"arcface"``, ``"gfpgan"``).
        family: The subsystem the target belongs to
            (``"detectors"``, ``"recognizers"``, ``"vaes"``,
            ``"clip"``, ``"restorers"``).
        input_resolution: The spatial size in pixels the wrapped model
            expects. ``None`` means the target accepts variable input.
        weight_url: Where the shipped weights live, if downloadable.
    """

    name: str
    family: str
    input_resolution: int | None = None
    weight_url: str | None = None


@dataclass(frozen=True, slots=True)
class TargetOutputs:
    """Structured outputs from a single ensemble target forward pass.

    Not every field is populated by every target. See the target's
    subsystem docstring for which fields it fills. Consumers must treat
    unpopulated fields as ``None`` and skip loss terms that depend on
    them.
    """

    logits: Tensor | None = None
    embedding: Tensor | None = None
    latent: Tensor | None = None
    aux: Mapping[str, Tensor] | None = None


@runtime_checkable
class EnsembleTarget(Protocol):
    """The single interface every ensemble target implements.

    An implementation is either a :class:`torch.nn.Module` or a plain
    class that exposes a callable ``__call__``.
    """

    spec: TargetSpec

    def __call__(self, image: Tensor) -> TargetOutputs:
        """Forward the perturbed image through the target model.

        Args:
            image: A float tensor with shape ``(N, 3, H, W)`` and values
                in ``[0.0, 1.0]``. The target is responsible for any
                normalization or resize the wrapped model requires.

        Returns:
            A :class:`TargetOutputs` populated with whichever tensors the
            target produces. See the target's subsystem docstring for the
            contract.
        """
        ...
