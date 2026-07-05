# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Public type interface for the models subsystem.
# See src/voidface/models/base.py for the runtime definitions.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

__all__ = ["EnsembleTarget", "TargetOutputs", "TargetSpec"]


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    family: str
    input_resolution: int | None = ...
    weight_url: str | None = ...


@dataclass(frozen=True, slots=True)
class TargetOutputs:
    logits: Tensor | None = ...
    embedding: Tensor | None = ...
    latent: Tensor | None = ...
    aux: Mapping[str, Tensor] | None = ...


@runtime_checkable
class EnsembleTarget(Protocol):
    spec: TargetSpec

    def __call__(self, image: Tensor) -> TargetOutputs: ...
