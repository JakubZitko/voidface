# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Weighted sampler over restorer instances.
#
# The bilevel training loop samples one restorer per step so that the
# generator learns to defeat a distribution of attackers rather than a
# single one. The distribution is a plain probability vector over the
# restorer options; the identity restorer must almost always be one
# of them so the generator also works against un-restored pipelines.
#
# See Documentation/training/bilevel-adversarial.md.

"""Weighted restorer sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voidface.models.restorers.base import Restorer

__all__ = ["RestorerSampler", "SamplerConfig"]


@dataclass(frozen=True)
class SamplerConfig:
    """Static configuration for :class:`RestorerSampler`.

    Attributes:
        seed: If not ``None``, initialize the internal RNG for
            deterministic sampling.
    """

    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", random.Random(self.seed))


class RestorerSampler:
    """Sample one restorer per training step from a weighted distribution.

    Example::

        sampler = RestorerSampler(
            [(IdentityRestorer(), 0.30),
             (Sd15VaeRestorer(vae), 0.70)],
            config=SamplerConfig(seed=0),
        )
        for _ in range(N):
            restorer = sampler.sample()
            ...

    Weights are normalized on construction; passing ``[(a, 3.0),
    (b, 1.0)]`` is equivalent to ``[(a, 0.75), (b, 0.25)]``. Zero-weight
    entries are permitted and skipped.
    """

    def __init__(
        self,
        options: list[tuple[Restorer, float]],
        config: SamplerConfig | None = None,
    ) -> None:
        if not options:
            msg = "RestorerSampler requires at least one option."
            raise ValueError(msg)
        active = [(restorer, weight) for restorer, weight in options if weight > 0]
        if not active:
            msg = "RestorerSampler requires at least one non-zero weight."
            raise ValueError(msg)

        total = sum(weight for _, weight in active)
        self._restorers = [restorer for restorer, _ in active]
        self._weights = [weight / total for _, weight in active]
        self._config = config or SamplerConfig()

    def sample(self) -> Restorer:
        """Draw one restorer from the configured distribution."""
        return self._config._rng.choices(self._restorers, weights=self._weights, k=1)[0]

    def names(self) -> list[str]:
        """Return the names of the sampled restorers, in weight order."""
        return [r.spec.name for r in self._restorers]

    def probabilities(self) -> dict[str, float]:
        """Return the (name, probability) map used for logging."""
        return dict(zip((r.spec.name for r in self._restorers), self._weights, strict=True))
