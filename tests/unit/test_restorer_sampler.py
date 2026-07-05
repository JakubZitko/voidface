# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""RestorerSampler weighted-choice properties."""

from __future__ import annotations

from collections import Counter

import pytest

from voidface.models.restorers.base import RestorerSpec
from voidface.models.restorers.sampler import RestorerSampler, SamplerConfig


class _NamedRestorer:
    def __init__(self, name: str) -> None:
        self.spec = RestorerSpec(name=name)

    def __call__(self, image):
        return image


def test_single_option_always_samples() -> None:
    r = _NamedRestorer("a")
    sampler = RestorerSampler([(r, 1.0)], SamplerConfig(seed=0))
    for _ in range(20):
        assert sampler.sample() is r


def test_weight_ratio_is_respected() -> None:
    a = _NamedRestorer("a")
    b = _NamedRestorer("b")
    sampler = RestorerSampler([(a, 1.0), (b, 3.0)], SamplerConfig(seed=0))
    counter = Counter(sampler.sample().spec.name for _ in range(2_000))
    # b should be sampled roughly 3x more often than a
    ratio = counter["b"] / counter["a"]
    assert 2.3 < ratio < 4.3, counter


def test_zero_weight_option_is_dropped() -> None:
    a = _NamedRestorer("a")
    b = _NamedRestorer("b")
    sampler = RestorerSampler([(a, 1.0), (b, 0.0)], SamplerConfig(seed=0))
    for _ in range(20):
        assert sampler.sample() is a


def test_all_zero_weights_raises() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        RestorerSampler([(_NamedRestorer("a"), 0.0)])


def test_empty_options_raises() -> None:
    with pytest.raises(ValueError, match="at least one option"):
        RestorerSampler([])


def test_probabilities_map_normalizes() -> None:
    sampler = RestorerSampler(
        [(_NamedRestorer("a"), 1.0), (_NamedRestorer("b"), 3.0)],
        SamplerConfig(seed=0),
    )
    probs = sampler.probabilities()
    assert probs["a"] == pytest.approx(0.25)
    assert probs["b"] == pytest.approx(0.75)
    assert sum(probs.values()) == pytest.approx(1.0)
