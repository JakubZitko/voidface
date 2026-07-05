# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Shared CLI helper tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_resolve_device_cpu() -> None:
    from voidface_cli.common import resolve_device

    device = resolve_device("cpu")
    assert str(device) == "cpu"


def test_resolve_device_auto() -> None:
    from voidface_cli.common import resolve_device

    device = resolve_device("auto")
    # Result depends on the host; just check we got a torch.device.
    assert hasattr(device, "type")


def test_renormalize_weights_scales_to_sum_one() -> None:
    from voidface_cli.common import renormalize_weights

    weights = {"a": 2.0, "b": 3.0}
    renormalize_weights(weights)
    assert weights == pytest.approx({"a": 0.4, "b": 0.6})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_renormalize_weights_empty_is_noop() -> None:
    from voidface_cli.common import renormalize_weights

    weights: dict[str, float] = {}
    renormalize_weights(weights)
    assert weights == {}


def test_renormalize_weights_all_zero_is_noop() -> None:
    from voidface_cli.common import renormalize_weights

    weights = {"a": 0.0, "b": 0.0}
    renormalize_weights(weights)
    # No division-by-zero; original values preserved.
    assert weights == {"a": 0.0, "b": 0.0}


def test_load_generator_checkpoint_missing_raises(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from voidface_cli.common import load_generator_checkpoint

    with pytest.raises(FileNotFoundError, match="not found"):
        load_generator_checkpoint(
            tmp_path / "does_not_exist.pt",
            torch.device("cpu"),
            MagicMock(),
        )


def test_load_generator_checkpoint_roundtrip(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from voidface_cli.common import load_generator_checkpoint

    from voidface.generator.architecture import Voidface, VoidfaceConfig

    ckpt = tmp_path / "gen.pt"
    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": 42, "state_dict": generator.state_dict(), "config": config}, ckpt)

    loaded, loaded_config = load_generator_checkpoint(
        ckpt, torch.device("cpu"), MagicMock()
    )
    assert loaded.config.base_channels == 8
    assert loaded_config.base_channels == 8


def test_load_generator_checkpoint_corrupt_raises(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from voidface_cli.common import load_generator_checkpoint

    corrupt = tmp_path / "bad.pt"
    corrupt.write_bytes(b"not a torch pickle")

    with pytest.raises(RuntimeError, match="could not load"):
        load_generator_checkpoint(corrupt, torch.device("cpu"), MagicMock())
