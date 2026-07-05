# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Bench harness --out-dir path tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from voidface.eval.benchmark import BenchConfig, run_bench
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.models.base import TargetOutputs, TargetSpec


class _StubDetector(torch.nn.Module):
    spec = TargetSpec(name="stub-det", family="detectors")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        return TargetOutputs(logits=torch.tensor([[[0.0, 0.5]]]))


class _StubRecognizer(torch.nn.Module):
    spec = TargetSpec(name="stub-rec", family="recognizers")

    def forward(self, image: Tensor) -> TargetOutputs:  # noqa: PLR6301
        vec = image.mean(dim=(2, 3))
        norm = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return TargetOutputs(embedding=norm)


def test_out_dir_writes_protected_images(tmp_path: Path) -> None:
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    torch.manual_seed(0)
    images = [torch.rand(3, 32, 32) for _ in range(3)]
    out_dir = tmp_path / "protected"

    summary = run_bench(
        generator=generator,
        images=images,
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
        config=BenchConfig(device="cpu"),
        output_dir=out_dir,
        image_names=["a.png", "b.png", "c.png"],
    )

    assert summary.count == 3
    assert (out_dir / "a.protected.png").exists()
    assert (out_dir / "b.protected.png").exists()
    assert (out_dir / "c.protected.png").exists()


def test_out_dir_none_skips_saves(tmp_path: Path) -> None:
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    torch.manual_seed(0)
    images = [torch.rand(3, 32, 32) for _ in range(2)]
    out_dir = tmp_path / "should_not_be_created"

    run_bench(
        generator=generator,
        images=images,
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
        config=BenchConfig(device="cpu"),
    )
    assert not out_dir.exists()
