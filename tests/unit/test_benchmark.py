# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Bench harness tests."""

from __future__ import annotations

import torch
from torch import Tensor

from voidface.eval.benchmark import BenchConfig, run_bench
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.models.base import TargetOutputs, TargetSpec


class _StubDetector(torch.nn.Module):
    """Face confidence proportional to mean pixel intensity.

    Serves as a stand-in for a real detector: images with
    darker perturbation produce lower confidence, so the generator
    can be exercised end-to-end.
    """

    spec = TargetSpec(name="stub-det", family="detectors")

    def forward(self, image: Tensor) -> TargetOutputs:
        mean = image.mean(dim=(1, 2, 3), keepdim=True).clamp(0, 1)
        # (N, 1, 2) softmax where face-present = mean, background = 1 - mean.
        # Use logits so downstream softmax gives back the same distribution.
        bg = torch.log((1.0 - mean).clamp(min=1e-6))
        fg = torch.log(mean.clamp(min=1e-6))
        logits = torch.cat([bg, fg], dim=-1).unsqueeze(0)  # (N, 1, 2) -> ensure 3D
        return TargetOutputs(logits=logits.view(image.size(0), 1, 2))


class _StubRecognizer(torch.nn.Module):
    """Emit a 6-D pooled embedding from mean + max pixel intensities."""

    spec = TargetSpec(name="stub-rec", family="recognizers")

    def forward(self, image: Tensor) -> TargetOutputs:
        mean = image.mean(dim=(2, 3))  # (N, 3)
        max_ = image.amax(dim=(2, 3))  # (N, 3)
        pooled = torch.cat([mean, max_], dim=-1)
        return TargetOutputs(embedding=pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def _images(n: int, size: int) -> list[Tensor]:
    torch.manual_seed(0)
    return [torch.rand(3, size, size) for _ in range(n)]


def test_bench_runs_and_populates_summary() -> None:
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    summary = run_bench(
        generator=generator,
        images=_images(4, 32),
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
    )
    assert summary.count == 4
    assert 0.0 <= summary.mean_ssim <= 1.0 + 1e-6
    assert summary.mean_psnr_db > 0


def test_bench_detection_asr_math() -> None:
    """Detection ASR is a share of eligible images broken by the generator."""
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    summary = run_bench(
        generator=generator,
        images=_images(5, 32),
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
    )
    asr = summary.detection_asr(threshold=0.5)
    assert 0.0 <= asr <= 1.0


def test_bench_identity_metric_bounds() -> None:
    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    summary = run_bench(
        generator=generator,
        images=_images(3, 32),
        detector=_StubDetector(),
        recognizer=_StubRecognizer(),
    )
    # cosine + 1 is in [0, 2]. Untrained G with a stub recognizer
    # should produce approximately full cosine (~2).
    assert 0.0 <= summary.mean_identity_cosine_plus_one <= 2.0 + 1e-6
