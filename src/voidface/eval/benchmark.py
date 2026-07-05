# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Bench harness for a trained generator.
#
# Iterates every image in a directory, runs the generator, forwards
# clean + protected through the ensemble targets, and reports:
#
#   * Detection ASR — the share of images where the detector's face
#     confidence drops below the threshold on the protected image
#     while it was above threshold on the clean image.
#   * Identity cosine displacement — mean cosine between the ArcFace
#     embedding on clean vs protected. Lower is better; -1 is the
#     ideal ("different identity"). Reported as ``1 + cos`` so 0
#     means the attack succeeded and 2 means it failed.
#   * PSNR and SSIM — perceptual invisibility.
#   * Wall-clock per image.
#
# This is what R5.5 uses to decide whether a checkpoint is
# shipping-quality.

"""Bench harness for a trained generator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from voidface.eval.perceptual import psnr, ssim
from voidface.util.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from voidface.generator.architecture import Voidface
    from voidface.models.base import EnsembleTarget

__all__ = ["BenchConfig", "BenchImageResult", "BenchSummary", "run_bench"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class BenchConfig:
    device: str = "cpu"
    detection_threshold: float = 0.5


@dataclass
class BenchImageResult:
    path: str
    detection_before: float
    detection_after: float
    identity_cosine_plus_one: float
    psnr_db: float
    ssim: float
    wall_ms: float


@dataclass
class BenchSummary:
    per_image: list[BenchImageResult] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.per_image)

    @property
    def mean_psnr_db(self) -> float:
        return _mean(x.psnr_db for x in self.per_image)

    @property
    def mean_ssim(self) -> float:
        return _mean(x.ssim for x in self.per_image)

    @property
    def mean_identity_cosine_plus_one(self) -> float:
        return _mean(x.identity_cosine_plus_one for x in self.per_image)

    def detection_asr(self, threshold: float) -> float:
        """Detection attack-success rate.

        Success = detector was above ``threshold`` on the clean image
        AND fell below ``threshold`` on the protected image. Reported
        as a fraction in ``[0, 1]``.
        """
        eligible = [x for x in self.per_image if x.detection_before >= threshold]
        if not eligible:
            return 0.0
        broken = sum(1 for x in eligible if x.detection_after < threshold)
        return broken / len(eligible)


def run_bench(
    generator: Voidface,
    images: Iterable[Tensor],
    detector: EnsembleTarget,
    recognizer: EnsembleTarget,
    config: BenchConfig = BenchConfig(),
) -> BenchSummary:
    """Run the bench harness against ``images``.

    Args:
        generator: A trained :class:`Voidface`.
        images: An iterable of ``(3, H, W)`` tensors in ``[0, 1]``.
            Each is processed independently.
        detector: A face-detector target. Used to score face-present
            confidence before/after.
        recognizer: An identity encoder target. Used to measure
            embedding cosine displacement.
        config: :class:`BenchConfig` for device / threshold.

    Returns:
        A :class:`BenchSummary` with per-image and aggregate stats.
    """
    device = torch.device(config.device)
    generator = generator.to(device).eval()
    summary = BenchSummary()

    with torch.no_grad():
        for index, image in enumerate(images):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)

            start = time.perf_counter()
            protected = generator(image)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            det_before = _detection_face_score(detector(image))
            det_after = _detection_face_score(detector(protected))

            clean_id = recognizer(image).embedding
            adv_id = recognizer(protected).embedding
            assert clean_id is not None and adv_id is not None
            cos_plus_one = (F.cosine_similarity(clean_id, adv_id, dim=-1) + 1.0).mean().item()

            summary.per_image.append(
                BenchImageResult(
                    path=f"image_{index:06d}",
                    detection_before=det_before,
                    detection_after=det_after,
                    identity_cosine_plus_one=cos_plus_one,
                    psnr_db=psnr(image, protected),
                    ssim=ssim(image, protected),
                    wall_ms=elapsed_ms,
                )
            )

    _log.info(
        "bench.done",
        count=summary.count,
        detection_asr=round(summary.detection_asr(config.detection_threshold), 4),
        identity_cos_plus_one=round(summary.mean_identity_cosine_plus_one, 4),
        psnr_db=round(summary.mean_psnr_db, 2),
        ssim=round(summary.mean_ssim, 4),
    )
    return summary


def _detection_face_score(outputs) -> float:  # noqa: ANN001
    """Extract a single face-present scalar from a detector's outputs.

    Uses the max face-present softmax probability over all anchors,
    matching the semantics of the R4.5.2b GfpganRestorer landmark
    picker.
    """
    if outputs.logits is None:
        return 0.0
    face = torch.softmax(outputs.logits, dim=-1)[..., 1]
    return float(face.max().item())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
