# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface report` — image-quality metrics for a (clean, protected) pair."""

from __future__ import annotations

import argparse
import sys

from voidface.eval.perceptual import psnr, ssim
from voidface.util.image import load_image


def run(args: argparse.Namespace) -> int:
    """Print PSNR / SSIM / L-inf between the original and protected images."""
    clean = load_image(args.original).unsqueeze(0)
    protected = load_image(args.protected).unsqueeze(0)
    if clean.shape != protected.shape:
        print(
            f"error: shape mismatch clean={tuple(clean.shape)} "
            f"protected={tuple(protected.shape)}",
            file=sys.stderr,
        )
        return 2

    print(f"PSNR:   {psnr(clean, protected):.2f} dB")
    print(f"SSIM:   {ssim(clean, protected):.4f}")
    diff = (clean - protected).abs()
    print(f"L-inf:  {diff.max().item() * 255:.2f}/255")
    return 0
