# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Voidface command-line entry point.
#
# Phase R1 wires the `protect` subcommand end-to-end against the
# minimum viable ensemble (RetinaFace + ArcFace). It does not yet
# support video, the trained generator G, or batching. Those land in
# later phases; this file grows subcommands as they do.

"""Voidface CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import voidface

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["app", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m voidface_cli.main`` and the console script.

    Args:
        argv: Argument vector without the program name. Defaults to
            :data:`sys.argv[1:]`.

    Returns:
        The exit code to hand back to the shell.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "protect":
        return _cmd_protect(args)
    if args.command == "report":
        return _cmd_report(args)

    parser.print_help(sys.stderr)
    return 2


def app() -> None:
    """Console-script wrapper. Exits the process with :func:`main`'s code."""
    raise SystemExit(main())


# --- argument parsing --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voidface",
        description="Voidface — adversarial face-region blindfold.",
    )
    parser.add_argument("--version", action="version", version=f"voidface {voidface.__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_protect = sub.add_parser(
        "protect",
        help="Add adversarial perturbation to a single image (Phase R1).",
    )
    p_protect.add_argument("image", type=Path, help="Input image path.")
    p_protect.add_argument(
        "-o", "--output", type=Path, default=None, help="Output image path (default: <image>.protected.png)."
    )
    p_protect.add_argument("--epsilon", type=int, default=12, help="L-inf budget as N/255 (default 12).")
    p_protect.add_argument("--steps", type=int, default=100, help="PGD steps (default 100).")
    p_protect.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps' (default auto).",
    )
    p_protect.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    p_protect.add_argument("--verbose", action="store_true", help="Log every PGD step.")

    p_report = sub.add_parser(
        "report",
        help="Print PSNR / SSIM / cosine numbers for a protected image.",
    )
    p_report.add_argument("original", type=Path, help="Original clean image.")
    p_report.add_argument("protected", type=Path, help="Protected image (Voidface output).")

    return parser


# --- commands ----------------------------------------------------------------


def _cmd_protect(args: argparse.Namespace) -> int:
    # Deferred so `--help` and `--version` do not incur PyTorch import cost.
    import torch

    from voidface.core.eot import EotConfig, EotSampler
    from voidface.core.loss import (
        CompositeLoss,
        LossWeights,
        arcface_identity_loss,
        retinaface_suppression_loss,
    )
    from voidface.core.pgd import PgdConfig, run_pgd
    from voidface.models.detectors.mtcnn_pnet import MtcnnPnet
    from voidface.models.recognizers.facenet import Facenet
    from voidface.util.image import load_image, save_image
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli")

    device = _resolve_device(args.device)
    log.info("device.selected", device=str(device))

    log.info("image.loading", path=str(args.image))
    clean = load_image(args.image).to(device).unsqueeze(0)
    log.info("image.loaded", shape=tuple(clean.shape))

    log.info("model.detector.loading")
    detector = MtcnnPnet(device=device)
    log.info("model.recognizer.loading")
    recognizer = Facenet(device=device)

    def identity_pair_loss(perturbed, clean_out):  # type: ignore[no-untyped-def]
        return arcface_identity_loss(perturbed, clean_out)

    weights = LossWeights(
        targets={"detector": 0.5, "recognizer": 0.5},
        lpips=0.0,             # Phase R1: no LPIPS to avoid extra weight download; add in R2.
        total_variation=0.01,
    )
    composite = CompositeLoss(
        weights=weights,
        target_losses={
            "detector": (detector, retinaface_suppression_loss),
            "recognizer": (recognizer, identity_pair_loss),
        },
    )
    eot = EotSampler(EotConfig(samples=2, seed=args.seed))
    pgd = PgdConfig(
        epsilon=args.epsilon / 255.0,
        alpha=max(1, args.epsilon // 6) / 255.0,
        steps=args.steps,
        momentum=0.9,
        log_every=1 if args.verbose else max(1, args.steps // 5),
        seed=args.seed,
    )

    log.info("pgd.start", epsilon=args.epsilon, steps=args.steps)
    result = run_pgd(clean=clean, composite_loss=composite, eot=eot, config=pgd)
    log.info("pgd.done", final=round(result.history[-1].total_loss, 4))

    output = args.output or args.image.with_suffix(".protected.png")
    save_image(result.adversarial.squeeze(0), output)
    log.info("image.saved", path=str(output))

    _print_summary(clean=clean, adversarial=result.adversarial, output=output)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    import torch

    from voidface.util.image import load_image

    clean = load_image(args.original)
    protected = load_image(args.protected)
    if clean.shape != protected.shape:
        print(f"error: shape mismatch clean={tuple(clean.shape)} protected={tuple(protected.shape)}",
              file=sys.stderr)
        return 2

    mse = torch.mean((clean - protected) ** 2).item()
    psnr = float("inf") if mse == 0.0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    linf = (clean - protected).abs().max().item()
    print(f"PSNR: {psnr:.2f} dB")
    print(f"L-inf: {linf:.4f}  ({linf * 255:.2f}/255)")
    return 0


# --- helpers -----------------------------------------------------------------


def _resolve_device(name: str) -> object:
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _print_summary(clean: object, adversarial: object, output: Path) -> None:
    import torch

    if not isinstance(clean, torch.Tensor) or not isinstance(adversarial, torch.Tensor):
        return
    diff = (clean - adversarial).abs()
    mse = torch.mean(diff**2).item()
    psnr = float("inf") if mse == 0.0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    linf = diff.max().item()
    print("--- summary ---")
    print(f"output:  {output}")
    print(f"PSNR:    {psnr:.2f} dB")
    print(f"L-inf:   {linf * 255:.2f}/255")


if __name__ == "__main__":
    raise SystemExit(main())
