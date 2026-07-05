# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Voidface command-line entry point.
#
# Phase R2 expands the ensemble to (detector + identity + VAE) with
# LPIPS perceptual constraint and identity-restorer wrapping via the
# bilevel loss shape. Video, full 13-target ensemble, and the trained
# generator G land in later phases; this file grows subcommands as
# they do.

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
    """Entry point for ``python -m voidface_cli.main`` and the console script."""
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
        help="Add adversarial perturbation to a single image.",
    )
    p_protect.add_argument("image", type=Path, help="Input image path.")
    p_protect.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <image>.protected.png).",
    )
    p_protect.add_argument(
        "--epsilon", type=int, default=12, help="L-inf budget as N/255 (default 12)."
    )
    p_protect.add_argument("--steps", type=int, default=100, help="PGD steps (default 100).")
    p_protect.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps' (default auto).",
    )
    p_protect.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    p_protect.add_argument("--verbose", action="store_true", help="Log every PGD step.")
    p_protect.add_argument(
        "--targets",
        type=str,
        default="detector,recognizer",
        help=(
            "Comma-separated target subset. Options: 'detector', 'recognizer', 'vae'. "
            "Default: 'detector,recognizer' (Phase R1 subset). Use "
            "'detector,recognizer,vae' for the full Phase R2 ensemble."
        ),
    )
    p_protect.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip the LPIPS perceptual constraint (faster, weaker imperceptibility).",
    )

    p_report = sub.add_parser(
        "report",
        help="Print PSNR / SSIM / L-inf numbers for a protected image.",
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
        vae_gray_latent_loss,
    )
    from voidface.core.pgd import PgdConfig, run_pgd
    from voidface.eval.perceptual import load_lpips, psnr, ssim
    from voidface.models.detectors.mtcnn_pnet import MtcnnPnet
    from voidface.models.recognizers.facenet import Facenet
    from voidface.models.restorers.identity import IdentityRestorer
    from voidface.util.image import load_image, save_image
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli")

    device = _resolve_device(args.device)
    log.info("device.selected", device=str(device))

    log.info("image.loading", path=str(args.image))
    clean = load_image(args.image).to(device).unsqueeze(0)
    log.info("image.loaded", shape=tuple(clean.shape))

    selected = {t.strip() for t in args.targets.split(",") if t.strip()}
    allowed = {"detector", "recognizer", "vae"}
    if not selected.issubset(allowed):
        unknown = sorted(selected - allowed)
        log.error("targets.unknown", unknown=unknown, allowed=sorted(allowed))
        return 2
    log.info("targets.selected", targets=sorted(selected))

    target_losses = {}
    target_static_data = {}
    weights_targets: dict[str, float] = {}

    if "detector" in selected:
        log.info("model.detector.loading", name="mtcnn-pnet")
        detector = MtcnnPnet(device=device)
        target_losses["detector"] = (detector, retinaface_suppression_loss)
        weights_targets["detector"] = 0.35

    if "recognizer" in selected:
        log.info("model.recognizer.loading", name="facenet-vggface2")
        recognizer = Facenet(device=device)
        target_losses["recognizer"] = (recognizer, arcface_identity_loss)
        weights_targets["recognizer"] = 0.40

    if "vae" in selected:
        from voidface.models.vaes.sd15 import Sd15Vae

        log.info("model.vae.loading", name="sd15-vae")
        vae = Sd15Vae(device=device)
        gray_target = vae.encode_gray_target(height=clean.shape[-2], width=clean.shape[-1])
        target_losses["vae"] = (vae, vae_gray_latent_loss)
        target_static_data["vae"] = gray_target
        weights_targets["vae"] = 0.25

    if not target_losses:
        log.error("targets.empty")
        return 2

    _renormalize(weights_targets)

    lpips_fn = None
    lpips_weight = 0.0
    if not args.no_lpips:
        log.info("perceptual.lpips.loading", backbone="alex")
        lpips_fn = load_lpips(net="alex", device=device)
        lpips_weight = 0.10

    weights = LossWeights(
        targets=weights_targets,
        lpips=lpips_weight,
        total_variation=0.01,
    )
    composite = CompositeLoss(
        weights=weights,
        target_losses=target_losses,
        target_static_data=target_static_data,
        lpips=lpips_fn,
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

    restorer = IdentityRestorer()
    log.info(
        "pgd.start",
        epsilon=args.epsilon,
        steps=args.steps,
        targets=sorted(weights_targets),
        lpips=(lpips_weight > 0.0),
        restorer=restorer.spec.name,
    )

    from functools import partial

    original_compute = composite.compute
    composite.compute = partial(original_compute, restorer=restorer)  # type: ignore[method-assign]

    result = run_pgd(clean=clean, composite_loss=composite, eot=eot, config=pgd)
    log.info("pgd.done", final=round(result.history[-1].total_loss, 4))

    output = args.output or args.image.with_suffix(".protected.png")
    save_image(result.adversarial.squeeze(0), output)
    log.info("image.saved", path=str(output))

    _print_summary(clean=clean, adversarial=result.adversarial, output=output)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from voidface.eval.perceptual import psnr, ssim
    from voidface.util.image import load_image

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


def _renormalize(weights: dict[str, float]) -> None:
    """Rescale ``weights`` in place so they sum to 1.0.

    When the user selects a subset of targets we do not want the
    remaining weights to be, say, 0.4 total. Renormalizing keeps the
    per-family loss magnitude comparable across subsets.
    """
    total = sum(weights.values())
    if total <= 0:
        return
    for k in weights:
        weights[k] /= total


def _print_summary(clean: object, adversarial: object, output: Path) -> None:
    import torch

    from voidface.eval.perceptual import psnr, ssim

    if not isinstance(clean, torch.Tensor) or not isinstance(adversarial, torch.Tensor):
        return
    p = psnr(clean, adversarial)
    s = ssim(clean, adversarial)
    linf = (clean - adversarial).abs().max().item()
    print("--- summary ---")
    print(f"output:  {output}")
    print(f"PSNR:    {p:.2f} dB")
    print(f"SSIM:    {s:.4f}")
    print(f"L-inf:   {linf * 255:.2f}/255")


if __name__ == "__main__":
    raise SystemExit(main())
