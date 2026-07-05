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
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "bench":
        return _cmd_bench(args)

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
    p_protect.add_argument(
        "image",
        type=Path,
        help=(
            "Input image path OR a directory (batch mode). In batch mode "
            "every supported image is processed and written to --output-dir "
            "with the same filename plus '.protected.png'."
        ),
    )
    p_protect.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (single-image mode; default: <image>.protected.png).",
    )
    p_protect.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Batch mode output directory. Required when the input is a "
            "directory; ignored otherwise."
        ),
    )
    p_protect.add_argument(
        "--recursive",
        action="store_true",
        help="Batch mode: recurse into subdirectories.",
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
            "Comma-separated target subset. Options: 'detector', 'recognizer', "
            "'vae' (SD 1.5), 'sdxl-vae'. Default: 'detector,recognizer' (Phase R1 "
            "subset). Use 'detector,recognizer,vae,sdxl-vae' for the R4 VAE-family "
            "ensemble."
        ),
    )
    p_protect.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip the LPIPS perceptual constraint (faster, weaker imperceptibility).",
    )
    p_protect.add_argument(
        "--restorers",
        type=str,
        default="identity",
        help=(
            "Comma-separated restorer distribution for the bilevel loop. "
            "Format: 'name[:weight],...'. Options: 'identity', 'sd15-vae', "
            "'gfpgan'. Default: 'identity'. Example: "
            "'identity:0.1,sd15-vae:0.3,gfpgan:0.6' — the R4 CEO-critic "
            "recommended mix — samples GFPGAN 60%% of steps. 'sd15-vae' "
            "implicitly loads the VAE target too; 'gfpgan' implicitly "
            "loads the RetinaFace detector for landmarks."
        ),
    )
    p_protect.add_argument(
        "--use-generator",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Skip the per-image PGD loop entirely; run one forward pass through "
            "a trained generator G loaded from CHECKPOINT. This is the deploy "
            "path — turns ~2 min of PGD into <1 s of generator forward. "
            "Ignores --targets, --restorers, --steps, and other PGD-related "
            "flags; --epsilon is still applied as the L-inf budget."
        ),
    )

    p_report = sub.add_parser(
        "report",
        help="Print PSNR / SSIM / L-inf numbers for a protected image.",
    )
    p_report.add_argument("original", type=Path, help="Original clean image.")
    p_report.add_argument("protected", type=Path, help="Protected image (Voidface output).")

    p_train = sub.add_parser(
        "train",
        help="Train the generator G against a folder of face images.",
    )
    p_train.add_argument("config", type=Path, help="Path to a TOML training config.")
    p_train.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps' (default auto).",
    )
    p_train.add_argument("--verbose", action="store_true", help="Log every step.")

    p_export = sub.add_parser(
        "export",
        help="Export the generator G to ONNX (optionally int8-quantized).",
    )
    p_export.add_argument("checkpoint", type=Path, help="Path to a trained .pt checkpoint.")
    p_export.add_argument("output", type=Path, help="Output .onnx path.")
    p_export.add_argument(
        "--example-resolution",
        type=int,
        default=256,
        help="Side length in pixels of the tracing example (default 256).",
    )
    p_export.add_argument(
        "--quantize",
        choices=["int8", "uint8"],
        default=None,
        help=(
            "Also emit a dynamically-quantized <output>.<type>.onnx alongside "
            "the fp32 file. Fast, but ORT has known op-support gaps for "
            "Conv-heavy graphs — the parity is not guaranteed. Prefer "
            "--quantize-static-dir for shipping quality."
        ),
    )
    p_export.add_argument(
        "--quantize-static-dir",
        type=Path,
        default=None,
        metavar="CALIBRATION_DIR",
        help=(
            "Also emit a statically-quantized <output>.static-int8.onnx "
            "using CALIBRATION_DIR as the calibration corpus. Iterates the "
            "folder via FolderImageDataset (up to --quantize-static-samples "
            "images). Static quant preserves runtime parity within a tight "
            "tolerance, unlike --quantize (dynamic)."
        ),
    )
    p_export.add_argument(
        "--quantize-static-samples",
        type=int,
        default=64,
        help="Number of calibration images (default 64).",
    )
    p_export.add_argument(
        "--coreml",
        action="store_true",
        help=(
            "Also emit a CoreML .mlpackage next to <output>. Only works on Apple "
            "Silicon macOS with coremltools installed."
        ),
    )
    p_export.add_argument(
        "--ort",
        action="store_true",
        help=(
            "Also emit a .ort file (ONNX Runtime Web format) in the output "
            "directory. Used by the browser demo for faster startup."
        ),
    )

    p_bench = sub.add_parser(
        "bench",
        help="Benchmark a trained G against a folder of face images.",
    )
    p_bench.add_argument("checkpoint", type=Path, help="Path to a trained .pt checkpoint.")
    p_bench.add_argument("images", type=Path, help="Directory of test images.")
    p_bench.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps' (default auto).",
    )
    p_bench.add_argument(
        "--detection-threshold",
        type=float,
        default=0.5,
        help="Face-present threshold for detection ASR (default 0.5).",
    )
    p_bench.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Side length in pixels each test image is resized to (default 256).",
    )
    p_bench.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write a machine-readable JSON summary to PATH alongside the "
            "human-readable print. Fields: count, detection_asr, "
            "mean_identity_cosine_plus_one, mean_psnr_db, mean_ssim, per_image."
        ),
    )

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
    from voidface.models.restorers.identity import IdentityRestorer
    from voidface.models.restorers.sampler import RestorerSampler, SamplerConfig
    from voidface.util.image import load_image, save_image
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli")

    device = _resolve_device(args.device)
    log.info("device.selected", device=str(device))

    # Batch mode dispatch: input is a directory.
    if args.image.is_dir():
        return _protect_batch(args, device, log)

    log.info("image.loading", path=str(args.image))
    clean = load_image(args.image).to(device).unsqueeze(0)
    log.info("image.loaded", shape=tuple(clean.shape))

    # Deploy fast-path — skip PGD entirely.
    if args.use_generator is not None:
        return _protect_via_generator(args, clean, log)

    selected = {t.strip() for t in args.targets.split(",") if t.strip()}
    allowed = {"detector", "recognizer", "vae", "sdxl-vae", "openclip"}
    if not selected.issubset(allowed):
        unknown = sorted(selected - allowed)
        log.error("targets.unknown", unknown=unknown, allowed=sorted(allowed))
        return 2

    restorer_spec = _parse_restorer_spec(args.restorers)
    if restorer_spec is None:
        log.error("restorers.unknown", spec=args.restorers)
        return 2
    if "sd15-vae" in {name for name, _ in restorer_spec} and "vae" not in selected:
        # The SD 1.5 VAE restorer needs the encoder loaded anyway.
        # Auto-add the target so the user does not have to.
        selected.add("vae")
        log.info("targets.autoadd", added="vae", reason="sd15-vae restorer selected")

    log.info("targets.selected", targets=sorted(selected))
    log.info("restorers.selected", spec=restorer_spec)

    target_losses = {}
    target_static_data = {}
    weights_targets: dict[str, float] = {}

    if "detector" in selected:
        from voidface.models.detectors.retinaface import RetinaFace

        log.info("model.detector.loading", name="retinaface-r50")
        detector = RetinaFace(device=device)
        target_losses["detector"] = (detector, retinaface_suppression_loss)
        weights_targets["detector"] = 0.35

    if "recognizer" in selected:
        from voidface.models.recognizers.arcface import Arcface

        log.info("model.recognizer.loading", name="arcface-r100")
        recognizer = Arcface(device=device)
        target_losses["recognizer"] = (recognizer, arcface_identity_loss)
        weights_targets["recognizer"] = 0.40

    vae = None
    if "vae" in selected:
        from voidface.models.vaes.sd15 import Sd15Vae

        log.info("model.vae.loading", name="sd15-vae")
        vae = Sd15Vae(device=device)
        gray_target = vae.encode_gray_target(height=clean.shape[-2], width=clean.shape[-1])
        target_losses["vae"] = (vae, vae_gray_latent_loss)
        target_static_data["vae"] = gray_target
        weights_targets["vae"] = 0.20

    if "sdxl-vae" in selected:
        from voidface.models.vaes.sdxl import SdxlVae

        log.info("model.sdxl-vae.loading", name="sdxl-vae")
        sdxl_vae = SdxlVae(device=device)
        sdxl_gray_target = sdxl_vae.encode_gray_target(
            height=clean.shape[-2], width=clean.shape[-1]
        )
        target_losses["sdxl-vae"] = (sdxl_vae, vae_gray_latent_loss)
        target_static_data["sdxl-vae"] = sdxl_gray_target
        weights_targets["sdxl-vae"] = 0.15

    if "openclip" in selected:
        from voidface.models.clip.openclip import OpenClip

        log.info("model.openclip.loading", name="openclip-vit-b-32")
        openclip = OpenClip(device=device)
        target_losses["openclip"] = (openclip, arcface_identity_loss)
        weights_targets["openclip"] = 0.10

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

    restorer_options = []
    for name, weight in restorer_spec:
        if name == "identity":
            restorer_options.append((IdentityRestorer(), weight))
        elif name == "sd15-vae":
            from voidface.models.restorers.sd_vae import Sd15VaeRestorer

            assert vae is not None, "sd15-vae restorer requires the VAE target."
            restorer_options.append((Sd15VaeRestorer(encoder=vae), weight))
        elif name == "gfpgan":
            from voidface.models.restorers.gfpgan import GfpganRestorer

            # gfpgan restorer needs a detector for landmarks — reuse
            # the one already loaded if present, otherwise load one.
            gfpgan_detector = target_losses.get("detector", (None,))[0]
            if gfpgan_detector is None:
                from voidface.models.detectors.retinaface import RetinaFace

                log.info("model.detector.loading", name="retinaface-r50", reason="gfpgan needs landmarks")
                gfpgan_detector = RetinaFace(device=device)
            log.info("model.gfpgan.loading", name="gfpgan-v1.4")
            restorer_options.append((GfpganRestorer(detector=gfpgan_detector, device=device), weight))

    restorer_sampler = RestorerSampler(restorer_options, SamplerConfig(seed=args.seed))
    log.info(
        "pgd.start",
        epsilon=args.epsilon,
        steps=args.steps,
        targets=sorted(weights_targets),
        lpips=(lpips_weight > 0.0),
        restorers=restorer_sampler.probabilities(),
    )

    result = run_pgd(
        clean=clean,
        composite_loss=composite,
        eot=eot,
        config=pgd,
        restorer_sampler=restorer_sampler,
    )
    log.info("pgd.done", final=round(result.history[-1].total_loss, 4))

    output = args.output or args.image.with_suffix(".protected.png")
    save_image(result.adversarial.squeeze(0), output)
    log.info("image.saved", path=str(output))

    _print_summary(clean=clean, adversarial=result.adversarial, output=output)
    return 0


def _protect_batch(args: argparse.Namespace, device, log) -> int:  # noqa: ANN001
    """Process every image in a directory, writing outputs to --output-dir.

    Batch mode currently requires --use-generator — running per-image
    PGD on a large folder is impractical (~2 min per image). The
    generator is loaded exactly once regardless of batch size.
    """
    from voidface.data.datasets import collect_image_paths
    from voidface.util.image import load_image

    if args.output_dir is None:
        log.error(
            "batch.missing_output_dir",
            hint="pass --output-dir DIR to write protected outputs",
        )
        return 2
    if args.use_generator is None:
        log.error(
            "batch.requires_generator",
            hint="pass --use-generator CHECKPOINT to enable batch mode",
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_image_paths(args.image, recursive=args.recursive)
    log.info("batch.starting", count=len(paths), output_dir=str(args.output_dir))

    # Hoist the checkpoint load out of the per-item hot path.
    generator, config = _load_generator_checkpoint(args.use_generator, device, log)

    for index, path in enumerate(paths):
        log.info("batch.item", index=index + 1, total=len(paths), path=str(path))
        clean = load_image(path).to(device).unsqueeze(0)
        output_path = args.output_dir / (path.stem + ".protected.png")
        _run_generator_and_save(
            generator=generator,
            config=config,
            clean=clean,
            output_path=output_path,
            epsilon_int=args.epsilon,
        )
        log.info("batch.item.done", path=str(output_path))
    log.info("batch.done", count=len(paths))
    return 0


def _load_generator_checkpoint(path: Path, device, log):  # noqa: ANN001,ANN202
    import torch

    from voidface.generator.architecture import Voidface, VoidfaceConfig

    log.info("generator.loading", path=str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored = payload.get("config")
        config = stored if isinstance(stored, VoidfaceConfig) else VoidfaceConfig()
    else:
        state_dict = payload
        config = VoidfaceConfig()
    generator = Voidface(config).to(device).eval()
    generator.load_state_dict(state_dict)
    log.info(
        "generator.loaded", params=sum(p.numel() for p in generator.parameters())
    )
    return generator, config


def _run_generator_and_save(  # noqa: ANN001,ANN201
    generator,
    config,
    clean,
    output_path: Path,
    epsilon_int: int,
):
    import torch
    import torch.nn.functional as F

    from voidface.util.image import save_image

    divisor = 1 << config.num_stages
    original_hw = clean.shape[-2:]
    padded_h = (original_hw[0] + divisor - 1) // divisor * divisor
    padded_w = (original_hw[1] + divisor - 1) // divisor * divisor
    if (padded_h, padded_w) != original_hw:
        clean_padded = F.pad(
            clean,
            (0, padded_w - original_hw[1], 0, padded_h - original_hw[0]),
            mode="reflect",
        )
    else:
        clean_padded = clean
    with torch.no_grad():
        adversarial = generator(clean_padded, epsilon=epsilon_int / 255.0)
    adversarial = adversarial[..., : original_hw[0], : original_hw[1]]
    save_image(adversarial.squeeze(0), output_path)
    return adversarial


def _protect_via_generator(args: argparse.Namespace, clean, log) -> int:  # noqa: ANN001
    """`voidface protect --use-generator` fast path: one G forward.

    Loads the checkpoint written by :func:`voidface.core.train.train_generator`,
    reconstructs the :class:`Voidface` with its saved config, and produces
    the protected image in a single forward pass.
    """
    generator, config = _load_generator_checkpoint(args.use_generator, clean.device, log)
    output = args.output or args.image.with_suffix(".protected.png")
    adversarial = _run_generator_and_save(
        generator=generator,
        config=config,
        clean=clean,
        output_path=output,
        epsilon_int=args.epsilon,
    )
    log.info("image.saved", path=str(output))
    _print_summary(clean=clean, adversarial=adversarial, output=output)
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


_ALLOWED_RESTORERS = {"identity", "sd15-vae", "gfpgan"}


def _cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark a trained generator against a folder of test images."""
    import torch

    from voidface.data.datasets import FolderImageDataset
    from voidface.eval.benchmark import BenchConfig, run_bench
    from voidface.generator.architecture import Voidface, VoidfaceConfig
    from voidface.models.detectors.retinaface import RetinaFace
    from voidface.models.recognizers.arcface import Arcface
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="INFO")
    log = get_logger("voidface.cli.bench")
    device = _resolve_device(args.device)

    log.info("checkpoint.loading", path=str(args.checkpoint))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored_config = payload.get("config")
        config = stored_config if isinstance(stored_config, VoidfaceConfig) else VoidfaceConfig()
    else:
        state_dict = payload
        config = VoidfaceConfig()
    generator = Voidface(config).eval()
    generator.load_state_dict(state_dict)

    log.info("dataset.loading", directory=str(args.images), resolution=args.resolution)
    dataset = FolderImageDataset(args.images, resolution=args.resolution, augment=False)
    log.info("dataset.loaded", size=len(dataset))

    log.info("model.detector.loading", name="retinaface-r50")
    detector = RetinaFace(device=device)
    log.info("model.recognizer.loading", name="arcface-r100")
    recognizer = Arcface(device=device)

    summary = run_bench(
        generator=generator,
        images=(dataset[i] for i in range(len(dataset))),
        detector=detector,
        recognizer=recognizer,
        config=BenchConfig(
            device=str(device),
            detection_threshold=args.detection_threshold,
        ),
    )

    print("--- bench summary ---")
    print(f"images:          {summary.count}")
    print(f"detection ASR:   {summary.detection_asr(args.detection_threshold):.4f}")
    print(f"identity cos+1:  {summary.mean_identity_cosine_plus_one:.4f}  (0=success, 2=failure)")
    print(f"PSNR (mean):     {summary.mean_psnr_db:.2f} dB")
    print(f"SSIM (mean):     {summary.mean_ssim:.4f}")

    if args.json is not None:
        import json

        payload = {
            "count": summary.count,
            "detection_asr": summary.detection_asr(args.detection_threshold),
            "detection_threshold": args.detection_threshold,
            "mean_identity_cosine_plus_one": summary.mean_identity_cosine_plus_one,
            "mean_psnr_db": summary.mean_psnr_db,
            "mean_ssim": summary.mean_ssim,
            "per_image": [
                {
                    "path": item.path,
                    "detection_before": item.detection_before,
                    "detection_after": item.detection_after,
                    "identity_cosine_plus_one": item.identity_cosine_plus_one,
                    "psnr_db": item.psnr_db,
                    "ssim": item.ssim,
                    "wall_ms": item.wall_ms,
                }
                for item in summary.per_image
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        log.info("bench.json.written", path=str(args.json))

    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Export a trained generator checkpoint to ONNX."""
    import torch

    from voidface.export.onnx import export_generator_to_onnx
    from voidface.generator.architecture import Voidface, VoidfaceConfig
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="INFO")
    log = get_logger("voidface.cli.export")

    log.info("checkpoint.loading", path=str(args.checkpoint))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored_config = payload.get("config")
        config = stored_config if isinstance(stored_config, VoidfaceConfig) else VoidfaceConfig()
    else:
        state_dict = payload
        config = VoidfaceConfig()

    generator = Voidface(config).eval()
    generator.load_state_dict(state_dict)
    log.info(
        "generator.loaded",
        params=sum(p.numel() for p in generator.parameters()),
    )

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("onnx.exporting", path=str(args.output), resolution=args.example_resolution)
    export_generator_to_onnx(
        generator, args.output, example_resolution=args.example_resolution
    )
    log.info("onnx.exported", size_bytes=args.output.stat().st_size)

    if args.quantize is not None:
        from voidface.export.quantize import quantize_onnx_generator

        quantized_path = args.output.with_suffix(f".{args.quantize}.onnx")
        log.info("quantize.starting", path=str(quantized_path), weight_type=args.quantize)
        quantize_onnx_generator(args.output, quantized_path, weight_type=args.quantize)
        log.info(
            "quantize.done",
            path=str(quantized_path),
            size_bytes=quantized_path.stat().st_size,
        )

    if args.quantize_static_dir is not None:
        from voidface.data.datasets import FolderImageDataset
        from voidface.export.quantize import quantize_onnx_generator_static

        log.info(
            "quantize_static.starting",
            calibration_dir=str(args.quantize_static_dir),
            samples=args.quantize_static_samples,
        )
        dataset = FolderImageDataset(
            args.quantize_static_dir,
            resolution=args.example_resolution,
            augment=False,
        )

        def _iterator():  # noqa: ANN202
            import numpy as np

            limit = min(args.quantize_static_samples, len(dataset))
            for i in range(limit):
                yield dataset[i].unsqueeze(0).cpu().numpy().astype(np.float32)

        static_path = args.output.with_suffix(".static-int8.onnx")
        quantize_onnx_generator_static(args.output, static_path, _iterator())
        log.info(
            "quantize_static.done",
            path=str(static_path),
            size_bytes=static_path.stat().st_size,
        )

    if args.coreml:
        try:
            from voidface.export.coreml import export_generator_to_coreml
        except ImportError as exc:
            log.error("coreml.import_failed", error=str(exc))
            return 3
        coreml_path = args.output.with_suffix(".mlpackage")
        log.info("coreml.exporting", path=str(coreml_path))
        try:
            export_generator_to_coreml(
                generator, coreml_path, example_resolution=args.example_resolution
            )
        except RuntimeError as exc:
            # CoreMlExportError is a RuntimeError; catches the
            # "coremltools not installed" case cleanly.
            log.error("coreml.export_failed", error=str(exc))
            return 3
        log.info("coreml.done", path=str(coreml_path))

    if args.ort:
        from voidface.export.ort import OrtConversionError, convert_onnx_to_ort

        log.info("ort.converting", input=str(args.output))
        try:
            ort_path = convert_onnx_to_ort(args.output, output_dir=args.output.parent)
        except OrtConversionError as exc:
            log.error("ort.convert_failed", error=str(exc))
            return 4
        log.info("ort.done", path=str(ort_path))

    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Train the generator G against a folder of face images."""
    from torch.utils.data import DataLoader

    from voidface.core.eot import EotConfig, EotSampler
    from voidface.core.loss import (
        CompositeLoss,
        LossWeights,
        arcface_identity_loss,
        retinaface_suppression_loss,
        vae_gray_latent_loss,
    )
    from voidface.core.train import TrainConfig, train_generator
    from voidface.data.datasets import FolderImageDataset
    from voidface.eval.perceptual import load_lpips
    from voidface.generator.architecture import Voidface, VoidfaceConfig
    from voidface.models.restorers.identity import IdentityRestorer
    from voidface.models.restorers.sampler import RestorerSampler, SamplerConfig
    from voidface.util.config import load_config
    from voidface.util.log import configure_logging, get_logger

    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli.train")

    config = load_config(args.config)
    device = _resolve_device(args.device)
    log.info("device.selected", device=str(device))

    experiment = config.get("experiment", {})
    data_conf = config.get("data", {})
    optim_conf = config.get("optim", {})
    percep_conf = config.get("loss", {}).get("perceptual", {})
    targets_conf = config.get("targets", {})
    restorers_conf = config.get("restorers", {})

    dataset_dir = Path(data_conf["directory"]).expanduser()
    resolution = int(data_conf.get("resolution", 256))
    batch_size = int(data_conf.get("batch_size", 4))
    augment = bool(data_conf.get("augment", True))

    log.info("dataset.loading", directory=str(dataset_dir), resolution=resolution)
    dataset = FolderImageDataset(dataset_dir, resolution=resolution, augment=augment)
    log.info("dataset.loaded", size=len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Voidface(
        VoidfaceConfig(epsilon=float(optim_conf.get("epsilon_frac", 12.0 / 255.0)))
    )
    log.info(
        "generator.built",
        params=sum(p.numel() for p in generator.parameters()),
    )

    target_losses, target_static_data, weights_targets, vae, sdxl_vae = _build_targets(
        targets_conf, generator_input=next(iter(loader))[:1], device=device, log=log
    )

    lpips_weight = float(percep_conf.get("lpips_weight", 0.10))
    lpips_fn = load_lpips(net="alex", device=device) if lpips_weight > 0 else None
    weights = LossWeights(
        targets=weights_targets,
        lpips=lpips_weight,
        total_variation=float(percep_conf.get("tv_weight", 0.01)),
    )
    composite = CompositeLoss(
        weights=weights,
        target_losses=target_losses,
        target_static_data=target_static_data,
        lpips=lpips_fn,
    )

    eot = EotSampler(EotConfig(samples=int(optim_conf.get("eot_samples", 2))))

    restorer_options: list = []
    for name, weight in restorers_conf.items():
        w = float(weight)
        if w <= 0.0:
            continue
        if name == "identity":
            restorer_options.append((IdentityRestorer(), w))
        elif name == "sd15-vae":
            from voidface.models.restorers.sd_vae import Sd15VaeRestorer

            if vae is None:
                log.error("restorer.sd15_vae.requires_target",
                          hint="enable [targets.vae] in the config")
                return 2
            restorer_options.append((Sd15VaeRestorer(encoder=vae), w))
        elif name == "gfpgan":
            from voidface.models.restorers.gfpgan import GfpganRestorer

            gfpgan_detector = target_losses.get("detector", (None,))[0]
            if gfpgan_detector is None:
                from voidface.models.detectors.retinaface import RetinaFace as _R

                log.info(
                    "model.detector.loading",
                    name="retinaface-r50",
                    reason="gfpgan needs landmarks",
                )
                gfpgan_detector = _R(device=device)
            restorer_options.append(
                (GfpganRestorer(detector=gfpgan_detector, device=device), w)
            )
        else:
            log.error("restorer.unknown", name=name)
            return 2

    if not restorer_options:
        restorer_options.append((IdentityRestorer(), 1.0))
    restorer_sampler = RestorerSampler(restorer_options, SamplerConfig(seed=0))

    train_config = TrainConfig(
        steps=int(experiment.get("steps", 1000)),
        learning_rate=float(optim_conf.get("learning_rate", 1e-4)),
        weight_decay=float(optim_conf.get("weight_decay", 1e-6)),
        log_every=1 if args.verbose else int(experiment.get("log_every", 100)),
        checkpoint_every=int(experiment.get("checkpoint_every", 1000)),
        checkpoint_dir=Path(experiment["checkpoint_dir"]).expanduser()
        if "checkpoint_dir" in experiment
        else None,
        device=str(device),
        seed=int(experiment.get("seed", 0)),
    )

    log.info(
        "train.start",
        steps=train_config.steps,
        batch_size=batch_size,
        targets=sorted(weights_targets),
    )
    result = train_generator(
        generator=generator,
        batches=loader,
        composite_loss=composite,
        eot=eot,
        config=train_config,
        restorer_sampler=restorer_sampler,
    )
    log.info(
        "train.done",
        steps=len(result.history),
        checkpoint=str(result.checkpoint_path)
        if result.checkpoint_path is not None
        else None,
        final_loss=round(result.history[-1].total_loss, 4) if result.history else None,
    )
    return 0


def _build_targets(
    targets_conf: dict,  # noqa: ANN001
    generator_input,  # noqa: ANN001
    device,  # noqa: ANN001
    log,  # noqa: ANN001
):  # noqa: ANN202
    """Assemble target_losses + target_static_data + weights for train_generator."""
    from voidface.core.loss import (
        arcface_identity_loss,
        retinaface_suppression_loss,
        vae_gray_latent_loss,
    )

    target_losses: dict = {}
    target_static_data: dict = {}
    weights_targets: dict[str, float] = {}
    vae = None
    sdxl_vae = None

    def _weight(name: str, default: float) -> float:
        return float(targets_conf.get(name, {}).get("weight", default))

    if targets_conf.get("detector", {}).get("enabled", False):
        from voidface.models.detectors.retinaface import RetinaFace

        log.info("model.detector.loading", name="retinaface-r50")
        detector = RetinaFace(device=device)
        target_losses["detector"] = (detector, retinaface_suppression_loss)
        weights_targets["detector"] = _weight("detector", 0.35)

    if targets_conf.get("recognizer", {}).get("enabled", False):
        from voidface.models.recognizers.arcface import Arcface

        log.info("model.recognizer.loading", name="arcface-r100")
        recognizer = Arcface(device=device)
        target_losses["recognizer"] = (recognizer, arcface_identity_loss)
        weights_targets["recognizer"] = _weight("recognizer", 0.40)

    if targets_conf.get("vae", {}).get("enabled", False):
        from voidface.models.vaes.sd15 import Sd15Vae

        log.info("model.vae.loading", name="sd15-vae")
        vae = Sd15Vae(device=device)
        gray = vae.encode_gray_target(
            height=generator_input.shape[-2], width=generator_input.shape[-1]
        )
        target_losses["vae"] = (vae, vae_gray_latent_loss)
        target_static_data["vae"] = gray
        weights_targets["vae"] = _weight("vae", 0.20)

    if targets_conf.get("sdxl-vae", {}).get("enabled", False):
        from voidface.models.vaes.sdxl import SdxlVae

        log.info("model.sdxl-vae.loading", name="sdxl-vae")
        sdxl_vae = SdxlVae(device=device)
        gray = sdxl_vae.encode_gray_target(
            height=generator_input.shape[-2], width=generator_input.shape[-1]
        )
        target_losses["sdxl-vae"] = (sdxl_vae, vae_gray_latent_loss)
        target_static_data["sdxl-vae"] = gray
        weights_targets["sdxl-vae"] = _weight("sdxl-vae", 0.15)

    if targets_conf.get("openclip", {}).get("enabled", False):
        from voidface.models.clip.openclip import OpenClip

        log.info("model.openclip.loading", name="openclip-vit-b-32")
        openclip = OpenClip(device=device)
        target_losses["openclip"] = (openclip, arcface_identity_loss)
        weights_targets["openclip"] = _weight("openclip", 0.10)

    _renormalize(weights_targets)
    return target_losses, target_static_data, weights_targets, vae, sdxl_vae


def _parse_restorer_spec(spec: str) -> list[tuple[str, float]] | None:
    """Parse a ``--restorers`` argument into ``[(name, weight), ...]``.

    Accepts ``"identity"``, ``"sd15-vae:0.7"``, or comma-separated
    combinations. Returns ``None`` on any unknown name.
    """
    result: list[tuple[str, float]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            name, weight_str = token.split(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                return None
        else:
            name, weight = token, 1.0
        if name not in _ALLOWED_RESTORERS:
            return None
        result.append((name, weight))
    if not result:
        return None
    return result


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
