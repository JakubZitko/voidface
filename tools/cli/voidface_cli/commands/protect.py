# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface protect` — single image / batch / hybrid / generator-fast-path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import voidface
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
from voidface.util.checksum import compute_sha256
from voidface.util.image import load_image, save_image
from voidface.util.log import configure_logging, get_logger
from voidface_cli.common import (
    ALLOWED_RESTORERS,
    ALLOWED_TARGETS,
    load_generator_checkpoint,
    renormalize_weights,
    resolve_device,
    run_generator_and_save,
)


def _parse_restorer_spec(spec: str) -> list[tuple[str, float]] | None:
    """Parse a ``--restorers`` argument into ``[(name, weight), ...]``.

    Accepts ``"identity"``, ``"sd15-vae:0.7"``, or comma-separated
    combinations. Returns ``None`` on any unknown name.
    """
    result: list[tuple[str, float]] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
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
        if name not in ALLOWED_RESTORERS:
            return None
        result.append((name, weight))
    if not result:
        return None
    return result


def _print_summary(clean: object, adversarial: object, output: Path) -> None:
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


def _print_attack_metrics(clean: torch.Tensor, adversarial: torch.Tensor, log: Any) -> None:
    """Run the real ensemble on both clean and protected and print ASR-style metrics."""
    from voidface.eval.benchmark import _detection_face_score  # noqa: PLC0415
    from voidface.models.detectors.retinaface import RetinaFace  # noqa: PLC0415
    from voidface.models.recognizers.arcface import Arcface  # noqa: PLC0415

    log.info("metrics.model.detector.loading", name="retinaface-r50")
    detector = RetinaFace(device=clean.device)
    log.info("metrics.model.recognizer.loading", name="arcface-r100")
    recognizer = Arcface(device=clean.device)

    with torch.no_grad():
        det_before = _detection_face_score(detector(clean))
        det_after = _detection_face_score(detector(adversarial))
        clean_id = recognizer(clean).embedding
        adv_id = recognizer(adversarial).embedding
        assert clean_id is not None
        assert adv_id is not None
        cos = F.cosine_similarity(clean_id, adv_id, dim=-1).mean().item()

    print("--- attack metrics ---")
    print(f"detector face score (before -> after):  {det_before:.4f} -> {det_after:.4f}")
    print(f"ArcFace cosine (clean vs protected):    {cos:.4f}  (1=same, -1=opposite)")


def _write_output_json(
    args: argparse.Namespace,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    output_path: Path,
) -> None:
    """Write an audit-trail sidecar next to the protected image."""
    if not isinstance(clean, torch.Tensor) or not isinstance(adversarial, torch.Tensor):
        return

    metadata = {
        "voidface_version": voidface.__version__,
        "output": str(output_path),
        "psnr_db": psnr(clean, adversarial),
        "ssim": ssim(clean, adversarial),
        "l_inf": (clean - adversarial).abs().max().item(),
        "epsilon_int_over_255": args.epsilon,
        "seed": args.seed,
        "flags": {
            "use_generator": str(args.use_generator) if args.use_generator else None,
            "refine_steps": args.refine_steps,
            "targets": args.targets,
            "restorers": args.restorers,
            "steps": args.steps,
            "no_lpips": args.no_lpips,
            "face_mask": args.face_mask,
            "semantic_warp": args.semantic_warp,
        },
    }
    if args.use_generator is not None and args.use_generator.exists():
        metadata["generator_checkpoint_sha256"] = compute_sha256(args.use_generator)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metadata, indent=2))


def _protect_batch(args: argparse.Namespace, device: object, log: Any) -> int:
    """Process every image in a directory, writing outputs to --output-dir.

    Batch mode currently requires --use-generator — running per-image
    PGD on a large folder is impractical (~2 min per image). The
    generator is loaded exactly once regardless of batch size.
    """
    from voidface.data.datasets import collect_image_paths  # noqa: PLC0415

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
    if not paths:
        log.error(
            "batch.no_images_found",
            path=str(args.image),
            hint=(
                "batch mode expects .png/.jpg/.jpeg/.webp files in the input "
                "directory; pass --recursive to walk subdirectories"
            ),
        )
        return 2
    log.info("batch.starting", count=len(paths), output_dir=str(args.output_dir))

    generator, config = load_generator_checkpoint(args.use_generator, device, log)

    for index, path in enumerate(paths):
        log.info("batch.item", index=index + 1, total=len(paths), path=str(path))
        clean = load_image(path).to(device).unsqueeze(0)
        output_path = args.output_dir / (path.stem + ".protected.png")
        run_generator_and_save(
            generator=generator,
            config=config,
            clean=clean,
            output_path=output_path,
            epsilon_int=args.epsilon,
            face_mask=args.face_mask,
        )
        log.info("batch.item.done", path=str(output_path))
    log.info("batch.done", count=len(paths))
    return 0


def _protect_via_generator(args: argparse.Namespace, clean: torch.Tensor, log: Any) -> int:
    """`voidface protect --use-generator` fast path: one G forward."""
    t_load_start = time.perf_counter()
    generator, config = load_generator_checkpoint(args.use_generator, clean.device, log)
    t_load = time.perf_counter() - t_load_start
    output = args.output or args.image.with_suffix(".protected.png")
    t_run_start = time.perf_counter()
    adversarial = run_generator_and_save(
        generator=generator,
        config=config,
        clean=clean,
        output_path=output,
        epsilon_int=args.epsilon,
        face_mask=args.face_mask,
    )
    t_run = time.perf_counter() - t_run_start
    log.info("image.saved", path=str(output))
    if not args.quiet:
        _print_summary(clean=clean, adversarial=adversarial, output=output)
    if getattr(args, "show_metrics", False):
        _print_attack_metrics(clean=clean, adversarial=adversarial, log=log)
    if args.output_json is not None:
        _write_output_json(args, clean, adversarial, output)
        log.info("output_json.written", path=str(args.output_json))
    if args.emit_delta is not None:
        args.emit_delta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(adversarial - clean, args.emit_delta)
        log.info("delta.written", path=str(args.emit_delta))
    if args.timing:
        print("--- timing ---")
        print(f"  checkpoint load:  {t_load * 1000:.1f} ms")
        print(f"  forward + save:   {t_run * 1000:.1f} ms")
    return 0


def run(args: argparse.Namespace) -> int:  # noqa: PLR0911
    """Main entry point for `voidface protect`."""
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli")

    if args.refine_steps > 0 and args.use_generator is None:
        log.error(
            "refine_steps.without_generator",
            hint="--refine-steps N requires --use-generator CKPT",
        )
        return 2

    if getattr(args, "iris_boost", False) and args.iris_ratio < 1.0:
        log.error(
            "iris_ratio.below_one",
            got=args.iris_ratio,
            hint="--iris-ratio must be >= 1.0 (1.0 means no boost)",
        )
        return 2

    device = resolve_device(args.device)
    log.info("device.selected", device=str(device))

    if not args.image.exists():
        log.error(
            "image.not_found",
            path=str(args.image),
            hint="pass a real image path, or a directory for batch mode",
        )
        return 2

    if args.use_generator is not None and not args.use_generator.exists():
        log.error(
            "generator.checkpoint.not_found",
            path=str(args.use_generator),
            hint="produce one with `voidface train cfg.toml` or download a release .pt",
        )
        return 2

    if args.image.is_dir():
        return _protect_batch(args, device, log)

    log.info("image.loading", path=str(args.image))
    clean = load_image(args.image).to(device).unsqueeze(0)
    log.info("image.loaded", shape=tuple(clean.shape))

    if args.use_generator is not None and args.refine_steps <= 0:
        return _protect_via_generator(args, clean, log)

    warm_start_delta: torch.Tensor | None = None
    if args.use_generator is not None and args.refine_steps > 0:
        generator, config = load_generator_checkpoint(args.use_generator, device, log)
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
            adv_from_gen = generator(clean_padded, epsilon=args.epsilon / 255.0)
        adv_from_gen = adv_from_gen[..., : original_hw[0], : original_hw[1]]
        warm_start_delta = adv_from_gen - clean
        log.info("pgd.warm_start", from_generator=str(args.use_generator))

    selected = {t.strip() for t in args.targets.split(",") if t.strip()}
    if not selected.issubset(ALLOWED_TARGETS):
        unknown = sorted(selected - ALLOWED_TARGETS)
        log.error("targets.unknown", unknown=unknown, allowed=sorted(ALLOWED_TARGETS))
        return 2

    restorer_spec = _parse_restorer_spec(args.restorers)
    if restorer_spec is None:
        log.error("restorers.unknown", spec=args.restorers)
        return 2
    if "sd15-vae" in {name for name, _ in restorer_spec} and "vae" not in selected:
        selected.add("vae")
        log.info("targets.autoadd", added="vae", reason="sd15-vae restorer selected")

    log.info("targets.selected", targets=sorted(selected))
    log.info("restorers.selected", spec=restorer_spec)

    target_losses: dict = {}
    target_static_data: dict = {}
    weights_targets: dict[str, float] = {}

    if "detector" in selected:
        from voidface.models.detectors.retinaface import RetinaFace  # noqa: PLC0415

        log.info("model.detector.loading", name="retinaface-r50")
        detector = RetinaFace(device=device)
        target_losses["detector"] = (detector, retinaface_suppression_loss)
        weights_targets["detector"] = 0.35

    if "recognizer" in selected:
        from voidface.models.recognizers.arcface import Arcface  # noqa: PLC0415

        log.info("model.recognizer.loading", name="arcface-r100")
        recognizer = Arcface(device=device)
        target_losses["recognizer"] = (recognizer, arcface_identity_loss)
        weights_targets["recognizer"] = 0.40

    vae = None
    if "vae" in selected:
        from voidface.models.vaes.sd15 import Sd15Vae  # noqa: PLC0415

        log.info("model.vae.loading", name="sd15-vae")
        vae = Sd15Vae(device=device)
        gray_target = vae.encode_gray_target(height=clean.shape[-2], width=clean.shape[-1])
        target_losses["vae"] = (vae, vae_gray_latent_loss)
        target_static_data["vae"] = gray_target
        weights_targets["vae"] = 0.20

    if "sdxl-vae" in selected:
        from voidface.models.vaes.sdxl import SdxlVae  # noqa: PLC0415

        log.info("model.sdxl-vae.loading", name="sdxl-vae")
        sdxl_vae = SdxlVae(device=device)
        sdxl_gray_target = sdxl_vae.encode_gray_target(
            height=clean.shape[-2], width=clean.shape[-1]
        )
        target_losses["sdxl-vae"] = (sdxl_vae, vae_gray_latent_loss)
        target_static_data["sdxl-vae"] = sdxl_gray_target
        weights_targets["sdxl-vae"] = 0.15

    if "openclip" in selected:
        from voidface.models.clip.openclip import OpenClip  # noqa: PLC0415

        log.info("model.openclip.loading", name="openclip-vit-b-32")
        openclip = OpenClip(device=device)
        target_losses["openclip"] = (openclip, arcface_identity_loss)
        weights_targets["openclip"] = 0.10

    if not target_losses:
        log.error("targets.empty")
        return 2

    renormalize_weights(weights_targets)

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
    effective_steps = args.refine_steps if args.refine_steps > 0 else args.steps
    pgd = PgdConfig(
        epsilon=args.epsilon / 255.0,
        alpha=max(1, args.epsilon // 6) / 255.0,
        steps=effective_steps,
        momentum=0.9,
        log_every=1 if args.verbose else max(1, effective_steps // 5),
        seed=args.seed,
        initial_delta=warm_start_delta,
    )

    restorer_options: list = []
    for name, weight in restorer_spec:
        if name == "identity":
            restorer_options.append((IdentityRestorer(), weight))
        elif name == "sd15-vae":
            from voidface.models.restorers.sd_vae import Sd15VaeRestorer  # noqa: PLC0415

            assert vae is not None, "sd15-vae restorer requires the VAE target."
            restorer_options.append((Sd15VaeRestorer(encoder=vae), weight))
        elif name == "gfpgan":
            from voidface.models.restorers.gfpgan import GfpganRestorer  # noqa: PLC0415

            gfpgan_detector = target_losses.get("detector", (None,))[0]
            if gfpgan_detector is None:
                from voidface.models.detectors.retinaface import RetinaFace  # noqa: PLC0415

                log.info(
                    "model.detector.loading",
                    name="retinaface-r50",
                    reason="gfpgan needs landmarks",
                )
                gfpgan_detector = RetinaFace(device=device)
            log.info("model.gfpgan.loading", name="gfpgan-v1.4")
            restorer_options.append(
                (GfpganRestorer(detector=gfpgan_detector, device=device), weight)
            )

    restorer_sampler = RestorerSampler(restorer_options, SamplerConfig(seed=args.seed))
    log.info(
        "pgd.start",
        epsilon=args.epsilon,
        steps=args.steps,
        targets=sorted(weights_targets),
        lpips=(lpips_weight > 0.0),
        restorers=restorer_sampler.probabilities(),
    )

    iris_mask = None
    if getattr(args, "iris_boost", False):
        detector_pair = target_losses.get("detector")
        if detector_pair is None:
            log.error(
                "iris_boost.needs_detector",
                hint="--iris-boost requires 'detector' in --targets",
            )
            return 2
        iris_detector = detector_pair[0]
        from voidface.attacks.iris import iris_region_mask  # noqa: PLC0415
        from voidface.models.restorers.gfpgan import pick_top_landmarks  # noqa: PLC0415

        with torch.no_grad():
            det_out = iris_detector(clean)
            landmarks = pick_top_landmarks(det_out, clean, threshold=0.5)
        if landmarks is None:
            log.warn(
                "iris_boost.no_face",
                hint="detector found no face above threshold; iris boost skipped",
            )
        else:
            iris_mask = iris_region_mask(
                landmarks, height=clean.shape[-2], width=clean.shape[-1]
            )
            log.info(
                "iris_boost.applied",
                ratio=args.iris_ratio,
                mask_coverage=float(iris_mask.mean().item()),
            )
            if getattr(args, "dump_iris_mask", None) is not None:
                args.dump_iris_mask.parent.mkdir(parents=True, exist_ok=True)
                # Broadcast (N, 1, H, W) -> (3, H, W) grayscale for save_image.
                mask_rgb = iris_mask[0].expand(3, -1, -1)
                save_image(mask_rgb, args.dump_iris_mask)
                log.info("iris_boost.mask_dumped", path=str(args.dump_iris_mask))

    result = run_pgd(
        clean=clean,
        composite_loss=composite,
        eot=eot,
        config=pgd,
        restorer_sampler=restorer_sampler,
        semantic_warp_max_pixels=args.semantic_warp,
        iris_mask=iris_mask,
        iris_epsilon_ratio=args.iris_ratio,
    )
    log.info("pgd.done", final=round(result.history[-1].total_loss, 4))

    adversarial = result.adversarial
    if args.face_mask:
        from voidface.util.facemask import face_region_mask  # noqa: PLC0415

        mask = face_region_mask(clean.squeeze(0)).to(device=clean.device)
        delta = adversarial - clean
        adversarial = (clean + delta * mask.unsqueeze(0)).clamp(0.0, 1.0)

    output = args.output or args.image.with_suffix(".protected.png")
    save_image(adversarial.squeeze(0), output)
    log.info("image.saved", path=str(output))

    if not args.quiet:
        _print_summary(clean=clean, adversarial=adversarial, output=output)
    if args.output_json is not None:
        _write_output_json(args, clean, adversarial, output)
        log.info("output_json.written", path=str(args.output_json))
    if args.emit_delta is not None:
        args.emit_delta.parent.mkdir(parents=True, exist_ok=True)
        torch.save(adversarial - clean, args.emit_delta)
        log.info("delta.written", path=str(args.emit_delta))
    return 0
