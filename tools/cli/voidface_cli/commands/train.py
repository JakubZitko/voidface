# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface train` — end-to-end generator training loop.

Consumes a TOML config and runs :func:`voidface.core.train.train_generator`
against the configured target ensemble + restorer sampler.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
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
from voidface_cli.common import renormalize_weights, resolve_device


def _build_targets(
    targets_conf: dict[str, Any],
    generator_input: torch.Tensor,
    device: object,
    log: Any,
) -> tuple[dict, dict, dict[str, float], Any, Any]:
    """Assemble target_losses + target_static_data + weights for train_generator."""
    target_losses: dict = {}
    target_static_data: dict = {}
    weights_targets: dict[str, float] = {}
    vae = None
    sdxl_vae = None

    def _weight(name: str, default: float) -> float:
        return float(targets_conf.get(name, {}).get("weight", default))

    if targets_conf.get("detector", {}).get("enabled", False):
        from voidface.models.detectors.retinaface import RetinaFace  # noqa: PLC0415

        log.info("model.detector.loading", name="retinaface-r50")
        detector = RetinaFace(device=device)
        target_losses["detector"] = (detector, retinaface_suppression_loss)
        weights_targets["detector"] = _weight("detector", 0.35)

    if targets_conf.get("recognizer", {}).get("enabled", False):
        from voidface.models.recognizers.arcface import Arcface  # noqa: PLC0415

        log.info("model.recognizer.loading", name="arcface-r100")
        recognizer = Arcface(device=device)
        target_losses["recognizer"] = (recognizer, arcface_identity_loss)
        weights_targets["recognizer"] = _weight("recognizer", 0.40)

    if targets_conf.get("vae", {}).get("enabled", False):
        from voidface.models.vaes.sd15 import Sd15Vae  # noqa: PLC0415

        log.info("model.vae.loading", name="sd15-vae")
        vae = Sd15Vae(device=device)
        gray = vae.encode_gray_target(
            height=generator_input.shape[-2], width=generator_input.shape[-1]
        )
        target_losses["vae"] = (vae, vae_gray_latent_loss)
        target_static_data["vae"] = gray
        weights_targets["vae"] = _weight("vae", 0.20)

    if targets_conf.get("sdxl-vae", {}).get("enabled", False):
        from voidface.models.vaes.sdxl import SdxlVae  # noqa: PLC0415

        log.info("model.sdxl-vae.loading", name="sdxl-vae")
        sdxl_vae = SdxlVae(device=device)
        gray = sdxl_vae.encode_gray_target(
            height=generator_input.shape[-2], width=generator_input.shape[-1]
        )
        target_losses["sdxl-vae"] = (sdxl_vae, vae_gray_latent_loss)
        target_static_data["sdxl-vae"] = gray
        weights_targets["sdxl-vae"] = _weight("sdxl-vae", 0.15)

    if targets_conf.get("openclip", {}).get("enabled", False):
        from voidface.models.clip.openclip import OpenClip  # noqa: PLC0415

        log.info("model.openclip.loading", name="openclip-vit-b-32")
        openclip = OpenClip(device=device)
        target_losses["openclip"] = (openclip, arcface_identity_loss)
        weights_targets["openclip"] = _weight("openclip", 0.10)

    renormalize_weights(weights_targets)
    return target_losses, target_static_data, weights_targets, vae, sdxl_vae


def run(args: argparse.Namespace) -> int:
    """Train the generator G against a folder of face images."""
    configure_logging(level="DEBUG" if args.verbose else "INFO")
    log = get_logger("voidface.cli.train")

    config = load_config(args.config)
    device = resolve_device(args.device)
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
    if args.resume is not None:
        log.info("generator.resume", path=str(args.resume))
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        state_dict = (
            payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload
            else payload
        )
        generator.load_state_dict(state_dict)
        resumed_step = payload.get("step") if isinstance(payload, dict) else None
        log.info("generator.resumed", from_step=resumed_step)

    target_losses, target_static_data, weights_targets, vae, _sdxl_vae = _build_targets(
        targets_conf, generator_input=next(iter(loader))[:1], device=device, log=log
    )

    lpips_weight = float(percep_conf.get("lpips_weight", 0.10))
    lpips_fn = load_lpips(net="alex", device=device) if lpips_weight > 0 else None
    loss_conf = config.get("loss", {})
    weights = LossWeights(
        targets=weights_targets,
        lpips=lpips_weight,
        total_variation=float(percep_conf.get("tv_weight", 0.01)),
        bilevel_lpips=float(loss_conf.get("bilevel_lpips", 0.0)),
        normalize_per_target=bool(loss_conf.get("normalize_per_target", False)),
        normalization_ema_decay=float(loss_conf.get("normalization_ema_decay", 0.99)),
    )
    composite = CompositeLoss(
        weights=weights,
        target_losses=target_losses,
        target_static_data=target_static_data,
        lpips=lpips_fn,
    )

    eot_conf = config.get("eot", {})
    eot = EotSampler(
        EotConfig(
            samples=int(eot_conf.get("k", optim_conf.get("eot_samples", 2))),
            resize_factors=tuple(eot_conf.get("resize_factors", (0.75, 1.0, 1.5))),
            gaussian_sigma=tuple(eot_conf.get("gaussian_sigma", (0.0, 0.5, 1.0))),
            jpeg_qualities=tuple(int(q) for q in eot_conf.get("jpeg_qualities", ())),
            seed=int(experiment.get("seed", 0)),
        )
    )
    log.info(
        "eot.configured",
        samples=eot._config.samples,
        resize_factors=eot._config.resize_factors,
        gaussian_sigma=eot._config.gaussian_sigma,
        jpeg_qualities=eot._config.jpeg_qualities,
    )

    restorer_options: list = []
    for name, weight in restorers_conf.items():
        w = float(weight)
        if w <= 0.0:
            continue
        if name == "identity":
            restorer_options.append((IdentityRestorer(), w))
        elif name == "sd15-vae":
            from voidface.models.restorers.sd_vae import Sd15VaeRestorer  # noqa: PLC0415

            if vae is None:
                log.error(
                    "restorer.sd15_vae.requires_target",
                    hint="enable [targets.vae] in the config",
                )
                return 2
            restorer_options.append((Sd15VaeRestorer(encoder=vae), w))
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
            restorer_options.append(
                (GfpganRestorer(detector=gfpgan_detector, device=device), w)
            )
        else:
            log.error("restorer.unknown", name=name)
            return 2

    if not restorer_options:
        restorer_options.append((IdentityRestorer(), 1.0))
    restorer_sampler = RestorerSampler(restorer_options, SamplerConfig(seed=0))

    effective_seed = args.seed if args.seed is not None else int(experiment.get("seed", 0))
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
        seed=effective_seed,
    )

    log.info(
        "train.start",
        steps=train_config.steps,
        batch_size=batch_size,
        targets=sorted(weights_targets),
        restorers=restorer_sampler.probabilities(),
    )

    if args.dry_run:
        print("--- dry run ---")
        print(f"config:          {args.config}")
        print(f"dataset:         {dataset_dir}  ({len(dataset)} images)")
        print(f"resolution:      {resolution}  batch: {batch_size}")
        print(f"generator params:{sum(p.numel() for p in generator.parameters()):,}")
        print(f"targets:         {sorted(weights_targets)}")
        print(f"restorer mix:    {restorer_sampler.probabilities()}")
        print(f"steps planned:   {train_config.steps}")
        print(f"lr:              {train_config.learning_rate}")
        print(f"lpips weight:    {lpips_weight}")
        print(f"bilevel LPIPS:   {weights.bilevel_lpips}")
        print(f"normalize:       {weights.normalize_per_target}")
        print(f"eot samples:     {eot._config.samples}")
        print(f"jpeg qualities:  {list(eot._config.jpeg_qualities)}")
        print("Dry run complete — no training steps executed.")
        return 0

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
