# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface bench` — release-gate metrics for a trained generator."""

from __future__ import annotations

import argparse
import json
import sys

import torch

from voidface.data.datasets import FolderImageDataset, collect_image_paths
from voidface.eval.benchmark import BenchConfig, run_bench
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.models.base import TargetOutputs
from voidface.models.detectors.retinaface import RetinaFace
from voidface.models.recognizers.arcface import Arcface
from voidface.util.log import configure_logging, get_logger
from voidface_cli.common import resolve_device


def run(args: argparse.Namespace) -> int:
    """Benchmark a trained generator against a folder of test images."""
    configure_logging(level="INFO")
    log = get_logger("voidface.cli.bench")
    device = resolve_device(args.device)

    if not args.checkpoint.exists():
        log.error(
            "checkpoint.not_found",
            path=str(args.checkpoint),
            hint="produce one with `voidface train cfg.toml` or download a release .pt",
        )
        return 2
    if not args.images.exists():
        log.error(
            "images.dir.not_found",
            path=str(args.images),
            hint="point at a directory of face crops to bench against",
        )
        return 2

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

    bench_targets = {t.strip() for t in args.targets.split(",") if t.strip()}

    class _Neutral(torch.nn.Module):
        """Fallback used when a family is disabled — full presence / identity."""

        def forward(self, image):
            n = image.size(0)
            if "detector" in bench_targets:
                return TargetOutputs(embedding=torch.ones(n, 4) / 2.0)
            return TargetOutputs(
                logits=torch.zeros(n, 1, 2),
                embedding=torch.ones(n, 4) / 2.0,
            )

    if "detector" in bench_targets:
        log.info("model.detector.loading", name="retinaface-r50")
        detector = RetinaFace(device=device)
    else:
        detector = _Neutral()
    if "recognizer" in bench_targets:
        log.info("model.recognizer.loading", name="arcface-r100")
        recognizer = Arcface(device=device)
    else:
        recognizer = _Neutral()

    image_paths = collect_image_paths(args.images, recursive=False)
    total = len(dataset)
    limit = args.limit if args.limit > 0 else total
    limit = min(limit, total)
    log.info("bench.limit", limit=limit, total=total)
    image_names = [str(p) for p in image_paths[:limit]]
    summary = run_bench(
        generator=generator,
        images=(dataset[i] for i in range(limit)),
        detector=detector,
        recognizer=recognizer,
        config=BenchConfig(
            device=str(device),
            detection_threshold=args.detection_threshold,
        ),
        output_dir=args.out_dir,
        image_names=image_names,
    )

    print("--- bench summary ---")
    print(f"images:          {summary.count}")
    print(f"detection ASR:   {summary.detection_asr(args.detection_threshold):.4f}")
    print(
        f"identity cos+1:  {summary.mean_identity_cosine_plus_one:.4f}  "
        f"(0=success, 2=failure)"
    )
    print(f"PSNR (mean):     {summary.mean_psnr_db:.2f} dB")
    print(f"SSIM (mean):     {summary.mean_ssim:.4f}")

    if args.json is not None:
        json_payload = {
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
        args.json.write_text(json.dumps(json_payload, indent=2))
        log.info("bench.json.written", path=str(args.json))

    if args.baseline is not None:
        if not args.baseline.exists():
            print(f"error: baseline file not found: {args.baseline}", file=sys.stderr)
            return 2
        baseline = json.loads(args.baseline.read_text())
        current = {
            "detection_asr": summary.detection_asr(args.detection_threshold),
            "mean_identity_cosine_plus_one": summary.mean_identity_cosine_plus_one,
            "mean_psnr_db": summary.mean_psnr_db,
            "mean_ssim": summary.mean_ssim,
        }
        print("--- baseline comparison ---")
        regressed = False
        for name in ("detection_asr", "mean_psnr_db", "mean_ssim"):
            base = float(baseline.get(name, 0))
            cur = current[name]
            delta = cur - base
            symbol = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            if delta < 0:
                regressed = True
            print(f"  {name:32s}  {base:.4f}  ->  {cur:.4f}  {symbol}{abs(delta):.4f}")
        base = float(baseline.get("mean_identity_cosine_plus_one", 2))
        cur = current["mean_identity_cosine_plus_one"]
        delta = cur - base
        symbol = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        if delta > 0:
            regressed = True
        print(
            f"  {'mean_identity_cosine_plus_one':32s}  {base:.4f}  ->  "
            f"{cur:.4f}  {symbol}{abs(delta):.4f}  (lower is better)"
        )
        if regressed:
            print("VERDICT: regression")
            return 1
        print("VERDICT: no regression")

    if args.strict:
        det_asr = summary.detection_asr(args.detection_threshold)
        id_cos = summary.mean_identity_cosine_plus_one
        psnr_val = summary.mean_psnr_db
        ssim_val = summary.mean_ssim
        print("--- ship gate ---")
        gate_failures: list[str] = []
        checks = (
            ("detection_asr", det_asr, args.strict_detection_asr, ">="),
            ("identity_cos_plus_one", id_cos, args.strict_identity_cos, "<="),
            ("mean_psnr_db", psnr_val, args.strict_psnr, ">="),
            ("mean_ssim", ssim_val, args.strict_ssim, ">="),
        )
        for name, actual, threshold, op in checks:
            ok = actual >= threshold if op == ">=" else actual <= threshold
            status = "PASS" if ok else "FAIL"
            print(f"  {name:24s}  {actual:.4f}  {op}  {threshold:.4f}   {status}")
            if not ok:
                gate_failures.append(name)
        if gate_failures:
            print(f"VERDICT: ship gate FAILED on {', '.join(gate_failures)}")
            return 3
        print("VERDICT: ship gate PASS")

    return 0
