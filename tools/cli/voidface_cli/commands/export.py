# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface export` — trained generator to ONNX/int8/CoreML/ORT."""

from __future__ import annotations

import argparse

import torch

from voidface.export.onnx import export_generator_to_onnx
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.util.log import configure_logging, get_logger


def run(args: argparse.Namespace) -> int:
    """Export a trained generator checkpoint to ONNX (+ optional int8/CoreML/ORT)."""
    configure_logging(level="INFO")
    log = get_logger("voidface.cli.export")

    if not args.checkpoint.exists():
        log.error(
            "checkpoint.not_found",
            path=str(args.checkpoint),
            hint="produce one with `voidface train cfg.toml` or download a release .pt",
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

        from collections.abc import Iterator

        import numpy as np

        def _iterator() -> Iterator[np.ndarray]:  # type: ignore[type-arg]
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
