# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface package` — bundle a release directory (ONNX + int8 + ORT + CoreML)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import voidface
from voidface.export.onnx import export_generator_to_onnx
from voidface.export.ort import OrtConversionError, convert_onnx_to_ort
from voidface.export.quantize import quantize_onnx_generator
from voidface.generator.architecture import Voidface, VoidfaceConfig
from voidface.util.checksum import compute_sha256
from voidface.util.log import configure_logging, get_logger


def run(args: argparse.Namespace) -> int:
    """Bundle a full release into a directory with checksums + README."""
    configure_logging(level="INFO")
    log = get_logger("voidface.cli.package")

    if not args.checkpoint.exists():
        log.error(
            "checkpoint.not_found",
            path=str(args.checkpoint),
            hint="produce one with `voidface train cfg.toml` or download a release .pt",
        )
        return 2

    if getattr(args, "dry_run", False):
        planned = ["onnx (fp32)", "int8 (dynamic)"]
        if args.calibration_dir is not None:
            planned.append("static-int8")
        planned.append("ort (best-effort)")
        if args.coreml:
            planned.append("coreml (.mlpackage; Apple Silicon only)")
        planned.append("CHECKSUMS.sha256 + MANIFEST.json + README")
        print("--- package dry run ---")
        print(f"checkpoint:        {args.checkpoint}")
        print(f"output_dir:        {args.output_dir}")
        print(f"name:              {args.name}")
        print(f"example_resolution:{args.example_resolution}")
        print(f"calibration_dir:   {args.calibration_dir or '(none)'}")
        print("planned artifacts:")
        for item in planned:
            print(f"  - {item}")
        print("Dry run complete — no exports executed.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("package.start", output_dir=str(args.output_dir))

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored = payload.get("config")
        config = stored if isinstance(stored, VoidfaceConfig) else VoidfaceConfig()
        step = payload.get("step")
    else:
        state_dict = payload
        config = VoidfaceConfig()
        step = None
    generator = Voidface(config).eval()
    generator.load_state_dict(state_dict)

    artifacts: dict[str, Path] = {}

    onnx_path = args.output_dir / f"{args.name}.onnx"
    log.info("package.onnx", path=str(onnx_path))
    export_generator_to_onnx(generator, onnx_path, example_resolution=args.example_resolution)
    artifacts["onnx"] = onnx_path

    dyn_path = args.output_dir / f"{args.name}.int8.onnx"
    log.info("package.int8", path=str(dyn_path))
    quantize_onnx_generator(onnx_path, dyn_path, weight_type="int8")
    artifacts["int8"] = dyn_path

    if args.calibration_dir is not None:
        from voidface.data.datasets import FolderImageDataset
        from voidface.export.quantize import quantize_onnx_generator_static

        static_path = args.output_dir / f"{args.name}.static-int8.onnx"
        log.info("package.static-int8", path=str(static_path))
        dataset = FolderImageDataset(
            args.calibration_dir, resolution=args.example_resolution, augment=False
        )

        from collections.abc import Iterator

        import numpy as np

        def _iter() -> Iterator[np.ndarray]:  # type: ignore[type-arg]
            for i in range(min(64, len(dataset))):
                yield dataset[i].unsqueeze(0).cpu().numpy().astype(np.float32)

        quantize_onnx_generator_static(onnx_path, static_path, _iter())
        artifacts["static-int8"] = static_path

    try:
        ort_path = convert_onnx_to_ort(onnx_path, args.output_dir)
        log.info("package.ort", path=str(ort_path))
        artifacts["ort"] = ort_path
    except OrtConversionError as exc:
        log.warn("package.ort.skip", error=str(exc))

    if args.coreml:
        try:
            from voidface.export.coreml import export_generator_to_coreml

            coreml_path = args.output_dir / f"{args.name}.mlpackage"
            export_generator_to_coreml(
                generator, coreml_path, example_resolution=args.example_resolution
            )
            artifacts["coreml"] = coreml_path
        except RuntimeError as exc:
            log.warn("package.coreml.skip", error=str(exc))

    checksums: dict[str, str] = {}
    for name, path in artifacts.items():
        if path.is_file():
            checksums[name] = compute_sha256(path)

    (args.output_dir / "CHECKSUMS.sha256").write_text(
        "\n".join(
            f"{sha}  {args.output_dir.name}/{Path(artifacts[name]).name}"
            for name, sha in checksums.items()
        )
        + "\n"
    )

    manifest = {
        "name": args.name,
        "voidface_version": voidface.__version__,
        "training_step": step,
        "config": {
            "epsilon": config.epsilon,
            "base_channels": config.base_channels,
            "num_stages": config.num_stages,
        },
        "artifacts": {name: str(path.name) for name, path in artifacts.items()},
        "checksums": checksums,
        "example_resolution": args.example_resolution,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    (args.output_dir / "README").write_text(
        f"""Voidface release bundle — {args.name}
Voidface {voidface.__version__} · training_step={step}

Artifacts:
{chr(10).join(f"  {name:12s} {path.name}" for name, path in artifacts.items())}

Verify integrity:
    sha256sum -c CHECKSUMS.sha256

Load in Python:
    import onnxruntime as ort
    session = ort.InferenceSession('{args.name}.onnx',
                                   providers=['CPUExecutionProvider'])

Load in browser:
    fetch('{args.name}.ort') -> ort.InferenceSession.create(...)

See Documentation/deployment/ for platform-specific loading code.
"""
    )

    log.info(
        "package.done",
        output_dir=str(args.output_dir),
        artifacts=list(artifacts),
    )
    return 0
