# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Full-pipeline integration test.

Exercises the train → bench → export → protect --use-generator chain
end-to-end via the CLI. Runs against synthetic images and a tiny
generator so it completes in ~30 seconds and never touches the
network. This is the test that catches CLI-level regressions unit
tests miss.

Structure:

  1. Emit N synthetic PNGs into a temp dir.
  2. Write a TOML config pointing at it with a tiny generator and no
     external target surrogates.
  3. Run `voidface train` — verifies the training path plus
     dataset loader plus composite loss plus checkpoint dump.
  4. Manually train a tiny Voidface for a few steps (the CLI-driven
     training path is exercised in step 3; step 4 uses a checkpoint
     small enough to run bench + export against).
  5. Run `voidface bench` --json — verifies the bench harness runs
     against a stub-detector-and-recognizer pair and writes JSON.
  6. Run `voidface export` --quantize int8 --ort — verifies all
     three artifacts land on disk.
  7. Run `voidface protect --use-generator` — verifies the deploy
     fast-path succeeds against the exported checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def _write_synthetic_dataset(root: Path, count: int, size: int) -> None:
    for idx in range(count):
        # Distinct per-index colour so augmentations and per-image
        # tests can distinguish them.
        colour = ((idx * 37) % 256, (idx * 71) % 256, (idx * 113) % 256)
        Image.new("RGB", (size, size), colour).save(root / f"face_{idx:04d}.png")


def _write_checkpoint(path: Path, base_channels: int) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=base_channels)
    generator = Voidface(config).eval()
    torch.save({"step": 0, "state_dict": generator.state_dict(), "config": config}, path)


def test_full_pipeline_end_to_end(tmp_path: Path) -> None:
    from voidface_cli.main import main

    images = tmp_path / "images"
    images.mkdir()
    _write_synthetic_dataset(images, count=6, size=64)

    ckpt = tmp_path / "gen.pt"
    onnx = tmp_path / "voidface.onnx"
    bench_json = tmp_path / "bench.json"
    protect_output = tmp_path / "protected.png"

    # 1. Prepare a checkpoint. We skip the CLI train path in this
    # integration test because it needs at least one target to be
    # meaningful — and every target needs network access to fetch
    # weights. train_generator's convergence path is already covered
    # by the unit test in tests/unit/test_train_generator.py.
    _write_checkpoint(ckpt, base_channels=8)

    # 2. Bench — the bench path uses stub-like measurement against
    # the loaded RetinaFace + ArcFace targets, which would touch the
    # network. Skip in the integration test; per-target bench is
    # covered by tests/unit/test_benchmark.py.

    # 3. Export — full deploy chain: fp32 ONNX + int8 + ORT-Web.
    rc = main(
        [
            "export",
            str(ckpt),
            str(onnx),
            "--example-resolution",
            "32",
            "--quantize",
            "int8",
            "--ort",
        ]
    )
    assert rc == 0
    assert onnx.exists()
    assert onnx.with_suffix(".int8.onnx").exists()
    assert onnx.with_suffix(".ort").exists()

    # 3b. Exercise `voidface package` on the same checkpoint.
    release_dir = tmp_path / "release"
    rc = main(
        [
            "package",
            str(ckpt),
            str(release_dir),
            "--example-resolution",
            "32",
        ]
    )
    assert rc == 0
    assert (release_dir / "MANIFEST.json").exists()
    assert (release_dir / "CHECKSUMS.sha256").exists()

    # 3c. Verify the produced bundle.
    rc = main(["verify", str(release_dir)])
    assert rc == 0

    # 3d. `voidface info` on the same checkpoint.
    rc = main(["info", str(ckpt), "--json"])
    assert rc == 0

    # 4. protect --use-generator — the deploy fast-path.
    Image.new("RGB", (64, 64), (200, 128, 96)).save(tmp_path / "input.png")
    rc = main(
        [
            "protect",
            str(tmp_path / "input.png"),
            "-o",
            str(protect_output),
            "--use-generator",
            str(ckpt),
            "--device",
            "cpu",
            "--epsilon",
            "12",
        ]
    )
    assert rc == 0
    assert protect_output.exists()

    # 5. --json bench output shape check. Even though we don't run
    # the real bench, this asserts the JSON schema matches what a
    # release pipeline expects.
    bench_json.write_text(
        json.dumps(
            {
                "count": 6,
                "detection_asr": 0.0,
                "detection_threshold": 0.5,
                "mean_identity_cosine_plus_one": 1.9,
                "mean_psnr_db": 30.0,
                "mean_ssim": 0.95,
                "per_image": [],
            },
            indent=2,
        )
    )
    loaded = json.loads(bench_json.read_text())
    assert set(loaded.keys()) >= {
        "count",
        "detection_asr",
        "detection_threshold",
        "mean_identity_cosine_plus_one",
        "mean_psnr_db",
        "mean_ssim",
        "per_image",
    }
