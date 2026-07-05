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

    dispatch = {
        "protect": _cmd_protect,
        "report": _cmd_report,
        "train": _cmd_train,
        "export": _cmd_export,
        "bench": _cmd_bench,
        "protect-video": _cmd_protect_video,
        "info": _cmd_info,
        "config-check": _cmd_config_check,
        "package": _cmd_package,
        "init": _cmd_init,
        "verify": _cmd_verify,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return handler(args)


def app() -> None:
    """Console-script wrapper. Exits the process with :func:`main`'s code."""
    raise SystemExit(main())


# --- argument parsing --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voidface",
        description="Voidface — adversarial face-region blindfold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_WORKFLOW_HELP,
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
    p_protect.add_argument(
        "--face-mask",
        action="store_true",
        help=(
            "Restrict the perturbation to the detected face region via a "
            "feathered mask (OpenCV Haar cascade). Cleaner backgrounds; no "
            "adversarial noise on smooth walls / sky. Only meaningful with "
            "--use-generator. Falls back to full-image perturbation when no "
            "face is detected."
        ),
    )
    p_protect.add_argument(
        "--semantic-warp",
        type=float,
        default=None,
        metavar="MAX_PIXELS",
        help=(
            "Compose a semantic geometric warp attack on top of the pixel "
            "PGD. Bounded to MAX_PIXELS of sub-pixel displacement. Applied "
            "via grid_sample; humans do not notice sub-2-px shifts but "
            "restorers regenerate a different face. Ignored when "
            "--use-generator is set without --refine-steps."
        ),
    )
    p_protect.add_argument(
        "--refine-steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "When used with --use-generator, initialize PGD from the "
            "generator's output (rather than uniform noise) and run N "
            "further PGD steps for higher-quality final output. Best of "
            "both worlds: single-forward-pass warm start plus N-step "
            "refinement. Requires the ensemble to be loaded (--targets)."
        ),
    )
    p_protect.add_argument(
        "--show-metrics",
        action="store_true",
        help=(
            "After protecting, run the real RetinaFace + ArcFace targets "
            "on both the original and the protected image and print the "
            "attack metrics inline (detection score before/after, ArcFace "
            "cosine displacement, PSNR, SSIM). Requires network access on "
            "first use for weight downloads."
        ),
    )
    p_protect.add_argument(
        "--output-json",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Emit a JSON sidecar file with metadata about the protection: "
            "voidface version, epsilon budget, flags used, PSNR/SSIM, "
            "generator checkpoint hash (if --use-generator), and seed. "
            "Users can audit later what protection was applied."
        ),
    )
    p_protect.add_argument(
        "--emit-delta",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Save the raw (protected - clean) delta as a torch .pt tensor. "
            "Research artifact — useful for analyzing the delta distribution "
            "or for reproducing the exact perturbation without re-running "
            "the tool. The delta plus the source image reconstructs the "
            "protected image bit-exactly."
        ),
    )
    p_protect.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Print wall-clock per phase: image load, generator forward, "
            "PGD steps if any, save. Useful for profiling."
        ),
    )
    p_protect.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Skip the human-readable summary block. Useful when driving "
            "voidface from a shell script that only cares about exit "
            "code + --output-json."
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
    p_train.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load the dataset and every target model and print a summary "
            "of the resolved configuration, but skip the actual training "
            "loop. Fast validation before committing to a long run."
        ),
    )
    p_train.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Load G's state_dict from CHECKPOINT before training starts. "
            "Continues optimization from where the last run left off. The "
            "checkpoint's VoidfaceConfig must match the config that would "
            "be built from the training TOML (architecture-compatible)."
        ),
    )
    p_train.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override [experiment].seed from the TOML for this run. Useful "
            "for A/B testing hyperparameters with the same data + config "
            "but different RNG."
        ),
    )

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
    p_bench.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Save the protected version of each test image to DIR alongside "
            "the metrics. Useful for visual A/B against the source images."
        ),
    )
    p_bench.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Cap the test set at N images. Useful for CI smoke checks on a "
            "full FFHQ directory or for fast iteration during tuning. "
            "0 (default) processes every image."
        ),
    )
    p_bench.add_argument(
        "--targets",
        type=str,
        default="detector,recognizer",
        help=(
            "Comma-separated target subset for bench. Options: 'detector', "
            "'recognizer'. Default 'detector,recognizer' runs both. Use a "
            "subset for fast per-family measurement."
        ),
    )
    p_bench.add_argument(
        "--baseline",
        type=Path,
        default=None,
        metavar="JSON",
        help=(
            "Load a previous bench JSON and print a delta table. Exits with "
            "code 1 if the new checkpoint regresses on ANY aggregate "
            "metric (detection ASR must not fall; identity cos+1 must not "
            "rise; PSNR/SSIM must not fall)."
        ),
    )
    p_bench.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Enforce the release ship gate defined in "
            "Documentation/process/release.md: detection ASR >= 0.60, "
            "identity cos+1 <= 0.20, PSNR mean >= 30 dB, SSIM mean >= "
            "0.92. Exits with code 3 if any threshold fails. Overridable "
            "per-metric via --strict-detection-asr / "
            "--strict-identity-cos / --strict-psnr / --strict-ssim."
        ),
    )
    p_bench.add_argument(
        "--strict-detection-asr",
        type=float,
        default=0.60,
        help="Ship-gate threshold for detection ASR (default 0.60).",
    )
    p_bench.add_argument(
        "--strict-identity-cos",
        type=float,
        default=0.20,
        help="Ship-gate threshold for identity cos+1 upper bound (default 0.20).",
    )
    p_bench.add_argument(
        "--strict-psnr",
        type=float,
        default=30.0,
        help="Ship-gate threshold for mean PSNR in dB (default 30.0).",
    )
    p_bench.add_argument(
        "--strict-ssim",
        type=float,
        default=0.92,
        help="Ship-gate threshold for mean SSIM (default 0.92).",
    )

    p_pv = sub.add_parser(
        "protect-video",
        help="Protect a video file by applying the generator to every frame.",
    )
    p_pv.add_argument("input", type=Path, help="Input video path.")
    p_pv.add_argument("output", type=Path, help="Output video path.")
    p_pv.add_argument(
        "--use-generator",
        type=Path,
        required=True,
        metavar="CHECKPOINT",
        help="Path to a trained .pt checkpoint. Video mode requires this.",
    )
    p_pv.add_argument(
        "--epsilon", type=int, default=12, help="L-inf budget as N/255 (default 12)."
    )
    p_pv.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: 'auto', 'cpu', 'cuda', 'mps' (default auto).",
    )
    p_pv.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC output codec (default 'mp4v'; 'avc1' if your OpenCV supports it).",
    )
    p_pv.add_argument(
        "--temporal-blend",
        type=float,
        default=0.7,
        metavar="ALPHA",
        help=(
            "Blend factor for temporal coherence (Farnebäck optical flow "
            "warping of the prior frame's delta). 0.0 disables and the "
            "video is per-frame independent (boiling texture). 1.0 uses "
            "only the warped prior. Default 0.7 — heavily weighted to the "
            "warped prior with a fresh G contribution to track scene changes."
        ),
    )
    p_pv.add_argument(
        "--face-mask",
        action="store_true",
        help=(
            "Restrict the perturbation to the detected face region on each "
            "frame via a feathered mask (OpenCV Haar cascade). Cleaner "
            "backgrounds; no adversarial noise on smooth walls or sky. "
            "Combines with --temporal-blend so both the fresh delta and the "
            "flow-warped delta are masked."
        ),
    )

    p_info = sub.add_parser(
        "info",
        help="Print metadata about a checkpoint (config, params, training step).",
    )
    p_info.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint.")
    p_info.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary.",
    )
    p_info.add_argument(
        "--diff",
        type=Path,
        default=None,
        metavar="OTHER_CHECKPOINT",
        help=(
            "Additional checkpoint to compare against. Prints a side-by-side "
            "table of the two checkpoints' training steps, param counts, "
            "file sizes, and configs. Useful for verifying two runs used "
            "the same hyperparameters."
        ),
    )

    p_cc = sub.add_parser(
        "config-check",
        help="Validate a training TOML config without running training.",
    )
    p_cc.add_argument("config", type=Path, help="Path to a training TOML.")

    p_init = sub.add_parser(
        "init",
        help="Write a starter training TOML for a common scenario.",
    )
    p_init.add_argument(
        "preset",
        choices=["smoke", "full", "detector-only", "recognizer-only"],
        help=(
            "Preset shape. 'smoke' = fast local check with no external "
            "weights. 'full' = the R5.5 reference (all targets + full "
            "restorer mix + normalize_per_target). 'detector-only' and "
            "'recognizer-only' are single-target experiments."
        ),
    )
    p_init.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to write the TOML (default: stdout).",
    )

    p_verify = sub.add_parser(
        "verify",
        help="Verify a release bundle's artifacts against its CHECKSUMS.sha256.",
    )
    p_verify.add_argument(
        "bundle_dir",
        type=Path,
        help="Release directory produced by `voidface package`.",
    )

    p_pkg = sub.add_parser(
        "package",
        help="Bundle a full release into a directory (ONNX + int8 + ORT + checksums).",
    )
    p_pkg.add_argument("checkpoint", type=Path, help="Trained checkpoint.")
    p_pkg.add_argument("output_dir", type=Path, help="Output release directory.")
    p_pkg.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory of images for static-quant calibration (recommended).",
    )
    p_pkg.add_argument(
        "--coreml",
        action="store_true",
        help="Also emit CoreML .mlpackage (Apple Silicon only).",
    )
    p_pkg.add_argument(
        "--example-resolution",
        type=int,
        default=512,
        help="Resolution for the ONNX tracing example (default 512).",
    )
    p_pkg.add_argument(
        "--name",
        type=str,
        default="voidface",
        help="Model name used for file stems (default 'voidface').",
    )

    return parser


# --- commands ----------------------------------------------------------------


def _cmd_protect(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.protect."""
    from voidface_cli.commands import protect as _protect_cmd

    return _protect_cmd.run(args)


def _cmd_report(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.report."""
    from voidface_cli.commands import report as _report_cmd

    return _report_cmd.run(args)


# All subcommand handlers live under voidface_cli.commands.*; the
# _cmd_* stubs above are thin delegators that keep argparse's
# `set_defaults(func=...)` wiring simple. Shared helpers live in
# voidface_cli.common.


_WORKFLOW_HELP = """
End-to-end workflow
-------------------

Bootstrap:     voidface init full -o cfg.toml           # or 'smoke' for a fast check
Train:         voidface train cfg.toml
Or use ref:    voidface train samples/configs/train_full.toml
Validate:      voidface bench runs/step-050000.pt path/to/test/ \\
                   --json bench.json --limit 200
Compare A/B:   voidface bench runs/step-050000.pt path/to/test/ \\
                   --baseline previous.json
Ship-bundle:   voidface package runs/step-050000.pt release-v1/ \\
                   --calibration-dir cal/ --coreml
Verify a       voidface verify release-v1/
downloaded
bundle:
Export à la    voidface export runs/step-050000.pt out/voidface.onnx \\
    carte:         --quantize int8 --quantize-static-dir cal/ --coreml --ort

Deploy one image (fast path):
    voidface protect user.jpg --use-generator runs/step-050000.pt --face-mask

Deploy one image (highest quality):
    voidface protect user.jpg --use-generator runs/step-050000.pt --refine-steps 20

Deploy a folder:
    voidface protect folder/ --output-dir out/ --use-generator runs/step-050000.pt

Deploy a video:
    voidface protect-video clip.mp4 out.mp4 \\
        --use-generator runs/step-050000.pt --temporal-blend 0.7 --face-mask

Inspect a checkpoint:      voidface info runs/step-050000.pt
Validate a config:         voidface config-check samples/configs/train_full.toml

Documentation/status.md is the authoritative "what ships today" reference.
"""


def _cmd_verify(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.verify."""
    from voidface_cli.commands import verify as _verify_cmd

    return _verify_cmd.run(args)


def _cmd_init(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.init."""
    from voidface_cli.commands import init as _init_cmd

    return _init_cmd.run(args)


def _cmd_package(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.package."""
    from voidface_cli.commands import package as _package_cmd

    return _package_cmd.run(args)


def _cmd_config_check(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.config_check."""
    from voidface_cli.commands import config_check as _cc_cmd

    return _cc_cmd.run(args)


def _cmd_info(args: argparse.Namespace) -> int:
    """Print a checkpoint's metadata (extracted to voidface_cli.commands.info)."""
    from voidface_cli.commands import info as _info_cmd

    return _info_cmd.run(args)


def _cmd_protect_video(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.protect_video."""
    from voidface_cli.commands import protect_video as _pv_cmd

    return _pv_cmd.run(args)


def _cmd_bench(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.bench."""
    from voidface_cli.commands import bench as _bench_cmd

    return _bench_cmd.run(args)


def _cmd_export(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.export."""
    from voidface_cli.commands import export as _export_cmd

    return _export_cmd.run(args)


def _cmd_train(args: argparse.Namespace) -> int:
    """Extracted to voidface_cli.commands.train."""
    from voidface_cli.commands import train as _train_cmd

    return _train_cmd.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
