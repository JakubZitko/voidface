# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
"""Generate the six figures for Documentation/paper.md.

Run from the voidface repo root:

    uv run python Documentation/paper/figures/generate_figures.py

Outputs six 300-dpi PNGs into the same directory as this script.

The figures use the actual voidface code (iris_region_mask, FFHQ landmark
template) where possible, and clearly-labeled illustrative values where
voidface is not yet trained (projected numbers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

FIGDIR = Path(__file__).parent
FIGDIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# Small deliberate palette.
C_PRIMARY = "#2E5B9B"      # voidface blue
C_ACCENT = "#D96C4B"       # attention orange
C_BG = "#F5F1EA"           # warm cream
C_DIM = "#8A8A8A"          # muted grey
C_GOOD = "#3B7DD8"
C_BAD = "#C85454"
C_TEXT = "#1F2933"


def _rounded_box(ax, x, y, w, h, label, facecolor=C_PRIMARY, textcolor="white", fontsize=10):
    """Helper: draw a rounded filled box centered at (x, y) with (w, h)."""
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.5, edgecolor=C_TEXT, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            color=textcolor, fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2, label=None, color=C_TEXT):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color, mutation_scale=16),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label,
                ha="center", va="bottom", fontsize=8, color=color, style="italic")


def fig1_iris_mask():
    """The actual iris_region_mask output on the canonical FFHQ template."""
    from voidface.attacks.iris import iris_region_mask
    from voidface.data.align import FFHQ_LANDMARKS_512

    landmarks = torch.tensor(FFHQ_LANDMARKS_512, dtype=torch.float32).unsqueeze(0)
    mask = iris_region_mask(landmarks, height=512, width=512, radius_frac=0.028, softness_px=1.5)
    mask_np = mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left panel: mask alone
    im1 = axes[0].imshow(mask_np, cmap="hot", vmin=0.0, vmax=1.0)
    axes[0].set_title("iris_region_mask (radius_frac=0.028)", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="mask intensity")

    # Right panel: mask over blank face template with landmarks marked
    axes[1].imshow(np.full((512, 512, 3), 220, dtype=np.uint8))
    im2 = axes[1].imshow(mask_np, cmap="hot", alpha=0.75, vmin=0.0, vmax=1.0)
    for name, (x, y) in zip(
        ["L eye", "R eye", "nose", "L mouth", "R mouth"],
        FFHQ_LANDMARKS_512,
    ):
        axes[1].plot(x, y, marker="o", markersize=6, markerfacecolor="cyan",
                     markeredgecolor="white", markeredgewidth=1.5)
        offset_y = -14 if "eye" in name else 14
        axes[1].annotate(name, xy=(x, y), xytext=(x, y + offset_y),
                         ha="center", va="center", fontsize=8, color=C_TEXT,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                   edgecolor=C_DIM, alpha=0.85))
    axes[1].set_title("Iris mask over 5-point FFHQ landmark template", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle("Figure 1: Iris-region mask — high recognizer signal, low human perceptibility",
                 fontsize=12, y=1.02)

    coverage = float((mask_np > 0.5).mean())
    fig.text(0.5, -0.02,
             f"Coverage of soft-thresholded mask: {coverage*100:.3f}% of image area "
             f"— human iris fraction ≈ 0.03% of a face crop",
             ha="center", fontsize=9, color=C_DIM, style="italic")

    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_01_iris_mask.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig2_bilevel_loss_flow():
    """Block diagram of CompositeLoss decomposition + backprop path."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Central node
    _rounded_box(ax, 6.5, 3.0, 2.2, 0.9, "L_total\n(weighted sum)",
                 facecolor=C_PRIMARY, fontsize=11)

    # Six input terms (per-target losses + LPIPS + TV + bilevel LPIPS)
    inputs = [
        (1.5, 5.7, "LPIPS(clean, adv)\nperceptual constraint"),
        (1.5, 4.4, "Total Variation\nsmoothness prior"),
        (1.5, 3.0, "bilevel_lpips(\n  restorer(clean),\n  restorer(adv))\n[NEGATED]"),
        (1.5, 1.6, "per-target losses:\ndetector / recognizer /\nVAE / SDXL-VAE / CLIP"),
    ]
    for x, y, label in inputs:
        _rounded_box(ax, x, y, 2.4, 0.9 if "\n" not in label[10:] else 1.3,
                     label, facecolor="white", textcolor=C_TEXT, fontsize=8)
        _arrow(ax, 2.75, y, 5.4, 3.0)

    # Output: backprop into delta
    _rounded_box(ax, 11.0, 4.4, 2.4, 0.9, "backprop into δ\n(and warp field)",
                 facecolor=C_ACCENT, fontsize=9)
    _arrow(ax, 7.6, 3.0, 9.75, 4.2, label="∂L/∂δ")

    _rounded_box(ax, 11.0, 1.6, 2.4, 0.9,
                 "no-grad restore\nGFPGAN forward",
                 facecolor="white", textcolor=C_TEXT, fontsize=8)
    _arrow(ax, 7.6, 3.0, 9.75, 1.8, label="restorer(x+δ)")

    ax.set_title("Figure 2: CompositeLoss decomposition inside the PGD loop\n"
                 "(bilevel term negated so gradient pushes δ to survive restoration)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_02_bilevel_loss_flow.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig3_ensemble_targets():
    """Six ensemble targets grid + per-target loss functions + weight labels."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    targets = [
        ("RetinaFace-R50", "detection\nsuppression", "detectors", "109 MB", 0.35, 1.6, 4.5),
        ("ArcFace IResNet-100", "identity cosine\ndissimilarity", "recognizers", "249 MB", 0.40, 4.4, 4.5),
        ("SD 1.5 VAE", "gray-latent loss\n(encoder attack)", "diffusion", "334 MB", 0.20, 7.2, 4.5),
        ("SDXL VAE", "gray-latent loss", "diffusion", "334 MB", 0.15, 10.0, 4.5),
        ("CLIP ViT-B/32", "embedding\ndissimilarity", "vision-language", "150 MB", 0.10, 1.6, 1.5),
        ("GFPGAN v1.4", "bilevel LPIPS\n(in inner loop)", "restorers", "348 MB", "novel", 4.4, 1.5),
    ]

    for name, loss, family, size, weight, x, y in targets:
        _rounded_box(ax, x, y + 0.4, 2.2, 0.5, name,
                     facecolor=C_PRIMARY, fontsize=9)
        _rounded_box(ax, x, y - 0.2, 2.2, 0.6, loss,
                     facecolor="white", textcolor=C_TEXT, fontsize=8)
        wlabel = f"w = {weight}" if isinstance(weight, float) else weight
        ax.text(x, y - 0.75, f"{family} · {size} · {wlabel}",
                ha="center", fontsize=7.5, color=C_DIM, style="italic")

    # Restorer sampler cluster
    _rounded_box(ax, 8.5, 1.5, 3.0, 1.4,
                 "RestorerSampler\n{identity, sd15-vae-roundtrip, gfpgan-v1.4}\ndrawn per PGD step",
                 facecolor=C_ACCENT, fontsize=8.5)

    ax.set_title("Figure 3: Six ensemble surrogate targets (weights renormalised across selected subset)",
                 fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_03_ensemble_targets.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig4_deploy_pipeline():
    """One trained .pt → four deploy artifacts → four target platforms."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_facecolor(C_BG)

    # Source
    _rounded_box(ax, 1.5, 2.75, 2.4, 1.2,
                 "voidface-g.pt\n(~5.35 M params, fp32)",
                 facecolor=C_PRIMARY, fontsize=9)

    # Four artifacts
    artifacts = [
        (6.0, 4.7, "voidface.onnx\n(fp32, ~21 MB)", C_PRIMARY),
        (6.0, 3.4, "voidface.int8.onnx\n(int8 dyn, ~5.5 MB)", C_PRIMARY),
        (6.0, 2.1, "voidface.static-int8.onnx\n(calibrated, ~5.5 MB)", C_PRIMARY),
        (6.0, 0.8, "voidface.mlpackage\n(CoreML, ~11 MB)", C_PRIMARY),
    ]
    for x, y, label, col in artifacts:
        _rounded_box(ax, x, y, 3.0, 0.9, label, facecolor=col, fontsize=9)
        _arrow(ax, 2.75, 2.75, 4.5, y)

    # Also produce ORT-Web
    _rounded_box(ax, 6.0, -0.5, 3.0, 0.9, "voidface.ort\n(ORT-Web, ~5.6 MB)", facecolor=C_PRIMARY, fontsize=9)
    # Draw arrow from static-int8 to ort
    ax.annotate("", xy=(6.0, -0.05), xytext=(6.0, 1.65),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C_DIM))

    # Target platforms
    platforms = [
        (11.0, 4.7, "Python /\nonnxruntime CPU"),
        (11.0, 3.4, "int8 CPU\n(Linux/mac/win)"),
        (11.0, 2.1, "int8 CPU\n(calibration = best)"),
        (11.0, 0.8, "iOS / macOS ANE"),
        (11.0, -0.5, "Browser\n(WebGPU + WASM)"),
    ]
    for i, (x, y, label) in enumerate(platforms):
        _rounded_box(ax, x, y, 2.4, 0.9, label, facecolor="white",
                     textcolor=C_TEXT, fontsize=9)
        art_y = artifacts[i][1] if i < 4 else -0.5
        _arrow(ax, 7.55, art_y, 9.75, y)

    ax.set_title("Figure 4: One trained checkpoint fans out to four deploy formats "
                 "across every consumer platform",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_04_deploy_pipeline.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig5_ship_gate():
    """Four ship-gate metrics with pass/fail threshold lines."""
    metrics = [
        ("Detection ASR", 0.60, 0.72, 1.0, "≥"),
        ("Identity cos+1", 0.20, 0.15, 2.0, "≤"),   # inverted: lower is better
        ("PSNR (dB)", 30.0, 31.5, 45.0, "≥"),
        ("SSIM", 0.92, 0.935, 1.0, "≥"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.patch.set_facecolor(C_BG)
    for ax, (name, thr, tgt, ymax, op) in zip(axes, metrics):
        ax.set_facecolor(C_BG)
        ax.bar([0], [tgt], color=C_GOOD, width=0.55, edgecolor=C_TEXT, linewidth=1)
        ax.axhline(thr, color=C_BAD, linestyle="--", linewidth=1.6,
                   label=f"gate: {op} {thr}")
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        ax.text(0, tgt + ymax * 0.03, f"target\n{tgt}",
                ha="center", fontsize=9, color=C_TEXT)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Figure 5: Ship-gate thresholds enforced by `voidface bench --strict`\n"
                 "(target values illustrative — voidface not yet trained; exit code 3 on any failure)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_05_ship_gate.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig6_projected_vs_baselines():
    """Illustrative bar chart: voidface projected vs baselines under GFPGAN-restored pipeline."""
    tools = ["No protection", "Fawkes\n(2020)", "PhotoGuard\n(2023)", "Glaze\n(2023)", "Voidface\n(PROJECTED)"]
    # Illustrative — sourced roughly from Radiya-Dixit & Tramer 2022 + Honig et al. 2025 findings
    # that prior cloaks drop to near-baseline under restoration.
    detection_asr = [0.02, 0.05, 0.12, 0.10, 0.60]
    identity_dist = [0.03, 0.08, 0.15, 0.14, 0.80]

    x = np.arange(len(tools))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    colors_a = ["#7d7d7d", C_BAD, C_BAD, C_BAD, C_GOOD]
    colors_b = ["#a5a5a5", "#e39898", "#e39898", "#e39898", "#7fa6e0"]

    bars1 = ax.bar(x - width/2, detection_asr, width, color=colors_a,
                   edgecolor=C_TEXT, linewidth=1, label="Detection ASR")
    bars2 = ax.bar(x + width/2, identity_dist, width, color=colors_b,
                   edgecolor=C_TEXT, linewidth=1, label="Identity dissimilarity (proxy)")

    ax.axhline(0.60, color=C_TEXT, linestyle=":", linewidth=1.2, alpha=0.6,
               label="Voidface ship-gate (Detection ASR ≥ 0.60)")

    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10)
    ax.set_ylabel("Metric value (higher = better protection)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for bar, val in zip(bars1, detection_asr):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                f"{val:.2f}", ha="center", fontsize=8.5, color=C_TEXT)
    for bar, val in zip(bars2, identity_dist):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                f"{val:.2f}", ha="center", fontsize=8.5, color=C_TEXT)

    fig.suptitle(
        "Figure 6: Detection ASR under a GFPGAN-restored face-swap pipeline\n"
        "(baselines from prior work; PROJECTED bar is Voidface's target, not a measurement)",
        fontsize=11, y=0.98,
    )
    fig.text(0.5, -0.02,
             "Baseline numbers are illustrative summaries of Radiya-Dixit & Tramer (ICLR 2022) "
             "and Honig et al. (ICLR 2025).\nVoidface's actual number will be published after "
             "the R5.5 training run; ship gate is 0.60.",
             ha="center", fontsize=8.5, color=C_DIM, style="italic")

    plt.tight_layout()
    plt.savefig(FIGDIR / "figure_06_projected_vs_baselines.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    print("Generating voidface paper figures...")
    fig1_iris_mask();               print("  ✓ figure_01_iris_mask.png")
    fig2_bilevel_loss_flow();       print("  ✓ figure_02_bilevel_loss_flow.png")
    fig3_ensemble_targets();        print("  ✓ figure_03_ensemble_targets.png")
    fig4_deploy_pipeline();         print("  ✓ figure_04_deploy_pipeline.png")
    fig5_ship_gate();               print("  ✓ figure_05_ship_gate.png")
    fig6_projected_vs_baselines();  print("  ✓ figure_06_projected_vs_baselines.png")
    print(f"Done. Figures at {FIGDIR}")


if __name__ == "__main__":
    main()
