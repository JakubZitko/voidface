# Voidface — current state (session 2026-07-05)

This file is the single-page source of truth for what is in the code
today versus what is still on the roadmap. Read this first when
starting a new session.

The per-subsystem docs describe design intent; the phase notes below
describe what actually ships.

---

## What ships today

Ensemble targets, all real, all differentiable, all wired into
:class:`voidface.core.loss.CompositeLoss`:

| Subsystem       | Concrete class                                               | Backing weights                                      |
| --------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| detector        | `voidface.models.detectors.retinaface.RetinaFace`            | biubug6/RetinaFace-R50 via `yakhyo/retinaface-pytorch` (109 MB) |
| recognizer      | `voidface.models.recognizers.arcface.Arcface`                | InsightFace IResNet-100 via `minchul/cvlface_arcface_ir101_webface4m` (249 MB) |
| vae             | `voidface.models.vaes.sd15.Sd15Vae`                          | `stabilityai/sd-vae-ft-mse` (334 MB)                 |
| sdxl-vae        | `voidface.models.vaes.sdxl.SdxlVae`                          | `madebyollin/sdxl-vae-fp16-fix` (334 MB)             |
| openclip        | `voidface.models.clip.openclip.OpenClip`                     | `openai/clip-vit-base-patch32` (150 MB)              |

Restorers (bilevel loop targets), all real, sampled per-step via
:class:`voidface.models.restorers.sampler.RestorerSampler`:

| Restorer          | Class                                                          | Notes                                                              |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| identity          | `voidface.models.restorers.identity.IdentityRestorer`          | Pass-through. Keeps the generator's un-restored performance.       |
| sd15-vae-roundtrip | `voidface.models.restorers.sd_vae.Sd15VaeRestorer`             | `decode(encode(x))` through SD 1.5 VAE.                            |
| gfpgan-v1.4       | `voidface.models.restorers.gfpgan.GfpganRestorer`              | Full RetinaFace-align → GFPGAN-v1.4 → unalign-paste pipeline. Weights via `TencentARC/GFPGANv1` (348 MB) with SHA-256 gate. |

Core training pieces:

- :class:`voidface.core.loss.CompositeLoss` — weighted target losses,
  LPIPS perceptual constraint on the clean/adversarial pair, and the
  bilevel LPIPS on the restored pair.
- :class:`voidface.core.eot.EotSampler` — resize + Gaussian blur EOT.
- :func:`voidface.core.pgd.run_pgd` — reference per-image PGD kernel.
- :func:`voidface.core.train.train_generator` — trains the shipped
  generator against the same composite loss.
- :class:`voidface.generator.architecture.Voidface` — the shipped
  model, ~5.35 M float32 params, ~5.5 MB int8.

Data pipeline:

- :func:`voidface.data.align.align_faces` and
  :func:`voidface.data.align.unalign_paste` — the differentiable
  5-point similarity transform the GFPGAN restorer relies on.
- :class:`voidface.data.datasets.FolderImageDataset` — folder-of-images
  dataset for the training loop.

Export:

- :func:`voidface.export.onnx.export_generator_to_onnx` — fp32,
  dynamic axes.
- :func:`voidface.export.quantize.quantize_onnx_generator` — int8
  dynamic quant.
- :func:`voidface.export.coreml.export_generator_to_coreml` — CoreML
  mlpackage (Apple Silicon only).
- :func:`voidface.export.ort.convert_onnx_to_ort` — ORT-Web format for
  the browser demo.

CLI (9 subcommands as of R7.10):

- `voidface protect <image>` — per-image PGD OR --use-generator fast
  path OR --face-mask restricted OR batch mode via `<dir> --output-dir`.
  --semantic-warp composes the geometric attack on top; --refine-steps
  N warm-starts PGD from G's output for hybrid quality.
- `voidface protect-video <in.mp4> <out.mp4>` — per-frame G with
  --temporal-blend Farnebäck-flow warping and --face-mask.
- `voidface report <original> <protected>` — PSNR / SSIM / L-inf.
- `voidface train <config.toml>` — full training run; TOML shape covers
  [experiment] / [data] / [optim] / [loss] / [loss.perceptual] /
  [eot] / [targets.*] / [restorers].
- `voidface bench <ckpt> <images/>` — release-gate metrics with
  --json (machine-readable), --out-dir (save protected images),
  --limit N (cap), --baseline JSON (A/B against a previous run).
- `voidface export <ckpt> <out.onnx>` — --quantize int8|uint8 (dynamic),
  --quantize-static-dir CAL (static, verified parity), --coreml, --ort.
- `voidface package <ckpt> <out-dir/>` — one command for a full
  release bundle (ONNX + int8 + static-int8 + ORT + optional CoreML +
  CHECKSUMS.sha256 + MANIFEST.json + README).
- `voidface info <ckpt>` — checkpoint metadata (config, params,
  training step) with --json.
- `voidface config-check <cfg.toml>` — validate a training config
  without waiting for weight downloads to surface a typo.

Tests: **147 unit + integration tests passing**, 1 CoreML test
correctly skipped on non-Apple-Silicon.

---

## What does NOT ship yet

- **Trained weight release.** The training system works; a real
  training run against a real face corpus on cloud GPUs is R5.5 and
  produces the checkpoint we distribute.
- **MPS gradient checkpointing for GFPGAN.** R4 CEO critic requested
  it as insurance against MPS OOM at 512×512. Skipped this session
  because Intel Mac has no MPS to validate against.
- **Desktop app** (`tools/desktop/`, Tauri). Placeholder README only.
- **Browser demo** (`tools/web/`). Real Vite + TypeScript +
  onnxruntime-web scaffold shipped in R6.1. Awaits a shipped `.ort`
  from R5.5. Deployment story documented in
  `Documentation/deployment/browser.md`.
- **iOS wrapper**. Not yet scoped.
- **Documented API stability.** Everything under `voidface.*` is
  pre-1.0; interfaces may change until R5.5 lands.

**Attacks (research):**

- Pixel PGD — R1, shipped.
- Semantic geometric warp — R7.1 shipped standalone; R7.3 composed
  into `run_pgd`. Sub-2-px displacement bounded, Gaussian-smoothed,
  grid_sample-applied.
- Iris attack — Documentation/attacks/iris.md scoped but not
  implemented; requires landmark detection.

**EOT (transform distribution during training):**

- Bilinear resize — shipped.
- Gaussian blur — shipped.
- Differentiable JPEG (Reich et al 2024) — R6.14 shipped. Standard
  luma/chroma quantization tables, STE round on quantize step.
  Wired into the training TOML via `[eot].jpeg_qualities`.

**Loss engineering:**

- LPIPS perceptual constraint on `(clean, adversarial)` — R2, shipped.
- Bilevel LPIPS on `(restorer(clean), restorer(adversarial))` — R4.5.2c-1
  shipped. Maximized (negative weight). Wired via `[loss].bilevel_lpips`.
- Per-target normalization via EMA — R5.5-follow shipped. Off by
  default; enabled per training run via `[loss].normalize_per_target`.
  Balances VAE terms (~50) against detector terms (~0.05).
- Total variation smoothness prior — R2 shipped.

---

## Reproducing the training loop

    uv run voidface train samples/configs/train_r5_smoke.toml \
        --device cpu

Point `[data].directory` in the TOML at your own face-crop folder to
train against real data. On a single GPU this is the invocation that
produces the shipped weights.

## Reproducing the export

    uv run voidface export runs/checkpoint.pt out/voidface.onnx \
        --example-resolution 512 \
        --quantize int8 \
        --coreml \
        --ort

Emits:

- `out/voidface.onnx` — fp32 portable
- `out/voidface.int8.onnx` — quantized
- `out/voidface.mlpackage/` — CoreML (Apple Silicon only)
- `out/voidface.ort` — ORT-Web fast startup

---

## Phase timeline

- R1  end-to-end PGD, stand-in models
- R2  Restorer abstraction, SD 1.5 VAE target, LPIPS
- R3  SD-VAE round-trip restorer, RestorerSampler, bilevel loop
- R4  Real ArcFace / RetinaFace / SDXL VAE / OpenCLIP / GFPGAN
  bilevel restorer, aligner, LICENSES/, dep unpinning
- R5.1 Generator architecture
- R5.2 G training loop; system converges
- R5.3 Dataset pipeline and `voidface train` CLI
- R5.4 ONNX + CoreML + ORT-Web + int8 quantization; `voidface export`
- R5.5 Real training run → shipped checkpoint (roadmap)
- R6+  Desktop / browser / iOS UIs (roadmap)
