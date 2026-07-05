# Composing multiple attacks

Voidface exposes several orthogonal attack mechanisms. This page
describes how they compose, in what order, and what the resulting
gradient signal looks like.

## The full pipeline

    x    ─►    delta_pixel = PGD(pixel attack, R7 signed-gradient MI-FGSM)
             +
             delta_semantic = SemanticWarp field (R7.1, if enabled)
             │
             ▼
    x + delta_pixel  ─►  apply_semantic_warp(x + delta_pixel, delta_semantic)
             │
             ▼
    restorer_sampled = RestorerSampler.sample()   (R3+)
             │
             ▼
    perturbed_restored = restorer_sampled(perturbed)
             │
             ▼
    EOT wrap (resize + blur + differentiable JPEG, R6.14)
             │
             ▼
    ensemble targets compute per-target losses
             │
             ▼
    CompositeLoss combines with LPIPS + bilevel LPIPS + TV
             │
             ▼
    backprop into delta_pixel AND delta_semantic

The `face_mask` post-op runs AFTER the training-time optimization
completes, at deploy time. It multiplies the final delta by a
soft face-region mask.

## What each layer contributes

- **Pixel PGD (R1 baseline).** Signed-gradient descent on the raw RGB
  delta bounded to L∞ = epsilon/255. Attacks whatever ensemble the
  composite loss selects. The workhorse — every other layer is a
  quality multiplier on top.
- **Semantic warp (R7.1 / R7.3).** Sub-pixel geometric displacement
  optimized jointly with the pixel delta. Bounded to
  ~2 pixels of displacement, Gaussian-smoothed so the warp field
  stays C¹. Survives face restoration better than the pixel delta
  because the restorer regenerates from a geometry-preserving prior.
- **Restorer sampler (R3, R4.5.2b).** Per-step draws from
  ``{identity, sd15-vae-roundtrip, gfpgan-v1.4}``. Everything
  downstream sees ``restorer(perturbed)`` instead of the raw
  perturbation, so the optimized delta survives the actual attacker
  pipeline (rather than a pre-restorer proxy).
- **EOT (R6.14).** Averaging over ``{resize factors, blur sigmas,
  JPEG qualities}`` inside the PGD step. Each PGD step samples ``k``
  transforms and averages the loss; the final delta is robust to
  the whole distribution, not just the raw input pixels.
- **Bilevel LPIPS (R4.5.2c-1).** Enters the loss with a NEGATIVE
  sign. Maximizes ``LPIPS(restorer(clean), restorer(adversarial))``
  — the CEO-critic proxy for "the delta survived restoration."
- **Face mask (R6.10, R7.21).** Deploy-time only. Multiplies the
  learned delta by a soft face-region mask so smooth backgrounds
  (walls, sky) don't show adversarial noise.

## Combinations to know

- **Cheapest useful attack.** Pixel PGD alone. `voidface protect
  img.jpg --steps 100`. ~2 minutes on CPU.
- **Deploy-quality attack.** Trained G forward pass.
  `voidface protect img.jpg --use-generator ckpt.pt`.
  ~500 ms with a real trained checkpoint.
- **Deploy + refine.** Warm-start PGD from G's output.
  `voidface protect --use-generator ckpt.pt --refine-steps 20`.
  ~15 seconds; strictly better than G alone.
- **Research-quality attack.** Everything on:
  `voidface protect img.jpg --steps 300 --restorers
  identity:0.1,sd15-vae:0.3,gfpgan:0.6 --semantic-warp 2.0
  --face-mask`.
- **Video with temporal coherence.** `voidface protect-video
  clip.mp4 out.mp4 --use-generator ckpt.pt --temporal-blend 0.7
  --face-mask`. Farnebäck flow warps the previous frame's delta
  forward so consecutive frames don't boil.

## What does NOT compose

- **`--use-generator` with `--semantic-warp`.** Semantic warp is a
  training-time gradient-descent field. G at deploy has no field to
  optimize; the flag is ignored on the pure `--use-generator` path.
  It IS respected when `--refine-steps > 0` since that path returns
  to PGD.
- **`--use-generator` alone with `--refine-steps > 0` but no
  `--targets`.** The refinement step requires the ensemble targets
  to be loaded. Currently the CLI accepts this and silently uses
  the default target set; a follow-up commit will make refine-steps
  imply a default targets subset.
- **`--face-mask` on batch mode without `--use-generator`.** Batch
  mode requires `--use-generator`; per-image PGD on a folder is
  impractical (~2 min per image).

## Where the code lives

- Pixel PGD: `src/voidface/core/pgd.py::run_pgd`
- Semantic warp: `src/voidface/attacks/semantic.py`
- Restorer sampler: `src/voidface/models/restorers/sampler.py`
- EOT: `src/voidface/core/eot.py`, `src/voidface/core/jpeg.py`
- Bilevel LPIPS: `src/voidface/core/loss.py::CompositeLoss.compute`
- Face mask: `src/voidface/util/facemask.py`
- Video temporal blend: `src/voidface/util/flow.py`

Each has its own subsystem doc; this file is the map that connects
them.
