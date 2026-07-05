# Training the generator G — tuning guide

This is the operational playbook for whoever runs the real R5.5
training. It captures the tuning intuitions that inform the default
values in `samples/configs/train_full.toml` and lists the failure
modes we know how to diagnose.

For the architecture and code layout, see
`Documentation/models/generator.md`. For the shipped subsystem
status, see `status.md`.

---

## The composite loss, at inference

The training objective is:

    L(G, x, δ) = Σᵢ  λᵢ · L_target_i( target_i( restorer( x + δ ) ) )
              + λ_lpips     · LPIPS( x, x + δ )
              - λ_blvl_lpips · LPIPS( restorer(x), restorer(x + δ) )   [if restorer ≠ identity]
              + λ_tv        · TV( δ )

The signs matter:

- `L_target_i` and `LPIPS(x, x+δ)` are MINIMIZED (invisibility + attack
  success on each target).
- The bilevel LPIPS term is MAXIMIZED — enters the loss with a
  negative sign — because we WANT the restored perturbed image to
  look different from the restored clean image (that's what "the
  attack survived restoration" looks like).

---

## Target weights

Defaults (in `train_full.toml`) are:

    detector   0.15
    scrfd      0.15
    yunet      0.10
    mtcnn      0.10
    arcface    0.50
    magface    0.25
    adaface    0.25
    sd15_vae   0.30
    sdxl_vae   0.25
    flux_vae   0.20
    openclip_h 0.20
    siglip_l   0.15
    dinov2_l   0.15
    gfpgan     0.40   (restorer, sampled per step)
    codeformer 0.35   (restorer, sampled per step)
    realesrgan 0.25   (restorer, sampled per step)

Rules of thumb:

- **ArcFace dominates the recognizer family** because every real
  face-swap pipeline uses it. Keep it at ≈0.5 of the recognizer
  budget.
- **RetinaFace is the choke point** for detection. Prefer it over the
  smaller detectors — no restorer helps the attacker if there's no
  face-present signal at all.
- **SD 1.5 VAE > SDXL VAE > Flux VAE** by weight. Cross-family
  transfer is under 40%, so treat each as its own subsystem.
- **OpenCLIP small (ViT-B/32) at 0.10** as a Phase-R4 bootstrap.
  When the R5 ViT-L/14 target lands, bump ViT-B/32 down and give
  ViT-L most of the CLIP budget.

Numerical scaling: VAE losses are on a `~50–200` scale where face
detectors run `~0.01–0.1`. Rescale before combining, or the VAE term
dominates. The training loop normalizes per-family in R5.5 (a
follow-up commit adds a `LossWeights.per_family_normalize` flag).

---

## Perceptual budget

Defaults:

    lpips        0.10   (invisibility constraint)
    total_var    0.01   (JPEG-survival smoothing)
    bilevel_lpips 0.05  (only active with non-identity restorer)

Symptoms:

- `lpips` too high → G refuses to attack. Slack the constraint by
  halving. Symptom in bench: PSNR very high (>40 dB) but ArcFace
  cosine barely below +1.
- `lpips` too low → outputs visibly noisy at the perturbation region.
  Bench SSIM drops below 0.90.
- `bilevel_lpips` too high → training oscillates. The term
  is unbounded (LPIPS has no upper limit in principle), and pushing
  it hard turns the loss into a moving target.

---

## EOT

Defaults:

    samples          4
    jpeg_qualities   [40, 55, 70, 85, 95]
    resize_factors   [0.5, 0.75, 1.0, 1.5, 2.0]
    gaussian_sigma   [0.0, 0.5, 1.0, 1.5]
    screenshot_prob  0.15
    webp_prob        0.10

Larger `samples` reduces variance in the gradient at the cost of
step time. `samples=8` is the ceiling before wall-clock per step
gets uncomfortable on a single A100.

The default `resize_factors` set matches what real social-media
pipelines do to uploads. Removing the extremes (0.5, 2.0) speeds
up steps ~15% at the cost of coverage of unusual re-sizes.

---

## Restorer mix

The R4 CEO-critic-recommended distribution:

    identity   0.10
    sd15-vae   0.30
    gfpgan     0.60

Rationale: GFPGAN is the actual attacker's finishing step, so most
of the training signal should live there. `sd15-vae` covers the
generic diffusion-purification strip. `identity` at 0.10 keeps the
generator useful against un-restored bulk scrapers.

Symptoms to watch:

- **GFPGAN OOM on MPS**: fp16 autocast on the CI configs is the
  first mitigation. Gradient checkpointing over the 8 StyleGAN2
  decoder blocks is the second — pre-decided per the R4 CEO critic
  but not yet wired in R4.5.2c.
- **Poor bilevel convergence**: the generator learns to defeat
  identity+sd15-vae but not gfpgan. Diagnose by disabling gfpgan
  and re-checking convergence. If the identity+sd15-vae leg still
  converges, the gfpgan target is the culprit — likely an alignment
  mismatch. Verify `data/align.py` receives well-formed landmarks
  by dumping the aligned crop to disk mid-training.

---

## Optimizer + schedule

Defaults for the shipping run:

    optimizer          Adam
    learning_rate      1e-4
    weight_decay       1e-6
    batch_size         16 (per-GPU)
    steps              300_000
    warmup_steps       1_000  (linear from 1e-6)
    cosine_decay       true    (down to 1e-6 at step 300k)

Symptoms:

- **Loss stalls at ~0.5-1.0 across all targets**: LR too low. Bump
  by 3× and retrain from checkpoint.
- **Total loss diverges (NaN)**: LR too high, gradient blow-up on
  the VAE term, or the bilevel LPIPS weight is too high. Halve the
  bilevel term first.
- **PSNR trends down while target losses trend down**: perceptual
  budget is under-weighted. Bump `lpips_weight` and retrain.

---

## Diagnosing per-step trainings

`train_generator` writes a `TrainStep` per step with per-target
breakdown. Log the full trace and grep for:

- `total = ...` — the primary training curve. Should trend
  monotonically down.
- `per_target = {...}` — check that every enabled target is
  contributing. A term stuck at ≈0 across steps is likely a
  wiring bug in the target's `EnsembleTarget` implementation.
- `restorer = ...` — verify the sampler is drawing the expected
  distribution. If you configured `gfpgan:0.6` and see 100%
  identity in the first 100 steps, seed is 0 and the RNG state
  hasn't advanced yet — verify `SamplerConfig(seed=...)`.

---

## Bench targets for R5.5 ship-vs-don't-ship gate

Values are the working consensus; adjust as we gather bench data.

    detection ASR                >= 0.60  (60% of face-visible inputs
                                          drop below 0.5 confidence
                                          after G)
    identity cos+1              <= 0.20  (equivalent to cosine <= -0.8
                                          against the ArcFace family)
    PSNR (mean)                 >= 30 dB
    SSIM (mean)                 >= 0.92

Run:

    voidface bench ckpt.pt path/to/FFHQ-test/ \
        --json bench.json \
        --detection-threshold 0.5

Any release-candidate checkpoint must meet all four thresholds
against a fresh, unseen test corpus (no images from training).

---

## Reproducibility

- Every training run pins the seed via `[experiment].seed`.
- The `[experiment].checkpoint_dir` becomes the run's artifact
  directory — never delete it until the bench numbers are archived.
- `train_generator` writes the full `VoidfaceConfig` inside the
  checkpoint payload. `voidface export` reads it back and
  reconstructs the generator with the same architecture — no
  architecture drift between train and deploy.
- `LossWeights`, `EotConfig`, and `SamplerConfig` are frozen
  dataclasses — hashable, safe to store alongside the checkpoint
  for later inspection.
