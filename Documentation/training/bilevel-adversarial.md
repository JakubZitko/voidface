# Bilevel adversarial training — restorer in the loop

This is the load-bearing design decision that separates Voidface from
PhotoGuard, Fawkes, Glaze, and Anti-DreamBooth. This document explains
the objective, the reason prior tools do not include it, and what breaks
if we omit it.

---

## The problem

Every off-the-shelf face-swap and nudify pipeline in 2026 runs a face
restorer (GFPGAN, CodeFormer, Real-ESRGAN, RestoreFormer++) as one of
its final steps. The restorer does not filter the input — it *regenerates*
face pixels from a StyleGAN2-style face prior trained on FFHQ.

An imperceptible adversarial perturbation lives in the RGB signal. The
restorer throws the RGB signal away and paints a fresh face. Everything
we optimized to break — the ArcFace embedding, the VAE latent, the CLIP
embedding — snaps back onto the natural face manifold.

Radiya-Dixit & Tramer (ICLR 2022) is the canonical result: Fawkes was
broken this way. Sang et al. and Honig et al. subsequently generalized
it. PhotoGuard is broken by the same mechanism, and Glaze is broken by a
purification variant of it.

Every prior tool computes its loss on the pre-restorer signal. We do not.

---

## The objective

    minimize   Σ  λ_i · L_i( pipeline_i( R( T( G(x) ) ) ) )
       G     i ∈ ensemble
                                                  ┬───┬───┬
                                                  │   │   │
                                                  │   │   └─ our generator
                                                  │   └───── EOT transform
                                                  │           (JPEG, resize, blur)
                                                  └───────── attacker's restorer

               + λ_percep · L_percep( G(x), x )

Where:

- `G(x)` is the generator output for source photo `x`. This is the
  image we ship to the user.
- `T(·)` is the expectation-over-transformation wrapper documented in
  `eot.md`. It simulates JPEG re-encoding, resize, and blur inside the
  gradient path.
- `R(·)` is a sampled restorer: `GFPGAN` with probability `p_gfp`,
  `CodeFormer` with probability `p_cf`, `Real-ESRGAN` with probability
  `p_rer`, identity with probability `p_id`.
- `pipeline_i(·)` is the *i*-th ensemble target: RetinaFace, SCRFD,
  YuNet, MTCNN, ArcFace, MagFace, AdaFace, SD 1.5 VAE, SDXL VAE, Flux
  VAE, OpenCLIP-H, SigLIP-L, DINOv2-L. Each has its own loss `L_i`
  documented in the corresponding subsystem doc.
- `L_percep` combines LPIPS, SSIM, and L∞ perceptual constraints. See
  `Documentation/attacks/pixel.md`.

---

## Why prior tools omit the restorer

Two reasons:

1. **Compute.** Every restorer forward pass is expensive. `GFPGAN`
   alone is roughly the same forward cost as a small StyleGAN2
   generator. Adding restorers to a 13-target ensemble multiplies the
   per-step wall-clock by ~2–3×. At the per-image PGD scale that
   PhotoGuard uses, this is prohibitive.

2. **Gradient plumbing.** The restorers are conditional generators that
   internally re-detect the face, crop, align, run a StyleGAN2 decoder,
   and blend the output back. Making the whole chain differentiable
   requires either replacing the discrete face-crop step with a
   differentiable relaxation or using a straight-through estimator on
   it. Both are engineering effort no shipped tool has invested.

We invest both because we amortize the cost across training rather than
paying it at deploy time. The end user runs a single forward pass through
`G`. The restorer cost only lands on us, once, during training.

---

## Implementation notes

- The restorer models live in `src/voidface/models/restorers/`. Each
  wraps its upstream implementation with a `EnsembleTarget` interface
  and a differentiable face-crop layer.
- Sampling the restorer per batch (rather than always applying the same
  one) mixes the gradient signal across restorers, which helps
  cross-restorer transfer.
- Identity restorer (`R = 1`) is included with probability `p_id ≈ 0.3`
  so that `G` still works well against unrestored pipelines.
- Empirically the training loss curve has two phases: a fast phase where
  `G` learns detector suppression (restorer-invariant, since detectors
  run before restoration) and a slow phase where `G` learns to embed
  ArcFace/CLIP disruptions that survive restoration. The second phase is
  where the compute goes.

---

## What we know does not work

- Adding a *frozen*, *non-differentiable* restorer as a rejection filter
  (train `G` normally, discard samples the restorer defeats) — trained
  models converge but the restorer-defeated share stays near 100%.
- Training against a *single* restorer — `G` overfits to that restorer's
  StyleGAN2 prior. Cross-restorer transfer is poor.
- Alternating training (`G` optimized for `k` steps, then a fresh
  restorer instance for `k` steps) — no measurable improvement over
  joint training, at ~2× the cost.

---

## Open questions

- Does a semantic warp objective (`Documentation/attacks/semantic.md`)
  compose cleanly with the bilevel restorer term? Initial experiments
  suggest yes, but the joint loss is under-explored.
- Which restorers we should include as attacker models is a moving
  target. As new restorers ship, the ensemble in
  `src/voidface/models/restorers/` grows.
