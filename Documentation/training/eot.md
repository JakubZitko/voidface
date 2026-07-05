# Expectation over transformation (EOT)

EOT is the mechanism that makes Voidface survive the ordinary lifecycle of
a photo on the internet: JPEG re-encoding, resizing, blur, color-profile
conversion, WEBP/AVIF transcoding, and screenshot.

Without EOT, an adversarial perturbation is only guaranteed to work on the
exact pixel array it was computed on. One JPEG pass at quality 75 strips
most known perturbations (Honig et al., UMD 2024, "Adversarial
Perturbations Cannot Reliably Protect Artists").

---

## The idea

Given an ensemble target `f` and a distribution `T` over transforms, we
optimize the expected loss:

    L_EOT(delta) = E_{t ∼ T}[ L( f(t(x + delta)), y ) ]

At each PGD step (or generator training step) we sample `k` transforms
from `T`, forward-pass the perturbed image through each, average the
loss, and backpropagate.

The larger `k` and the wider the support of `T`, the more robust the
perturbation. Both come at compute cost.

---

## Voidface's transform distribution

Every training step samples `k = 4` transforms from a pipeline of
mutually independent operations:

- **JPEG re-encoding** at quality `q ∼ Uniform{40, 55, 70, 85, 95}`
  using the differentiable JPEG surrogate from Reich et al. (2024).
- **Bilinear resize** by factor `s ∼ Uniform{0.5, 0.75, 1.0, 1.5, 2.0}`
  followed by resize back to the original resolution.
- **Gaussian blur** with `σ ∼ Uniform{0, 0.5, 1.0, 1.5}`.
- **Screenshot recapture surrogate** with probability 0.15 — a
  differentiable color-profile shift + nearest-neighbor resample chain.
- **WEBP/AVIF surrogate** with probability 0.10 — BPDA gradient through
  a frozen encode/decode round-trip.

The mask over these — which subset gets applied on each sample — is
itself sampled uniformly.

Sampling is per-sample rather than per-batch: the four EOT samples in a
minibatch see four different transformations, so the gradient we
backpropagate is genuinely an expectation.

---

## What we deliberately do not include

- **Diffusion purification (`DiffPure`, `IMPRESS`, `GrIDPure`).** These
  are *adversarial* transforms — an attacker's active choice, not a
  passive channel. Including them in EOT teaches `G` to survive them
  but at very high cost. They are handled separately by the
  anti-purification objective in `bilevel-adversarial.md` (restorer term
  covers most of this signal).
- **Camera recapture.** A real phone photo of a real screen introduces
  sensor noise, moire, and lens distortion that we cannot cheaply
  differentiate through. We document this failure honestly in
  `Documentation/limits.md` rather than approximate it poorly.

---

## Implementation

- `src/voidface/core/eot.py` is the sampler and the wrapper.
- Differentiable JPEG lives in `src/voidface/core/jpeg.py` (adapted from
  Reich et al. 2024, MIT-licensed).
- BPDA (backward pass differentiable approximation) for WEBP/AVIF lives
  in `src/voidface/core/bpda.py`.

---

## Choosing `k`

Larger `k` reduces gradient noise; smaller `k` reduces cost. Voidface
uses `k = 4` for the main training loop. Reference PGD in
`src/voidface/core/pgd.py` uses `k = 8` for higher-fidelity per-image
attacks (used only for evaluation).
