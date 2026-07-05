# Semantic attack — sub-pixel geometric perturbation

The semantic attack is the second layer of Voidface's defense. Where
the pixel-space attack lives in RGB values, the semantic attack lives
in the *geometry* of the face.

The insight: face restorers regenerate face pixels from a StyleGAN2
prior, but they preserve overall face structure. If we change the
underlying geometry — sub-millimeter shifts in jawline, cheekbone
position, chin, and eye placement — humans do not notice, but the
restorer produces a differently-restored face, which then hashes to a
different ArcFace / CLIP identity.

## Constraints

- Warp field `w : (H, W) → (dx, dy)` with `‖ w ‖_∞ ≤ 2 px`.
- Warp is smoothed by a Gaussian with `σ = 4 px` so no sharp
  discontinuities.
- LPIPS budget as pixel attack.

## Implementation

- Warp field parameters are learned alongside `delta`. Applied via a
  differentiable `grid_sample` in `src/voidface/attacks/semantic.py`.
- Restricted to the face-region mask (does not warp background).
- Gradient flows through both the warp and the pixel perturbation
  simultaneously.

## Status

Phase R3. Not part of the phase R1/R2 baseline. Documented here so the
subsystem layout matches the eventual implementation.
