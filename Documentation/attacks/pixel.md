# Pixel-space attack

The pixel-space attack is the baseline for every ensemble target. It
optimizes an `L∞`-bounded additive perturbation `delta` in the RGB
domain and is subject to a perceptual budget (LPIPS, SSIM, total
variation).

## Constraints

- `‖ delta ‖_∞ ≤ epsilon`, epsilon in `[4/255, 16/255]` (default 12/255).
- `LPIPS( x, x + delta ) ≤ 0.05` (Alex-net backbone).
- `SSIM( x, x + delta ) ≥ 0.98`.
- `PSNR( x, x + delta ) ≥ 40 dB`.

## Optimizer

- Sign-gradient PGD with momentum (β = 0.9), step size `α = epsilon / 6`.
- Initialized uniformly in `[-epsilon/2, +epsilon/2]`.
- 200–500 steps for reference-quality per-image attacks; the generator
  `G` replaces this with a single forward pass at deploy time.

## Loss

The composite loss is documented in
`Documentation/training/overview.md`. In short, it is a weighted sum
across the ensemble targets:

    L = Σ λ_i · L_i( pipeline_i( restorer( eot( x + delta ) ) ) )
      + λ_percep · L_percep( x, x + delta )

The pixel-space attack is where every `L_i` is computed and where the
gradient step lives. Semantic and iris attacks (documented separately)
extend the perturbation with additional non-pixel channels.

Source lives in `src/voidface/attacks/pixel.py`.
