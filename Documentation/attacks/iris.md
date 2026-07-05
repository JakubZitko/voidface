# Iris attack — high-signal, low-perceptibility identity disruption

Face recognizers assign heavy weight to iris texture. Humans do not
perceive sub-millimeter iris texture changes at ordinary viewing
distance. This asymmetry is the largest single perceptibility-vs-signal
opportunity in the face image.

## Idea

Detect the iris region via the aligner landmarks, apply a targeted
higher-magnitude perturbation inside that mask while remaining at the
usual pixel budget elsewhere.

## Constraints

- Iris mask derived from the aligner's iris landmarks (available in
  MediaPipe FaceMesh; approximated from 5-point aligners via a fixed
  offset from pupil center).
- Inside the iris mask: `‖ delta_iris ‖_∞ ≤ 2 · epsilon`.
- LPIPS budget same as global constraint.

## Notes

- Prior work: rarely explored for adversarial cloaking. Some antispoofing
  literature adjacent.
- Because irises are small (roughly 3–5% of a face crop), the LPIPS
  contribution from a 2x-strength perturbation there is negligible.

## Status

Phase R7.74 and R7.75. Ships.

- `voidface.attacks.iris.iris_region_mask` — computes a soft binary
  mask over both irises from the same 5-point landmarks the aligner
  uses. Radius scales with inter-ocular distance so it is
  resolution-independent. Coverage in a 512x512 face crop is on the
  order of 0.03% by default.

- `voidface.core.pgd.run_pgd` accepts `iris_mask` and
  `iris_epsilon_ratio`. The per-pixel L-infinity budget inside the
  mask becomes `epsilon * iris_epsilon_ratio` while every pixel
  outside stays at the standard `epsilon`. Default ratio is `2.0`
  per the constraint above.

Composition into the shipped generator's training loop and a
`voidface protect --iris-boost` CLI flag remain follow-up.
