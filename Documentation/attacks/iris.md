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

Phase R3. Documented here so the subsystem layout matches the eventual
implementation.
