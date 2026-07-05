# Face detector ensemble

The face detector is the top of the attack pipeline and, uniquely, the
one stage a face restorer cannot compensate for. If the detector misses
the face, the entire downstream pipeline does not run. Every attack in
Voidface targets this stage first.

We include four detectors as differentiable surrogates. Blinding all
four simultaneously gives strong transfer to the fifth, sixth, seventh
detectors that we did not include, because they share the same FPN /
anchor design and are pretrained on WIDER FACE.

| Detector    | Backbone        | Anchor style     | Weight in ensemble |
| ----------- | --------------- | ---------------- | ------------------ |
| RetinaFace  | ResNet-50       | Anchor-based FPN | 0.15               |
| SCRFD       | Custom ResNet   | Anchor-free      | 0.15               |
| YuNet       | Tiny (~85K)     | Anchor-free      | 0.10               |
| MTCNN       | P/R/O-Net       | Cascade          | 0.10               |

Full attack surface analysis, per-detector loss formulas, and current
attack-success-rate numbers live in the subsystem source under
`src/voidface/models/detectors/` and its docstrings.

The load-bearing insight: the "face-present" logit in each of these is a
low-rank projection of a small FPN feature. Suppressing it requires
gradient signal along a small subspace of feature space, which is why
even 4/255 perturbations often work at the pyramid P3 stride-8 level.

See `Documentation/attacks/pixel.md` for the mechanics of how we drive
the suppression loss.
