# Face detector ensemble

The face detector is the top of the attack pipeline and, uniquely, the
one stage a face restorer cannot compensate for. If the detector misses
the face, the entire downstream pipeline does not run. Every attack in
Voidface targets this stage first.

**Currently shipped (R4.4):** RetinaFace-R50. Vendored from
biubug6/Pytorch_Retinaface (MIT), weights fetched from
`yakhyo/retinaface-pytorch` (109 MB). Returns raw pre-softmax
classification logits + bbox + landmarks. The wrapper is at
`src/voidface/models/detectors/retinaface.py`.

**Ensemble plan (R5.5+):** the composite loss uses the R5.5 reference
weights below; RetinaFace ships today and the other three land as
additional surrogates in follow-up commits.

| Detector    | Backbone        | Anchor style     | Weight in ensemble | Status         |
| ----------- | --------------- | ---------------- | ------------------ | -------------- |
| RetinaFace  | ResNet-50       | Anchor-based FPN | 0.15               | ✅ shipped R4.4|
| SCRFD       | Custom ResNet   | Anchor-free      | 0.15               | roadmap        |
| YuNet       | Tiny (~85K)     | Anchor-free      | 0.10               | roadmap        |
| MTCNN       | P/R/O-Net       | Cascade          | 0.10               | roadmap        |

Full attack surface analysis, per-detector loss formulas, and current
attack-success-rate numbers live in the subsystem source under
`src/voidface/models/detectors/` and its docstrings.

The load-bearing insight: the "face-present" logit in each of these is a
low-rank projection of a small FPN feature. Suppressing it requires
gradient signal along a small subspace of feature space, which is why
even 4/255 perturbations often work at the pyramid P3 stride-8 level.

See `Documentation/attacks/pixel.md` for the mechanics of how we drive
the suppression loss.
