# Identity recognizer ensemble

Face-swap and personalization pipelines universally use a frozen face
recognizer as the identity conditioner. If we can drive the recognizer's
output embedding far enough from the true identity, the downstream
generator either produces a different person, produces artifacts, or
refuses to converge.

**Currently shipped (R4.3):** ArcFace / IResNet-100. Vendored
architecture from InsightFace arcface_torch (Apache-2.0) at
`src/voidface/models/recognizers/_iresnet.py`, wrapped at
`src/voidface/models/recognizers/arcface.py`. Weights fetched from
`minchul/cvlface_arcface_ir101_webface4m` (249 MB). BGR channel flip
before normalization to [-1, 1] since arcface_torch was trained on
cv2 BGR pixels — the R4 correctness critic caught this.

**Ensemble plan (R5.5+):** ArcFace ships today; MagFace and AdaFace
land as follow-up wrappers reusing the vendored IResNet-100 with
their own state-dict conversion.

| Recognizer  | Backbone    | Loss family      | Weight | Status         |
| ----------- | ----------- | ---------------- | ------ | -------------- |
| ArcFace     | IResNet-100 | Additive margin  | 0.50   | ✅ shipped R4.3|
| MagFace     | IResNet-100 | Magnitude-aware  | 0.25   | roadmap        |
| AdaFace     | IResNet-100 | Quality-adaptive | 0.25   | roadmap        |

ArcFace is weighted most heavily because every mainstream face swap
(SimSwap, InSwapper, FaceShifter, HifiFace, InstantID) uses ArcFace as
the identity conditioner. MagFace and AdaFace provide transfer to
newer/derived face-recognition backbones and add adversarial diversity.

The attack goal is `cos(f(x + delta), f(x)) < -0.4` — pushing the
embedding to nearly opposite direction on the hypersphere. Verification
thresholds at FAR=1e-4 sit around 0.35–0.40, so any downstream
verification treats the two embeddings as different people.

MagFace additionally exposes magnitude as a signal (low-magnitude
embeddings are treated as low-quality by MagFace's adaptive margin). We
attack magnitude as a secondary term to slip past class boundaries.

Full loss formulas and current benchmark numbers live in
`src/voidface/models/recognizers/` and its module docstrings.
