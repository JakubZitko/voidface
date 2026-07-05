# Identity recognizer ensemble

Face-swap and personalization pipelines universally use a frozen face
recognizer as the identity conditioner. If we can drive the recognizer's
output embedding far enough from the true identity, the downstream
generator either produces a different person, produces artifacts, or
refuses to converge.

| Recognizer  | Backbone    | Loss family    | Weight |
| ----------- | ----------- | -------------- | ------ |
| ArcFace     | IResNet-100 | Additive margin | 0.50   |
| MagFace     | IResNet-100 | Magnitude-aware| 0.25   |
| AdaFace     | IResNet-100 | Quality-adaptive| 0.25   |

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
