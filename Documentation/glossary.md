# Voidface glossary

Technical terms used throughout the Voidface codebase and
documentation. Cross-references point to the file where each term
is defined or first appears.

## Adversarial perturbation

A small, bounded modification to a pixel image that changes how a
downstream neural network reads that image while leaving the image
visually intact to a human. Voidface's protection is a perturbation
bounded to `[-eps, +eps]` per pixel (default `eps = 4/255`).

See: `Documentation/attacks/pixel.md`, `src/voidface/core/pgd.py`.

## Bilevel loop

A training loop that includes a downstream defender inside the
attacker's optimization. Voidface's bilevel loop puts real
face-restoration (GFPGAN v1.4) between the perturbed image and the
loss — so the delta must survive restoration, not just fool the
target on the raw pixels.

See: `Documentation/training/bilevel-adversarial.md`,
`src/voidface/models/restorers/`.

## CoreML

Apple's on-device inference format (`.mlpackage`). Voidface exports
a CoreML version of the generator so iOS and macOS apps can run
protection on the Apple Neural Engine.

See: `Documentation/deployment/coreml.md`,
`src/voidface/export/coreml.py`.

## EOT (Expectation over Transformation)

A training-time strategy where each perturbed image is randomly
transformed (resize, blur, JPEG compress) before the loss is
computed. The generator is trained against the *distribution* of
transforms, not one clean copy — which is what makes the protection
survive real-world processing.

See: `Documentation/training/eot.md`, `src/voidface/core/eot.py`.

## Encoder attack

The general shape of Voidface's protection. Instead of just fooling
the *classifier* (identify-who-this-is), we fool the *encoder*
(embed-this-face-as-a-vector) so that face-swap models see either a
non-face or a wildly different face. Kills the pipeline at its
first stage.

See: `Documentation/architecture.md`.

## GFPGAN

A widely-shipped face-restoration model (TencentARC). Runs in every
real face-swap pipeline before the swap step, because attackers
know low-quality inputs make swaps look bad. Voidface trains
against it *inside* the loop — the delta has to survive
GFPGAN-restore.

See: `Documentation/models/restorers.md`,
`src/voidface/models/restorers/gfpgan.py`.

## Generator (G)

The small U-Net that ships as the deploy artifact. Trained once
against the ensemble; at deploy time, `G(image) = protected_image`
in ~500 ms on a basic MacBook. No GPU, no PyTorch, no per-image
optimization.

See: `Documentation/models/generator.md`,
`src/voidface/generator/architecture.py`.

## LPIPS

Learned Perceptual Image Patch Similarity — a distance function
that scores how different two images look to a human. Voidface uses
LPIPS as the "don't visibly wreck the image" constraint on the
adversarial pair, and as the "the restored versions should also be
different" bilevel term.

See: `Documentation/training/overview.md#loss-composition`.

## MI-FGSM

Momentum Iterative Fast Gradient Sign Method — the PGD variant
Voidface uses. Adds a momentum term to the sign-gradient update so
the optimization doesn't oscillate around narrow adversarial
regions.

See: `src/voidface/core/pgd.py`.

## ONNX

Open Neural Network Exchange — the portable model format Voidface
exports the generator to. Runs on CPUs everywhere via ONNX Runtime.

See: `Documentation/deployment/onnx.md`,
`src/voidface/export/onnx.py`.

## ORT-Web

A converted ONNX model in a format ONNX Runtime Web can load fast
in a browser. Voidface exports one so the browser demo starts up
without downloading and JIT-compiling a full ONNX graph.

See: `Documentation/deployment/browser.md`,
`src/voidface/export/ort.py`.

## PGD (Projected Gradient Descent)

The canonical way to compute adversarial perturbations: run gradient
descent on the pixel image with the constraint that each pixel
stays within `[-eps, +eps]` of the clean value. Voidface's
`voidface protect` without `--use-generator` uses PGD per image.

See: `Documentation/training/overview.md`, `src/voidface/core/pgd.py`.

## Restorer

Any function that takes a possibly-degraded image and returns a
cleaner one. Voidface's `RestorerSampler` picks one per training
step from `{identity, sd15-vae-roundtrip, gfpgan-v1.4}` so the
generator learns to be robust against all of them.

See: `Documentation/models/restorers.md`.

## Iris boost

A composable extension to pixel PGD: the iris region gets a locally
higher L-infinity budget (default 2× epsilon) while the rest of the
image stays at the standard budget. Motivation: face recognizers
assign heavy weight to iris texture, but humans do not perceive
sub-millimeter iris changes at ordinary viewing distance, so this
is budget the eye pays back to the attacker without visual cost.

See: `Documentation/attacks/iris.md`,
`src/voidface/attacks/iris.py`, `voidface protect --iris-boost`.

## Semantic warp

The geometric attack that composes with the pixel PGD. Instead of
only nudging pixels, small (~2 px) Gaussian-smoothed displacement
fields warp the image slightly. Detectors and recognizers are much
more sensitive to spatial disruption than they are to pure pixel
noise.

See: `Documentation/attacks/semantic.md`,
`src/voidface/attacks/semantic.py`.

## Surrogate

A concrete model Voidface trains against as a stand-in for "the
real thing an attacker will use." The Voidface ensemble surrogates
are: RetinaFace-R50 (detector), ArcFace-IResNet-100 (recognizer),
SD 1.5 VAE + SDXL VAE (diffusion encoders), OpenAI CLIP ViT-B/32
(vision-language). Better ensemble → better transfer.

See: `Documentation/models/`.

## Transfer

The degree to which an attack trained against surrogates A, B, C
still works against a real deployed model D that wasn't in the
training set. Voidface's design bet is: with a broad-enough
surrogate ensemble, transfer to real deployed pipelines is high.

See: `Documentation/architecture.md#the-transfer-bet`.
