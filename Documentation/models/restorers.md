# Face restorer ensemble — the bilevel target

The restorer ensemble is the differentiating feature of Voidface. Prior
adversarial-cloaking tools optimized against a pre-restorer signal and
were defeated by any face-swap pipeline that ran a restorer. Voidface
includes the restorers *inside* the training loop.

See `Documentation/training/bilevel-adversarial.md` for the objective.

| Restorer      | Family      | Prior                    | Weight |
| ------------- | ----------- | ------------------------ | ------ |
| GFPGAN        | StyleGAN2    | FFHQ                     | 0.40   |
| CodeFormer    | Codebook + Transformer | FFHQ         | 0.35   |
| Real-ESRGAN   | ESRGAN + Face enhance | FFHQ            | 0.25   |

Every training step samples one restorer (with an identity option) so
that `G` learns a perturbation that survives the restorer distribution
rather than overfitting to any single one.

## Making the restorer differentiable

Each restorer includes a face detection + crop + align step internally.
Making the full chain differentiable is the primary engineering cost.
Voidface uses:

- **Differentiable face crop.** A soft attention mask over the input
  image, driven by the RetinaFace surrogate output. Zero-gradient at
  train time is avoided by a temperature-controlled softmax over the
  candidate bounding boxes.
- **Differentiable landmark alignment.** A 5-point similarity transform
  built as a learnable affine layer whose parameters are the aligned
  landmark output of the surrogate. The transform is fully
  differentiable (grid-sample + bilinear).
- **Straight-through blend.** The blend of the restored face back into
  the source image uses a straight-through estimator over the alpha
  mask so gradients flow into both the restored region and the
  perimeter.

## Known open problems

- **Restorer arms race.** Restorers ship faster than we can add them.
  We plan an incremental "add restorer" retraining protocol documented
  in `Documentation/training/retraining.md` (to be written after phase
  R2).
- **Chained restorers.** GFPGAN followed by Real-ESRGAN is a common
  strip pipeline; the second restorer partially undoes cross-restorer
  transfer gains. Current mitigation: sample chains of length 2 with
  low probability during training.
- **Cost.** Restorers are the single biggest wall-clock item in each
  training step. Optimizing them (e.g., pruning, distillation of the
  StyleGAN2 decoder) is on the roadmap.

Source lives in `src/voidface/models/restorers/`.
