# The generator G — what actually ships

`G` is the single artifact end users receive. It is a small neural
network trained by the system described in `Documentation/training/`,
exported to CoreML / ONNX / WASM per
`Documentation/deployment/`, and invoked from `tools/cli/`,
`tools/desktop/`, or `tools/web/`.

## Requirements on G

- **Small.** ≤ 15 MB after 8-bit quantization. Users download this
  once. Anything larger will not clear a normal broadband-connection
  first-run experience.
- **Fast.** ≤ 500 ms per 1024×1024 image on an Apple-Silicon Mac
  (Neural Engine). ≤ 3 s on CPU on a 2018 Intel Mac. No CUDA required.
- **Deterministic.** Same input, same output. `G` has no stochastic
  components at inference time.
- **Input/output.** RGB image in `[0, 1]`, arbitrary aspect ratio,
  variable resolution ≥ 512 shortest side. Output is same shape.
- **Universally applicable.** Should protect any face, any pose,
  any lighting. No per-user finetune required.

## Architecture

A conditional U-Net variant with:

- Encoder: 4 downsample stages, group-norm, GELU.
- Bottleneck: two self-attention blocks over 32×32 features (256-D).
- Decoder: 4 upsample stages, skip connections from the encoder.
- Final layer: `tanh` projected to `[-epsilon, +epsilon]` per pixel.
- Output image is `clip(input + G_delta, 0, 1)`.

Parameter count target: 5–10 M (float32), ≤ 4 M after 8-bit quant.

The concrete architecture lives in
`src/voidface/generator/architecture.py`. It is deliberately small; we
optimize for train-time-cost-vs-inference-cost tradeoff, and the
inference-cost side of that tradeoff is the constraint.

## What G is *not*

- Not a diffusion model. Diffusion is too slow to run per-photo on a
  MacBook CPU. The generator is a single forward pass.
- Not a GAN with a discriminator. The perceptual constraint is imposed
  directly via LPIPS + L∞ clip; no adversarial discriminator is
  needed.
- Not conditioned on user preference. `G(x)` is a fixed function; the
  user has no knobs beyond image input.

## Training the generator

See `Documentation/training/overview.md` and
`Documentation/training/bilevel-adversarial.md`.

## Exporting the generator

See `Documentation/deployment/coreml.md` and
`Documentation/deployment/onnx.md`.
