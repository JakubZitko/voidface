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

## Architecture (as shipped in R5.1)

The concrete architecture lives in
`src/voidface/generator/architecture.py`; the summary here matches
the code:

- **Encoder.** 4 stages of ``_ResidualBlock(GroupNorm → GELU → 3×3
  Conv → GroupNorm → GELU → 3×3 Conv + skip)`` followed by a 2×2
  stride-2 conv downsample. Channel schedule: `base_channels`
  doubles per stage (16 → 32 → 64 → 128 → 256) with a cap at 384.
- **Bottleneck.** Two residual blocks at the deepest channel width.
  A self-attention block is available via
  `VoidfaceConfig.attention_at_bottleneck=True` but is off by default
  in R5.1 for CoreML export cleanliness.
- **Decoder.** 4 stages mirroring the encoder — 2×2 stride-2
  transposed conv upsample, concat with the encoder skip, then a
  `_ResidualBlock` that ingests `2 * channels`. The channel schedule
  reverses (256 → 128 → 64 → 32 → 16).
- **Head.** GroupNorm → GELU → 3×3 Conv → 3 output channels.
- **Delta projection.** `torch.tanh(raw) * epsilon`, added to input,
  clamped to `[0, 1]`. `epsilon` can be overridden per forward for
  deploy-time budget control.

At the default `base_channels=16` (the R5.1 shipping config), the
generator has ~5.35 M parameters float32 → ~21 MB fp32 → ~5.5 MB
int8 after static quantization. Well inside the R5 ≤ 15 MB shipping
ceiling.

**Input constraint.** Height and width must be divisible by
`2^num_stages` (16 at the default four stages). The CLI's protect /
batch / protect-video paths reflect-pad and crop; the generator's
`forward` skips shape validation when
`torch.jit.is_tracing()` returns True so ONNX export produces a
shape-agnostic graph.

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
