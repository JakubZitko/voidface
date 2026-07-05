# ONNX deployment

ONNX is the cross-platform shipping target. It runs `G` via ONNX Runtime
on Windows, Linux, and older macOS, and it is the source for the
WASM/WebGPU browser build.

## Export flow

    src/voidface/generator/architecture.py   ── PyTorch definition of G
                     │
                     ▼
    src/voidface/export/onnx.py              ── exports to ONNX
                     │
                     ▼
    runtime/onnx/voidface.onnx               ── shipped artifact
                     │
                     ▼
    runtime/wasm/voidface.ort                ── ORT format for web

## Export flags

- `opset_version=17` (broad ORT support, includes GridSample if we need
  the semantic-warp path).
- `dynamic_axes` for height/width, so a single ONNX serves multiple
  input resolutions.
- Weight quantization to int8 via ORT quantization tools where the ANE
  is not available.

## Runtime wrappers

- `runtime/onnx/` — Python wrapper for use from `tools/cli/` on
  non-Apple platforms; also usable from Rust via `ort` crate for the
  desktop app.
- `runtime/wasm/` — ONNX Runtime Web build with WebGPU execution
  provider, used by `tools/web/`.

## What we do not ship

- **ONNX GraphSurgeon patched graphs.** If the model needs
  restructuring, we fix it in the PyTorch source, not the ONNX artifact.
- **Per-platform ONNX variants.** One ONNX file per architecture family
  (32-bit float reference; int8 quantized). Everything else is a
  runtime concern.
