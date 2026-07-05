# ONNX deployment

ONNX is the cross-platform shipping target. It runs `G` via ONNX Runtime
on Windows, Linux, and older macOS, and it is the source for the
WASM/WebGPU browser build.

## Current shipped state

- `src/voidface/export/onnx.py::export_generator_to_onnx` — fp32 export
  with opset 17 and dynamic axes (batch, height, width).
- `src/voidface/export/quantize.py::quantize_onnx_generator` — dynamic
  int8 / uint8 quantization via ORT's `quantize_dynamic`. Static
  (calibrated) quant lands after R5.5 when we have a calibration
  corpus.
- `src/voidface/export/ort.py::convert_onnx_to_ort` — ORT-Web `.ort`
  format for the browser demo.
- `voidface export <ckpt> <out.onnx> [--quantize int8|uint8] [--coreml]
  [--ort]` — one command produces every deploy artifact.

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
