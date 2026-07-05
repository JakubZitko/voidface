# CoreML deployment

CoreML is the primary shipping target for Apple Silicon. It runs `G`
on the Neural Engine on any M-series Mac, iPad, or iPhone, and falls
back cleanly to the GPU or CPU on Intel and older devices.

## Current shipped state

- `src/voidface/export/coreml.py::export_generator_to_coreml` traces
  the generator on CPU, converts to ml-program CoreML with flexible
  input dims (64..2048), and applies coremltools' 8-bit linear
  symmetric weight quant by default.
- `voidface export --coreml` emits a Voidface.mlpackage sibling to
  the fp32 .onnx artifact.
- `coremltools` is Apple Silicon-only in modern releases; the
  pyproject dep is gated. On Intel Mac or non-Apple platforms, the
  code path raises `CoreMlExportError` instead of a cryptic
  `ImportError`.

## Export flow

    src/voidface/generator/architecture.py   ── PyTorch definition of G
                     │
                     ▼
    src/voidface/export/coreml.py            ── traces + converts
                     │
                     ▼
    runtime/coreml/Voidface.mlpackage        ── shipped artifact

The exporter uses `coremltools>=8.0`. Key steps:

1. Freeze `G` into a `torch.jit.trace` on a representative input.
2. Convert with `ct.convert(..., minimum_deployment_target=iOS17,
   compute_units=ALL)`.
3. Apply 8-bit weight quantization for size.
4. Validate: run the CoreML model on the sample images from
   `samples/images/` and compare against the PyTorch reference. RMS
   error must be below the threshold in
   `tests/integration/test_export_parity.py`.

## Constraints on G that CoreML imposes

- No dynamic control flow that varies with data.
- Input dimensions are variable but must be declared as flexible ranges.
- CoreML supports fp16 natively on the ANE; internal ops should be
  fp16-compatible.

## Runtime wrapper

`runtime/coreml/` contains a small Swift package that loads the
mlpackage, exposes a `Voidface.protect(_ image: CIImage) -> CIImage`
call, and is embedded by `tools/desktop/` (Tauri Swift plugin) and any
future iOS app.
