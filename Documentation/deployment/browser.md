# Browser deployment

Voidface runs entirely in the user's browser via ONNX Runtime Web with
the WebGPU execution provider (WASM as fallback). Files never leave
the tab. No accounts. No servers.

The scaffold is `tools/web/`. This document tells you how to move
from that scaffold to a shipped deployment.

## Current shipped state

- `tools/web/` — real Vite + TypeScript + `onnxruntime-web` scaffold.
- `voidface export --ort` produces the `.ort` file the browser demo
  loads. `voidface export --quantize int8` produces a smaller `.onnx`
  that also works.
- `voidface package` produces a full release bundle (fp32 ONNX +
  int8 + static-int8 + ORT + CHECKSUMS + MANIFEST + README).

## Building the demo locally

    cd tools/web
    npm install
    npm run dev

Then load `http://localhost:5173`. Pick a `.ort` file via the model
picker; drop an image on the drop zone; the protected version renders
side-by-side.

## Building for production

    cd tools/web
    npm run build
    # -> tools/web/dist/ contains the static site.

Deploy that directory to any static host (Netlify, Cloudflare Pages,
GitHub Pages, S3+CloudFront, ...). The included Vite config sets
COOP/COEP headers required by WebGPU cross-origin isolation.

## Model bundling options

There are three ways to get the `.ort` file to the user:

1. **Ship the model in the site** — copy the exported `.ort` into
   `tools/web/public/` before `npm run build`. Users get the model
   as part of the initial site load. Simplest, best for a demo,
   worst for site size (~5-15 MB).
2. **Fetch the model on demand** — leave `tools/web/public/` empty
   and have the user click a "Download model" button that fetches
   the `.ort` from a release URL and caches in IndexedDB. Best for
   a production PWA.
3. **Let the user pick a local file** — the R6.1 scaffold's default
   behavior. Best for a research demo where users bring their own
   model.

## Model size targets

For an installable PWA the interactive experience budget is roughly
5-10 MB of assets. The int8 static-quantized `.onnx` or `.ort` fits
easily. The fp32 `.onnx` (~21 MB) is too large for a first-visit
download; use the quantized artifact.

## Load-then-run pipeline

The scaffold's `src/main.ts` shows the concrete pipeline; the notable
pieces:

- **Provider preference.** WebGPU when `navigator.gpu` is present;
  WASM fallback otherwise. Both work; WebGPU is 5-20x faster.
- **Input padding.** The generator expects side lengths divisible
  by 2^num_stages (16 for the default configuration). The scaffold
  reflect-pads uploaded images to the next multiple before inference
  and crops back.
- **RGBA -> planar float32 CHW.** The scaffold's
  `imageDataToFloatTensor` handles this conversion. Watch for the
  common bug of leaving stride-based iteration in place — planar
  float32 wants `[all_R, all_G, all_B]`, not `[R,G,B, R,G,B, ...]`.

## Known limitations

- **WebGPU on Safari.** As of 2026, WebGPU support in Safari is
  behind a flag on most releases. The WASM fallback keeps the demo
  working there, at 5-20x slower.
- **Memory ceiling on mobile browsers.** Very large images
  (>2048 px on the short side) may hit WebGPU or WASM memory limits.
  The scaffold does not currently downscale before inference; add
  a client-side downscale to a working resolution if you target
  high-res inputs.
- **Video is out of scope for the browser demo.** Per-frame video
  protection needs an entire ORT session's worth of compute per
  frame; realistic browser experiences want a server-side path or a
  desktop app. `voidface protect-video` in the CLI covers the
  offline case today.

## What R6.1 does NOT ship

- **Automatic model download.** The scaffold uses a file picker.
  Wire in `fetch(url).then(response => response.arrayBuffer())` plus
  IndexedDB caching for a production PWA.
- **PWA install prompt.** Standard `manifest.json` + service worker.
- **Batch mode.** Drop multiple images at once — currently one image
  at a time.
- **Progress UI.** Long WebGPU inferences currently show a plain
  "running..." status. Add a `<progress>` element with
  session.run's progress callback when your workload demands it.
