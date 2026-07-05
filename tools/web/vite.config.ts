// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Voidface contributors

import { defineConfig } from "vite";

// onnxruntime-web ships its .wasm and .mjs artifacts as separate files
// that Vite must serve alongside the main bundle. The `optimizeDeps`
// pre-bundle exclusion keeps ORT's dynamic loader working.
export default defineConfig({
  server: {
    headers: {
      // Required for WebAssembly + SharedArrayBuffer (used by ORT's
      // WebGPU EP for cross-origin isolation).
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  worker: {
    format: "es",
  },
});
