# Desktop deployment

`tools/desktop/` is a Tauri 2.x scaffold (currently placeholder-only)
for a native macOS / Windows / Linux protection app. This document
tells you how to move from the placeholder to a shipped app.

## Current shipped state

Placeholder. `tools/desktop/README` documents the intent. Everything
else routes through the CLI or the browser demo.

## Why Tauri

- Small binary (~5 MB shell + ~5 MB model + Swift/ObjC/Rust glue)
  vs Electron's ~150 MB baseline.
- Native OS APIs (menu bar, drag-and-drop, share-sheet integration).
- macOS build hosts the CoreML runtime naturally — the `.mlpackage`
  Voidface's `voidface export --coreml` produces embeds directly.
- Rust `ort` crate ships ORT + WebGPU support for cross-platform
  inference on Windows / Linux.

## Recommended shape

    tools/desktop/
    ├── src/                     Frontend (React + TypeScript recommended)
    │   ├── App.tsx              Drag-drop UI mirroring tools/web/
    │   ├── main.tsx             Vite entry
    │   └── model/
    │       └── inference.ts     ort-web or Rust IPC glue
    ├── src-tauri/               Rust backend
    │   ├── Cargo.toml
    │   └── src/
    │       ├── main.rs          Tauri app + IPC handlers
    │       ├── inference.rs     `ort` crate wiring
    │       └── coreml.rs        macOS-only CoreML wiring
    ├── package.json
    ├── vite.config.ts
    └── README

## Model bundling

- **macOS:** Bundle `voidface.mlpackage` in the app's `Resources/`
  folder. Load via `CoreML` framework through the Swift/ObjC bridge.
- **Windows/Linux:** Bundle `voidface.int8.onnx` (or `.static-int8`)
  and load through the Rust `ort` crate.

Do NOT bundle the full fp32 `.onnx` — the ~21 MB adds materially to
download size and quantized parity is verified in R5.4d-follow.

## Signing and notarization

- **macOS:** Apple Developer ID required. `tauri build` handles
  signing when `TAURI_APPLE_SIGNING_IDENTITY` is set. Notarization
  runs via `codesign` + `notarytool`.
- **Windows:** Code signing certificate required for SmartScreen.
- **Linux:** No signing required for tarballs; distributions may
  require additional metadata for their package managers.

## Distribution channels

- **Direct download.** Recommended for the first release. Users
  fetch the DMG / MSI / AppImage from the project's GitHub releases.
  Attach the `voidface package` bundle's MANIFEST.json to the
  release notes; users can `voidface verify` after download.
- **Homebrew, Winget, Flatpak.** Follow-up once the direct-download
  release is stable.

## What's NOT in scope

- **App-Store distribution.** Voidface's threat model — user
  provides source photos which never leave the machine — doesn't
  cleanly fit either the Mac App Store's sandbox rules (which limit
  file-system access) or its content policies (which may object to
  adversarial-security research tooling). Direct download is
  simpler and preserves user privacy better.
- **Cloud sync.** Voidface is expressly local-only. Any "sync your
  protected photos across devices" feature belongs in a separate
  companion product.

## Roadmap

0.2.0 is the first shipping desktop version, gated on R5.5 producing
a real checkpoint. Between now and then the `tools/desktop/`
scaffold gets:

- Tauri 2.x project init.
- macOS-first inference path via the R5.4b CoreML export.
- Drag-drop UI mirroring the browser demo's shape.
- Batch mode.
- Video protection.
- Auto-updater.
