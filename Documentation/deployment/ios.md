# iOS deployment

Voidface is well-shaped for iOS via CoreML on the Apple Neural
Engine. This document describes the intended shape.

## Current shipped state

`tools/ios/` does not yet exist. The `voidface export --coreml`
path produces the `.mlpackage` an iOS app would embed.

## Why iOS matters

The concrete NCII / face-swap threat model most often targets
consumer photos captured on and shared from smartphones. The best
possible UX is a share-sheet extension that intercepts a photo
between the camera roll and Instagram / TikTok / Messages and hands
back the protected version invisibly.

## Recommended shape

    tools/ios/
    ├── Voidface.xcodeproj/
    ├── Voidface/
    │   ├── App/                 SwiftUI shell
    │   ├── Model/
    │   │   ├── Voidface.mlpackage       (~5-15 MB, from voidface export)
    │   │   └── VoidfaceRunner.swift     Load + predict wrapper
    │   ├── Views/
    │   │   ├── ProtectView.swift        Drag-drop or picker
    │   │   ├── BatchView.swift          Camera-roll multi-select
    │   │   └── VideoView.swift          Per-frame protection
    │   └── ShareExtension/               Extension target
    │       └── ShareViewController.swift
    └── VoidfaceTests/

## Model bundling

- **Static:** Ship `Voidface.mlpackage` in the app bundle. Load with
  `MLModel(contentsOf: url, configuration:)`. Compute units:
  `.all` so the Neural Engine is prioritized with GPU/CPU fallback.
- **Dynamic:** Optionally fetch a newer model from the release URL
  and cache in the app's Application Support directory. Wire in
  after 0.2.0.

## Neural Engine considerations

- The R5.1 generator is fully convolutional + GroupNorm + GELU +
  ConvTranspose. Every layer maps to Neural Engine ops on M-series
  iPhones (A15+) and iPads (M1+). Older devices (A13/A14) fall back
  to GPU with a small latency hit.
- Input side lengths must be divisible by 16 (the R5.1 stride
  constraint). The share-extension wrapper reflect-pads before
  inference and crops back.
- Expected latency at 1024×1024 on A17 Pro / M2: <300 ms end-to-end
  including image encode/decode.

## Share-sheet extension

The killer feature is a `Share to Voidface` share-sheet action that:

1. Receives a photo from Photos, Messages, or another app.
2. Runs `Voidface.mlpackage` on-device.
3. Copies the protected version back to Photos, or hands it back to
   the source app via the extension response.

This turns Voidface from a manual workflow into a one-tap habit.

## App Store review

Apple typically approves adversarial-defense tooling framed as
"protects the user's own photos from AI misuse." Frame the App
Store description accordingly:

- Emphasize the user's own photos, on-device processing, no data
  collection.
- Downplay research/attack terminology in the marketing surface.
- Include the honest limits from `Documentation/limits.md` in the
  app's About screen so review notes preemptively.

## What's NOT in scope

- **Video-editing app.** iOS is best for one-shot photo protection.
  Video protection at scale should stay on the desktop / CLI.
- **Cloud upload.** The whole point of Voidface is that photos
  never leave the device. An iOS Voidface app must not offer any
  upload / sync / share-to-cloud feature.

## Roadmap

0.3.0 is the first iOS-shipping version, gated on:
- 0.1.0 core (done).
- 0.2.0 trained checkpoint ships (R5.5).
- Apple Developer account for signing.

Between now and then `tools/ios/` gains:

- SwiftUI shell + Model loader.
- Photos permission wiring.
- Share extension.
- On-device batch mode.
- Basic UI polish.
