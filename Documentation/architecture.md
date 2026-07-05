# Voidface architecture

This document describes the big picture: what Voidface does, why the code
is laid out the way it is, and where the load-bearing decisions live.

For the exact style rules that apply everywhere, see `coding-style.md`.
For per-subsystem detail, see the sibling files under `Documentation/`.

---

## What we ship

Voidface ships two things:

1. **A trained generator `G`.** A small (~10 MB after quantization)
   neural network exported to CoreML (Apple Neural Engine), ONNX
   (cross-platform), and a WASM/WebGPU build (browser). Applied to any
   face photo, `G` outputs a visually indistinguishable image whose
   internal representation, as seen by the standard AI face pipeline
   (detection → alignment → identity embedding → generation), is
   nulled or misdirected.

2. **The training system that produced `G`.** Open source. Reproducible.
   Anyone can retrain against new attackers as they emerge.

Users interact only with `G` via one of three surfaces:
`tools/cli/`, `tools/desktop/`, or `tools/web/`. They do not need a GPU,
Python, or any technical knowledge. The training system lives in
`src/voidface/` and is used only by researchers and contributors.

---

## The threat we are defending against

The concrete threat is off-the-shelf face-swap and nudify pipelines that
take a scraped photo of a person and produce non-consensual imagery.
Every such pipeline in 2026 has essentially the same shape:

    input photo
        │
        ▼
    face DETECTOR   ── RetinaFace / SCRFD / YuNet / MTCNN / BlazeFace
        │            (fails here → whole pipeline stops)
        ▼
    landmark ALIGNER
        │
        ▼
    identity ENCODER ── ArcFace / MagFace / AdaFace / CurricularFace
        │
        ▼
    generative MODEL ── SD 1.5 / SDXL / Flux / InstantID / IP-Adapter
        │
        ▼
    face RESTORER   ── GFPGAN / CodeFormer / Real-ESRGAN
        │            (this is what erased prior defenses)
        ▼
    output

Prior tools (PhotoGuard, Fawkes, Glaze) attacked one or two boxes above
the restorer, and the restorer erased the attack. Voidface is designed
around this observation.

---

## Design principle 1 — attack the choke point

The face detector is the only stage the restorer cannot compensate for:
if the detector says "no face here", the restorer never runs. Voidface's
strongest single lever is a detector-blinding perturbation that
transfers across all mainstream detectors simultaneously. This lives in
`src/voidface/models/detectors/` and `src/voidface/attacks/pixel.py`.

## Design principle 2 — include the restorer in the training loop

The novel training-time contribution is a bilevel objective:

    minimize   L(pipeline(restorer(G(x)))) + L_perceptual(G(x), x)
        G

That is, we optimize `G` so that even *after* the attacker runs
GFPGAN / CodeFormer / Real-ESRGAN on our output, the resulting image
still fails downstream detection and identity extraction. Prior work
optimized against the pre-restorer signal only. See
`Documentation/training/bilevel-adversarial.md`.

## Design principle 3 — separate the training system from the shipped model

The training system (`src/voidface/`) is a heavyweight PyTorch codebase.
The shipped model (`runtime/`) is a small pure-inference artifact with
no PyTorch dependency. Users of Voidface never install PyTorch. This
separation is what keeps the "runs on a basic MacBook" promise honest.

## Design principle 4 — the shape of the tree matches the shape of the system

`Documentation/`, `src/`, and `tests/` all follow the same subsystem
layout. A change to the RetinaFace surrogate touches:

- `src/voidface/models/detectors/retinaface.py`
- `Documentation/models/detectors.md`
- `tests/unit/models/detectors/test_retinaface.py`

and nothing else. If a change spans more subsystems than that, it is
probably the wrong change.

---

## Subsystem map

    voidface/
    ├── src/voidface/
    │   ├── core/           training loop, PGD, EOT, loss composition
    │   ├── models/         differentiable surrogates of attacker models
    │   │   ├── detectors/  RetinaFace, SCRFD, YuNet, MTCNN
    │   │   ├── recognizers/ArcFace, MagFace, AdaFace
    │   │   ├── vaes/       SD 1.5 VAE, SDXL VAE, Flux VAE
    │   │   ├── clip/       OpenCLIP-H, SigLIP-L, DINOv2-L
    │   │   └── restorers/  GFPGAN, CodeFormer, Real-ESRGAN  ← bilevel target
    │   ├── attacks/        pixel-space PGD, semantic warp, iris attack
    │   ├── generator/      the G network being trained (this is what ships)
    │   ├── data/           datasets, face alignment, augmentation
    │   ├── eval/           attack-success-rate, perceptual metrics
    │   ├── export/         CoreML, ONNX export
    │   └── util/           image I/O, logging, config
    ├── tools/
    │   ├── cli/            command-line tool
    │   ├── desktop/        Tauri desktop app
    │   └── web/            browser demo (WebGPU/WASM)
    ├── runtime/            shipped inference wrappers (CoreML, ONNX, WASM)
    ├── scripts/            maintenance (model download, benchmark, etc.)
    ├── samples/            sample configs, sample images
    ├── tests/              unit and integration tests
    └── Documentation/      docs mirroring the src/ layout

---

## Dependency graph (training-side)

    core        ─────────────────────────────────────► generator
      │                                                    ▲
      ▼                                                    │
    models  ────────────────────────► attacks  ────────────┘
      │                                  │
      ▼                                  ▼
    data                                eval
      │                                  ▲
      └──────────────────────────────────┘

    util is depended on by everything and depends on nothing.

Runtime rule: no module in `models/` may import from `core/`, `attacks/`,
or `generator/`. This is a hard architectural rule; a lint check
enforces it.

---

## What is deliberately not in scope

- **Detection of AI-generated content.** Voidface prevents generation
  from your photo; it does not detect deepfakes after the fact. Use
  DeepFake-o-Meter or Sensity for that.
- **Provenance / signing.** Voidface does not attest that a photo is
  authentic. Use C2PA / Content Credentials for that.
- **Reactive takedown.** Voidface does not scan the internet for misused
  images. Use StopNCII.org or NCMEC Take It Down for that.
- **Voice cloning.** Voidface is image and short video only.

We recommend Voidface be used *in combination* with all of the above.
See `Documentation/limits.md` for the honest failure modes and the
projects we recommend alongside Voidface.
