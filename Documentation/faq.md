# Frequently asked questions

## Which target photos does Voidface actually protect against?

At 0.1.0 the shipped ensemble covers:

- **Face-swap.** SimSwap, InSwapper, FaceShifter, HifiFace,
  InstantID (via ArcFace attack).
- **Nudify.** SD 1.5 + LoRA inpainting workflows (via SD 1.5 VAE
  attack).
- **Personalization.** IP-Adapter / InstantID / PhotoMaker style
  identity conditioning (via CLIP + ArcFace).
- **Modern face restoration.** GFPGAN v1.4 in the bilevel loop
  (which every real face-swap pipeline runs as a final step).

## What doesn't Voidface protect against?

Documented honestly in `Documentation/limits.md`. Short version:

- **Camera recapture.** A phone photo of a screen showing a
  Voidface-protected image is essentially clean. Physical channel
  strips the perturbation.
- **Post-facto image already scraped.** Voidface is a pre-upload
  practice. If the attacker already has a clean copy, we can't
  help.
- **Adaptive attackers fine-tuning on protected data.** An attacker
  who collects 50+ protected photos of the same person and
  fine-tunes a generator on that distribution will strip most of
  the defense.
- **Restoration arms race.** A new face restorer trained after
  Voidface's shipped weights land can erode the bilevel defense
  until we retrain and re-release.

## Where does the protection live in the file?

In perturbations to pixel values (and optionally sub-pixel geometry
via `--semantic-warp`). Bounded to `L∞ = epsilon/255` (default
12/255 ≈ 5%). Invisible to humans at typical viewing distances;
looks like normal photograph noise on close inspection.

## Do I need a GPU?

No. The shipped generator runs on:

- Apple Neural Engine on M-series Macs/iPads/iPhones (<500 ms per
  1024×1024 image).
- Intel Mac CPU (~2-3 seconds).
- Windows / Linux CPU (comparable to Intel Mac).
- WebGPU in the browser (varies; ~1-2 seconds).

The training system needs a GPU. Users who apply Voidface to their
photos do not.

## Are my photos uploaded anywhere?

No. Everything is local:

- CLI runs on your machine.
- Browser demo runs entirely in your tab (WebGPU / WASM).
- No accounts, no telemetry, no cloud sync.

## Which version of Voidface should I use?

- **0.1.x** — infrastructure only. No trained shipping weights.
  Useful for researchers and package maintainers preparing for
  0.2.0. If you're not a researcher, wait.
- **0.2.x** — first shipping release with a trained checkpoint.
  Use this if you're a consumer trying to protect your photos.
- **0.3.x+** — iOS app, desktop app, browser deployment.

## What's the recommended workflow for a regular user?

Once 0.2.x ships:

    voidface protect user.jpg --use-generator voidface.pt --face-mask

Under a second per photo. Post the protected image; delete or
archive the original locally.

## What if I care about maximum protection quality?

    voidface protect user.jpg \\
        --use-generator voidface.pt \\
        --refine-steps 20 \\
        --face-mask \\
        --semantic-warp 2.0

Roughly 15 seconds per photo. Composes the deploy-quality G forward
pass with 20 PGD refinement steps, sub-pixel geometric warp, and
face-region masking.

## Is it safe to share protected photos on social media?

Safer than the un-protected original, yes. NOT bulletproof (see
"What doesn't Voidface protect against" above). Combine with:

- Uploading images (not videos) — video deepfakes have their own
  arms race.
- Removing EXIF metadata before upload.
- Using platform-provided privacy controls.
- Registering photos with StopNCII.org / NCMEC Take It Down for
  reactive takedown.

Voidface is a proactive layer, not a substitute for the reactive
tools.

## Why doesn't Voidface have a phone app yet?

0.3.x. The 0.2.0 shipping checkpoint is a prerequisite because the
iOS app embeds the trained `.mlpackage`. See
`Documentation/deployment/ios.md`.

## Can I contribute?

Yes. See `Documentation/process/contributing.md`. Areas that most
need help:

- Additional ensemble targets (SCRFD, MagFace, AdaFace, Flux VAE).
- iOS / Android / desktop app work.
- Adversarial validation on new attacker pipelines (Kling, Sora,
  Runway Gen-3, etc.).
- Documentation improvements.
