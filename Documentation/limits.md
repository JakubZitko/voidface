# Voidface — honest limits

This document exists because the class of tools Voidface belongs to has a
history of being oversold. Every claim below is deliberately conservative.
If we discover we were wrong in either direction, this file is updated.

Read this before you ship a photo you care about.

---

## What Voidface can do

- Raise the cost of automated bulk scraping. A scraper that runs a face
  detector to triage millions of photos will discard Voidface-protected
  photos as "no face found" and move on.
- Break un-augmented (default-config) face-swap pipelines that skip the
  face-restoration step. This is the small share of hobbyist workflows,
  not the professional or commercialized pipelines.
- Provide meaningful protection against off-the-shelf IP-Adapter,
  InstantID, PhotoMaker, and PuLID personalization built on the model
  ensemble we included in training, at the moment we shipped this
  version of `G`.

---

## What Voidface cannot do

- **Beat face restoration in every case.** Face restorers (GFPGAN,
  CodeFormer, Real-ESRGAN, RestoreFormer++) regenerate face pixels from
  an FFHQ prior. We include them in the training loop, but the arms race
  is real: a restorer trained *after* Voidface was released, or a chained
  restorer, will erode our defense. Assume our worst-case attack-success
  rate against a fully-restored pipeline is 30–60%, not >90%.
- **Survive camera recapture.** A photograph of a screen showing a
  Voidface-protected image is essentially clean. Any physically-mediated
  channel — phone photo, printout, projector recording — strips the
  perturbation. This is a physics limit, not an engineering one.
- **Protect a photo that is already public.** If the attacker already
  has a clean copy of your image from before you protected it, Voidface
  cannot help. Protection is a *pre-upload* practice.
- **Give lifetime protection.** New encoders, new detectors, new
  restorers will appear. Voidface is a snapshot-in-time defense against
  the pipelines that existed when the shipped `G` was trained. Retrain
  and re-release cycles are how this stays useful.
- **Serve as a substitute for the criminal justice system, platform
  policy, or content-provenance standards.** Use Voidface *in addition
  to* StopNCII.org, NCMEC Take It Down, C2PA, and prosecution under
  applicable law. Not instead of them.

---

## What we deliberately do not target

- Detection-only tools (DeepFake-o-Meter, Sensity). These identify AI
  output after the fact; they are orthogonal to Voidface.
- Personalized fine-tunes trained on many Voidface-protected images.
  An adaptive attacker who collects 50+ protected photos of the same
  person and fine-tunes their generator on the protected distribution
  will strip most of the defense. See Radiya-Dixit & Tramer (ICLR 2022)
  for the general result.

---

## Reporting a real-world failure

If you observe Voidface-protected media successfully attacked by a
specific pipeline, please report it — anonymously if needed — via the
process in `Documentation/process/security.md`. Concrete failure cases
are the input to the next retraining cycle.
