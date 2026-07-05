# Cutting a Voidface release

This is the runbook for a release-maintainer. Follow the steps in
order; every step has a checkpoint before the next begins.

---

## Prerequisites

- Write access to the git remote.
- HF Hub write access to the release repo (once self-hosted weights
  land).
- Apple Developer ID (macOS build) — 0.2.0 onward.
- The four artifacts a real release ships:
  * fp32 ONNX
  * int8 static-quantized ONNX
  * CoreML `.mlpackage` (Apple Silicon build)
  * ORT-Web `.ort`

---

## Step 1 — Freeze `main`

- Ensure CI is green on `main`.
- Verify `voidface --version` matches the target release version
  (`0.1.0`, `0.1.1`, etc.). Bump if needed with a single commit
  that updates `src/voidface/__init__.py`, `pyproject.toml`, and
  `tools/cli/pyproject.toml`.

---

## Step 2 — Train + validate

- Point the training config at the corpus you want to train on.
  `voidface init full -o release-cfg.toml` and edit `[data]`.
- `voidface config-check release-cfg.toml` — catches typos.
- `voidface train release-cfg.toml --dry-run` — catches resource
  drift.
- `voidface train release-cfg.toml` — cloud GPU. 200k-500k steps.
- `voidface bench <latest-checkpoint> ~/data/ffhq-test/ --json
  release-bench.json --limit 500` — get the top-line metrics.
- Compare against the last release with `voidface bench ... --baseline
  previous-release-bench.json`. Any regression is a blocker.

Ship gate (working consensus):

    detection ASR                >= 0.60
    identity cos+1              <= 0.20
    PSNR (mean)                 >= 30 dB
    SSIM (mean)                 >= 0.92

Enforce automatically:

    voidface bench <ckpt> ~/data/ffhq-test/ --limit 500 --strict

Exits with code 3 if any threshold fails. Per-metric overrides are
`--strict-detection-asr`, `--strict-identity-cos`, `--strict-psnr`,
`--strict-ssim` — keep these tuned to the numbers above so the CLI
and this runbook cannot drift apart.

---

## Step 3 — Package + verify

- `voidface package <ckpt> release/<version>/ --calibration-dir
  ~/data/ffhq-cal/ --coreml`
- `voidface verify release/<version>/` — sanity check the bundle
  before upload.
- Attach `release/<version>/MANIFEST.json` to the release notes
  draft.

---

## Step 4 — Write release notes

Follow the `CHANGELOG` layout: version, one-line description, an
"What ships" bullet list, an honest "What still doesn't ship"
list. Reference the R-phase(s) that landed the significant work.

---

## Step 5 — Tag + push

    git tag -a v0.1.0 -m "voidface 0.1.0"
    git push origin main --tags

---

## Step 6 — Upload the release bundle

Upload `release/<version>/` to the release channel:

- GitHub release attached to the tag.
- HF Hub self-hosted repo (once landed).
- CDN URL for the browser demo.

Include MANIFEST.json + CHECKSUMS.sha256 as top-level attachments so
consumers can `voidface verify` after download.

---

## Step 7 — Announce

- Update `Documentation/status.md` with the new "shipped today"
  numbers.
- Update the browser demo's default model URL to point at the new
  release.
- Post to the project's channels (blog, mailing list, X/Bluesky).
- Include the honest limits section of the release notes — Voidface
  overpromising has real downstream consequences.

---

## Rollback path

If a released checkpoint turns out to have a regression:

1. Revert the release announcement channels to point at the
   previous release URL.
2. Retain the buggy release bundle on the CDN so links stay live;
   add a `KNOWN-ISSUES.md` in that bundle that describes the
   problem.
3. Cut a `0.1.1` (or equivalent) point-release with the fix.
4. Do NOT delete or move the buggy release from the CDN — cached
   installers may still reference it.
