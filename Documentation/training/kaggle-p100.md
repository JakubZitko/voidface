<!-- SPDX-License-Identifier: MIT -->

# Training voidface on Kaggle P100 (free tier)

This is the user-facing setup guide for running the voidface v0.1
generator training loop on Kaggle Notebooks' free GPU tier. It exists
because most people reading the voidface repo do not have an A100
sitting on their desk, and Kaggle is the only zero-cost path to a
real, resumable, multi-day training run.

For the loss, target weights, and general tuning intuitions, see
`tuning.md`. For the bilevel-with-GFPGAN story, see
`bilevel-adversarial.md`. This document is only about the Kaggle
plumbing.

---

## 1. Overview

**What this guide lets you do.** Train a voidface v0.1 MVP
checkpoint on Kaggle's free P100 tier in roughly two calendar weeks
of elapsed time (about 30 GPU-hours a week, spread across five or
six ~10-hour sessions). At the end you have a checkpoint you can
run through `voidface bench` and use as a proof-of-concept for grant
applications, screenshots, or downstream research.

**What this guide does not let you do.** Hit the R5.5 ship-gate
thresholds in `tuning.md`. Those numbers (detection ASR ≥ 0.60,
identity cos+1 ≤ 0.20, PSNR ≥ 30 dB, SSIM ≥ 0.92) assume the full
200k-step run in `samples/configs/train_full.toml` at resolution
512 with batch 4-8 on an A100. Kaggle's P100 is roughly 8× slower
per step; you will run ~20-30k steps at batch 2 / res 384 with
gradient checkpointing, and you should expect the MVP checkpoint to
be well short of ship-gate on every axis. That is fine — the point
of this path is *any* real trained checkpoint, not the production
one.

---

## 2. Prerequisites

- A Kaggle account with **phone verification completed**. Phone
  verification is what unlocks GPU/TPU access and the notebook
  "Internet on" toggle. Without it you get CPU-only, no
  `pip install`, no `git clone`.
- A GitHub account (or any way of getting the voidface source into
  the notebook — the guide assumes `git clone` from the public
  repo).
- Roughly 10 GB free on your local machine, to hold the final
  checkpoint after download.

You do not need a GPU locally. Everything runs on Kaggle.

---

## 3. Step 1 — Set up Kaggle

1. Sign up at <https://www.kaggle.com/account/login?phase=startRegisterTab>.
   SSO (Google/Facebook/Yahoo) or email + 6-digit email verification
   both work.
2. **Phone-verify the account.** Profile menu → Settings → "Phone
   Verification", or use the form at
   <https://www.kaggle.com/contact#/account/activate/phone>. Some
   temporary/VoIP numbers get rejected; use a normal mobile number.
   If it silently fails, the contact form is the escalation path —
   Kaggle staff have publicly said the verification flow is
   deliberately permissive for legitimate users, but it is not
   instant.
3. Generate an API token. Account page → Settings → API →
   **"Create New API Token"**. This downloads a `kaggle.json` file
   containing `{"username": "...", "key": "..."}`. Treat it like a
   password.
4. Install the CLI locally so you can push notebooks and download
   the final checkpoint without clicking through the web UI:

    ```bash
    pip install kaggle
    ```

5. Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/macOS) or
   `C:\Users\<user>\.kaggle\kaggle.json` (Windows). Lock the
   permissions so the CLI stops warning:

    ```bash
    mkdir -p ~/.kaggle
    mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    ```

6. Verify:

    ```bash
    kaggle datasets list
    ```

    Any non-empty listing means the token works.

---

## 4. Step 2 — Get FFHQ on Kaggle

You do not need to re-upload FFHQ. Multiple public mirrors already
exist. Pick one and attach it to your notebook via **+ Add Input**
in the notebook sidebar.

Recommended for the P100 MVP path:

- `chelove4draste/ffhq-512x512` — 70k images at 512×512, ~20 GB.
  This is what the sample notebook expects. Downsample to 384 in
  the loader.
- `rahulbhalley/ffhq-256x256` — smaller (~6-7 GB) if you decide to
  train at res 256 for a faster smoke run.
- `badasstechie/celebahq-resized-256x256` — 30k CelebA-HQ faces at
  256×256, useful as an eval-only set separate from FFHQ train.

If you want a subset (recommended for early iteration — full FFHQ
is overkill for a Pascal-era GPU), build it locally and upload it as
a private dataset:

```bash
kaggle datasets init -p ~/data/ffhq-subset/
# edit dataset-metadata.json (id: "<username>/ffhq-subset", title, license)
kaggle datasets create -p ~/data/ffhq-subset/ --dir-mode zip
```

Free-tier private-dataset quota is 100 GB total; a 5-10k FFHQ subset
at 512×512 is a few GB and fits comfortably.

---

## 5. Step 3 — Create the training notebook

1. Go to <https://www.kaggle.com/code> and click **+ New Notebook**.
2. In the right sidebar, under **Settings**:
   - **Accelerator**: **GPU P100**. (T4 x2 draws from the same
     weekly quota; P100 is the more reliable single-GPU choice for
     this workload because it has 16 GB in one device and does not
     force you to fight data-parallel overhead.)
   - **Language**: Python.
   - **Environment**: "Latest" is fine.
   - **Persistence**: on. Preserves variables across interactive
     restarts; do not rely on it for anything larger than a few GB.
   - **Internet**: **on**. Required for `pip install` and
     `git clone`. Only available after phone verification.
3. Upload the training notebook from the repo:
   `tools/kaggle/train.ipynb`. Either drag-and-drop into the editor
   or use `kaggle kernels push` from a local folder with a
   `kernel-metadata.json` next to it.
4. Attach the FFHQ dataset from Step 2 via **+ Add Input** →
   search → Add. Files then appear read-only under
   `/kaggle/input/<dataset-slug>/`.

Sanity check: run the first cell only. It should print a live
`torch.cuda.is_available() == True`, the P100 device string, and a
non-empty listing of `/kaggle/input/<your-ffhq-slug>/`. If any of
those is wrong, fix it before spending session hours.

---

## 6. Step 4 — Run session 1

1. Top-right of the notebook: **Save Version → Save & Run All
   (Commit)**. This launches a batch session that ignores the ~40
   minute interactive-idle timeout and runs top-to-bottom up to the
   12-hour session cap. You can close the tab.
2. Expect roughly **10 hours of wall clock** for the first session.
   Realistic budget on P100 at batch 2 / res 384 / gradient
   checkpointing is **~4 seconds per outer step** (GFPGAN's
   StyleGAN2 decoder in the inner loop dominates). At 4 s/step,
   ~9,000 steps fit in 10 hours with margin for warmup and
   checkpoint I/O.
3. The notebook is expected to save a checkpoint to
   `/kaggle/working/voidface-ckpts/` every 500 steps or every 20
   minutes, whichever comes first, and to publish the directory as
   a new Kaggle Dataset at the end of the session:

    ```python
    subprocess.run([
        "kaggle", "datasets", "create", "-p", CKPT_DIR, "--dir-mode", "zip"
    ], check=True)   # first session only
    ```

    Subsequent sessions call `kaggle datasets version` against the
    same slug instead. The default slug in the sample notebook is
    `<your-username>/voidface-checkpoints`.
4. When the batch run finishes (or is cut off at hour 12), the
   dataset version is what you rely on. `/kaggle/working` is wiped
   between sessions; the attached dataset is not.

Do the mid-run push once (say, around step 8,000) as well as at the
end. Kaggle occasionally kills sessions early or gets stuck
uploading in the last minute, and having a step-8k dataset version
means you never lose more than ~30 minutes of training.

---

## 7. Step 5 — Resume in session 2

1. New notebook run (or the same notebook, "Save & Run All" again).
2. This time, **also attach the `voidface-checkpoints` dataset** you
   created at the end of session 1, via **+ Add Input** → search
   your own username. It mounts read-only at
   `/kaggle/input/voidface-checkpoints/`.
3. The training notebook's resume block looks for the newest
   `ckpt_*.pt` under `/kaggle/input/voidface-checkpoints/` and, if
   present, restores G weights, ensemble surrogate weights, both
   optimizer states, the AMP `GradScaler`, and all four RNGs
   (torch, cuda, numpy, python) before entering the training loop.
   The `start_step` printed at the top of the log is your source of
   truth — verify it matches what session 1 left off at.
4. If the resume block prints "no checkpoint found, starting fresh"
   when you expect a resume, stop the run immediately. Either the
   dataset didn't attach (check the sidebar) or the mount path
   inside the notebook doesn't match the dataset slug. Restarting
   from scratch at step 0 with 9,000 steps of quota already burned
   is the most expensive avoidable mistake here.

There is also a config-hash guard: if you edit the training TOML in
a way that changes the resolved config, the resume refuses and
crashes loudly rather than silently loading G into a
subtly-different graph. If you want to change the config partway
through, start a new run under a new checkpoint slug.

---

## 8. Step 6 — Repeat until 30k steps

Expect **five or six sessions across roughly two calendar weeks** to
reach 25-30k steps. Rough budget:

| Session | Steps this session | Cumulative | Wall clock |
|---------|--------------------|------------|------------|
| 1       | 0 → 9,000          | 9k         | ~10 h      |
| 2       | 9,000 → 18,000     | 18k        | ~10 h      |
| 3       | 18,000 → 25,000    | 25k        | ~8 h       |
| 4-6     | polish / iterate   | 25-30k     | ~5-10 h ea |

The free weekly quota is roughly **30 GPU-hours, rolling**. Both
P100 and T4 x2 draw from the same pool. Three ~10-hour sessions
in a calendar week is the ceiling; a fourth in the same week will
be refused with a quota-exhausted error. This is why 30k steps
takes closer to two weeks of elapsed time than one — the pacing is
imposed by the quota, not by per-step throughput.

Practical cadence: run one long session Mon, one Wed, one Fri, let
the quota reset over the weekend, repeat. Do the "Save & Run All"
click before you go to sleep; wake up to a finished session.

---

## 9. Step 7 — Download the final checkpoint

Once the latest `voidface-checkpoints` version has the checkpoint
you want (usually the step-highest `ckpt_*.pt`), pull it locally:

```bash
mkdir -p ./ckpt
kaggle datasets download <your-username>/voidface-checkpoints -p ./ckpt
unzip -o ./ckpt/voidface-checkpoints.zip -d ./ckpt
```

Then run the bench against a fresh test corpus (no images from
training — FFHQ's held-out split, or CelebA-HQ if you used FFHQ for
training):

```bash
voidface bench ./ckpt/ckpt_<step>.pt path/to/FFHQ-test/ \
    --json bench.json \
    --detection-threshold 0.5
```

The bench prints detection ASR, identity cos+1 against the ArcFace
family, PSNR, and SSIM. Save `bench.json` next to the checkpoint —
it is what you compare against in Step 8.

---

## 10. Step 8 — Evaluate and iterate

Running the bench in strict mode surfaces the current-vs-ship-gate
delta directly:

```bash
voidface bench ./ckpt/ckpt_<step>.pt path/to/FFHQ-test/ --strict
```

`--strict` treats the R5.5 thresholds from `tuning.md` as pass/fail
and exits nonzero on the worst offender. Expect the MVP checkpoint
to fail; the useful signal is *which* metric is furthest from gate:

- **Detection ASR too low.** The generator is not defeating
  RetinaFace/SCRFD/YuNet often enough. Bump the detector-family
  weights (`detector`, `scrfd`, `yunet`) as a group, not
  individually — cross-detector transfer is real. Halve the VAE
  weights temporarily to free budget.
- **Identity cos+1 too high.** ArcFace is not being fooled. Give
  the recognizer family more of the loss budget, and confirm
  ArcFace is still ≈0.5 of that family (per `tuning.md`).
- **PSNR/SSIM too low.** The perturbation is too visible. Bump
  `lpips_weight` and `total_var`. This usually costs you detection
  ASR — trade explicitly.
- **All four bad by roughly the same amount.** You are simply
  undertrained. Add more sessions before touching weights.

Change one loss-weight group per iteration. Two changes at once and
you lose the ability to attribute the delta.

---

## 11. Troubleshooting

- **OOM within the first few hundred steps.** Batch too big, or
  gradient checkpointing not on. On P100 16 GB with GFPGAN in the
  inner loop, batch 2 / res 384 / grad-ckpt is the ceiling. Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at the top of
  the notebook to reduce fragmentation-driven OOM on long runs.
- **OOM after several hundred steps ("worked for a while, then
  died").** Fragmentation. Same env var; also `torch.cuda.empty_cache()`
  between bilevel phases if you cannot upgrade PyTorch. Do not
  respond by raising batch — respond by lowering it.
- **NaN loss shortly after start or shortly after resume.** LR too
  high, or the bilevel LPIPS weight is too high, or the AMP scaler
  was not restored. Halve `bilevel_lpips` first; if that does not
  fix it, halve LR; if still NaN, confirm the resume block
  restored `scaler_state`.
- **Silent hang at "starting training", no steps ever logged.**
  Dataset path wrong. The loader is enumerating an empty directory.
  Print `len(dataset)` before the loop starts; it should be in the
  tens of thousands. If it is 0, fix the `/kaggle/input/...` path.
- **Session preempted before hour 12.** Kaggle enforces the 12-hour
  ceiling but sometimes kills earlier. This is why the sample
  notebook checkpoints every 500 steps or 20 minutes and pushes a
  dataset version once mid-run (around hour 9) as well as at the
  end. If you tune checkpoint frequency down for speed, you are
  betting quota on Kaggle not preempting you — don't.
- **"No accelerator" on session start.** Kaggle occasionally
  assigns CPU-only despite the P100 setting. The sample notebook
  asserts `torch.cuda.is_available()` at the top and aborts;
  don't silently run at 100× slower.
- **`num_workers > 2` deadlock.** Kaggle's CPU-side environment
  does not tolerate high worker counts. Use `num_workers=2`,
  `persistent_workers=True`, `pin_memory=True`.
- **Fresh dataset version not showing up when you try to attach
  it.** Kaggle's version-publish is asynchronous; a fresh version
  can take 1-3 minutes before it's attachable to a new notebook.
  Don't panic-republish — wait and refresh.

---

## 12. Realistic expectations

This is an MVP path. It will not hit the R5.5 ship-gate. You should
expect, at 25-30k steps on P100:

- Detection ASR moves from the ~0.03 baseline to roughly
  **0.25-0.40** — clearly non-random, clearly working, clearly not
  60%.
- Identity cos+1 drops from ~1.0 to somewhere in the
  **0.60-0.75** range. Real damage to the ArcFace signal, but well
  short of the ≤ 0.20 gate.
- PSNR and SSIM depend on how you traded the perceptual budget;
  aim for PSNR ≥ 32 dB and SSIM ≥ 0.94 at MVP scale, since your
  attack is weaker there is less reason to spend visible artifact
  budget.

What that is good for: grant applications, research
proofs-of-concept, screenshots, and end-to-end validation that the
loss, dataloader, resume path, and bench numbers all work against
a real trained model rather than a mock. It is not good for
shipping a defensive tool to end users.

Production quality — the ship-gate numbers in `tuning.md` — needs
the full 200k-step run at resolution 512 with batch 4-8 on
paid H100-class hardware. A rented H100 or A100 pod finishes that
in roughly 3-5 hours of wall clock and is, end-to-end, cheaper than
several weeks of Kaggle if you value your time at all. The Kaggle
path exists to get you a real checkpoint at zero dollars, not to
ship the final artifact.
