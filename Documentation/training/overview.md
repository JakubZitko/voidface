# Training overview

Voidface training produces one artifact: the generator `G`. Everything in
`src/voidface/core/` and `src/voidface/generator/` exists to train `G`.

The training loop, at the highest level:

    for step in range(N):
        clean = sample_face_photo(dataset)          # data/
        cloaked = G(clean)                          # generator/
        strip = restorer(transform(cloaked))        # models/restorers/ + eot
        loss = ensemble_loss(strip, clean)          # core/loss.py
             + perceptual_loss(cloaked, clean)      # eval/perceptual.py
        loss.backward()
        optimizer.step()

The unusual property is the `restorer(...)` step. Prior work computed the
loss on `cloaked` directly. We compute it on the restored version of the
cloak, which forces `G` to place its perturbation in a signal that survives
face restoration. See `bilevel-adversarial.md`.

The three pillars of the training loop each have their own document:

- `bilevel-adversarial.md` — restorer-in-the-loop training.
- `eot.md` — expectation over transformation (JPEG, resize, blur, etc.).
- `experiments/` — reproducibility records for each training run.

The composite loss balances 13 ensemble terms against a perceptual budget.
The exact weights live in `samples/configs/train_full.toml` and are tuned
via the procedure documented in `Documentation/training/tuning.md` (to be
written after phase R1).

Training runs on a single H100 or A100 for the current scale (~5–10M
parameter `G`, ~200K training faces, 300K steps). Multi-GPU support is
planned but not required for the baseline.
