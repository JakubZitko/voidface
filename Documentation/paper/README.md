# Voidface paper

`Documentation/paper.md` is the v0.1 technical writeup of the voidface project.

## What it's for

- Grant applications (Foresight AI Safety Nodes, HuggingFace Community GPU
  Grant discussion body, NVIDIA Academic Grant, NAIRR Pilot).
- Technical reference for contributors, collaborators, and reviewers.
- Public-facing "what is this project" document — linked from the repo
  README, HF Space README, and any external outreach.

## Status

**v0.1 draft, 2026-07-06.** Every quantitative claim is a projection, a
threshold, or a metric definition — not a measurement. Voidface has not
yet been trained on real face data at scale. All numeric figures will be
re-reported after the R5.5 production training run completes; that will
be v0.2 of this paper.

Anything that reads like a measurement should have "PROJECTED" or "TARGET"
next to it. If it doesn't, that's a documentation bug — please file an
issue on GitHub.

## Figures

The six figures in `figures/` are regenerated with:

    bash Documentation/paper/figures/generate.sh

The script runs on the voidface Python environment (`uv run python …`) and
uses the real `voidface.attacks.iris.iris_region_mask` for Figure 1 —
the iris mask on the canonical FFHQ template. The other five figures
are matplotlib block diagrams and comparison bars.

Regenerate whenever:

- The iris `radius_frac` default changes (Figure 1 recomputes)
- Ship-gate thresholds move in `Documentation/process/release.md` (Figure 5)
- The ensemble surrogate set changes (Figure 3)
- Deploy artifacts change size/set (Figure 4)
- After R5.5 completes with real measured numbers (Figure 6 stops being
  projected)

## Citing voidface

Until a formal preprint lands, cite the paper by its repo URL and commit
SHA:

    @misc{voidface2026,
      author       = {Zítko, Jakub},
      title        = {Voidface: A Bilevel Adversarial Perturbation Framework
                      for Face-Swap Defense},
      year         = {2026},
      howpublished = {\url{https://github.com/JakubZitko/voidface}},
      note         = {v0.1 draft; MIT-licensed}
    }

After R5.5 + v0.2 lands, this section will be replaced with the
arXiv / venue citation.

## Contributing to the paper

Section drafts started as parallel LLM-authored drafts based on live
literature research (July 2026), then were hand-assembled and normalised.
The drafts remain at `/tmp/voidface-paper-parts/*.md` while the assembly
is in flight; those are ephemeral. Canonical source is `paper.md`.

Corrections, missed citations, factual errors, and clarifications are
welcome via GitHub Issues or PRs against `Documentation/paper.md`.

Deliberate constraints for future edits:

1. **Do not fabricate quantitative results.** Voidface v0.1 has no
   trained-model measurements. Present numbers as expected/target/projected
   with explicit caveats until v0.2.
2. **Cite honestly.** If you can't verify a citation, mark it `[unverified]`
   and open an issue.
3. **Preserve the honest-limits framing.** `Documentation/limits.md` and
   the paper's Section 7 (Limits) are load-bearing — reviewers, journalists,
   and downstream users read them to calibrate expectations. Do not soften.
4. **Keep the paper reproducible.** Every figure regenerates from
   `generate_figures.py`; every method claim maps to a file path under
   `src/voidface/` or `tools/cli/voidface_cli/`.
