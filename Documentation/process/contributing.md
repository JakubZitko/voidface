# Contributing to Voidface

Voidface is a research project first and a shipped product second. Both
kinds of contribution matter: new attack targets, better training
objectives, robustness experiments, model improvements, packaging fixes,
documentation. Read this before you send a patch.

---

## Before you write code

1. Read `Documentation/architecture.md` and `Documentation/coding-style.md`.
2. Read the subsystem doc for the area you plan to touch
   (`Documentation/models/detectors.md`, etc.).
3. Look at the corresponding subsystem in `MAINTAINERS`. Reach out to
   the maintainer before starting a large change.
4. If your change requires a new dependency, propose it in an issue
   first. Voidface aims to stay dependency-light on the runtime side.

---

## The patch cycle

    make install
    make lint
    make test
    make check-format
    make check-licenses

Every one of these must pass before you send a patch. CI runs them again
on every pull request; if they fail there, they will not merge.

New non-trivial code needs a test. Bug fixes need a regression test.
Documentation-only changes are exempt.

---

## Commit messages

Follow the format described in `Documentation/coding-style.md`. In short:

    subsystem: short imperative subject (<=72 chars)

    Body explaining the *why*, wrapped at 72 chars. When the change is
    tied to a research result or an external bypass, cite it here.

    Signed-off-by: Your Name <you@example.com>

The subject subsystem tag matches the top-level directory or subsystem in
`MAINTAINERS`, e.g. `core:`, `models/detectors:`, `attacks:`, `docs:`.

---

## Pull request checklist

- [ ] Every source file touched keeps its SPDX header.
- [ ] `make lint`, `make test`, `make check-format`,
      `make check-licenses` all pass.
- [ ] Tests added or updated for the change.
- [ ] `Documentation/` updated where behavior or interface changed.
- [ ] `MAINTAINERS` still accurate.
- [ ] Commit history is clean; no fixup commits merged.

---

## Reproducibility

Voidface must be reproducible. If you add a training experiment, include:

- The exact config file under `samples/configs/`.
- A short `Documentation/training/experiments/<slug>.md` describing what
  ran, what hardware, what result.
- Seeds. Log them.

---

## What we do not accept

- Weights or checkpoints in the git tree. Weights are distributed via
  release artifacts, not `git`.
- Datasets in the git tree.
- Code that hard-codes personal paths, keys, or credentials.
- Contributions whose license is incompatible with MIT.

---

## Getting help

Ask a question by opening an issue. Do not email maintainers directly
for questions; do email them for anything covered by
`Documentation/process/security.md`.
