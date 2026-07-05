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

---

## Adding a new CLI subcommand

The CLI is deliberately partitioned: `main.py` owns only argparse
wiring + a dispatch dict + a handful of one-line delegators. Every
subcommand's actual logic lives in `tools/cli/voidface_cli/commands/<name>.py`.

To add a subcommand `voidface foo`:

1. Write `tools/cli/voidface_cli/commands/foo.py`. Give it exactly
   one public function `run(args: argparse.Namespace) -> int`. If
   `foo` shares helpers with other subcommands, put those helpers
   in `voidface_cli/common.py` (never in `main.py`).

2. In `main.py::_build_parser`, add the argparse subparser with
   arguments and help text.

3. In `main.py`, add a four-line delegator:

        def _cmd_foo(args: argparse.Namespace) -> int:
            """Extracted to voidface_cli.commands.foo."""
            from voidface_cli.commands import foo as _foo_cmd

            return _foo_cmd.run(args)

4. In `main.py::main()`, add `"foo": _cmd_foo,` to the dispatch
   dict AND `"foo"` to `test_dispatch_covers_all_documented_subcommands`
   in `tests/unit/test_cli_dispatch_completeness.py`.

5. Add tests in `tests/unit/test_cli_foo.py`. If the subcommand
   has non-trivial helpers, write dedicated tests for them
   (see `tests/unit/test_parse_restorer_spec.py` for the pattern).

6. Update `Documentation/status.md` with the new subcommand under
   the CLI section.

The dispatch-completeness test will fail loudly if you miss step 4.

