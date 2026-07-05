# Voidface coding style

This document is the source of truth for how Voidface code is written. It is
short on purpose: enforceable rules, not opinions. Every rule is either
already checked by ruff/mypy or checked by a script under `scripts/`.

If you disagree with a rule, open an issue proposing a change to *this file*,
not to your patch.

---

## The zero rule: SPDX header on every source file

Every source file — Python, Rust, TypeScript, shell, Makefile — begins with:

```
# SPDX-License-Identifier: MIT
```

or the language's comment equivalent. `scripts/check-license-headers.sh`
rejects any file lacking one. No exceptions.

---

## Files, directories, and packages

- **One concern per file.** When a file passes ~500 lines it is time to
  split. Prefer subsystem-shaped files (`detectors/retinaface.py`) over
  role-shaped files (`utils.py`, `helpers.py`, `misc.py`).
- **Directories are subsystems.** A directory answers "what part of the
  system is this?" (`models/`, `attacks/`, `core/`), not "what kind of
  thing lives here?" (`controllers/`, `services/`, `managers/`).
- **No top-level god-file.** `voidface/__init__.py` re-exports the small
  public API and nothing else. Real code lives in a subsystem.
- **Python packages have `__init__.py`** with an explicit `__all__`. If
  it is not in `__all__`, it is private.

---

## Naming

| Kind                | Convention          | Example                            |
| ------------------- | ------------------- | ---------------------------------- |
| Module              | `snake_case`        | `retinaface.py`                    |
| Package             | `snake_case`        | `models/detectors/`                |
| Function            | `snake_case`        | `compute_arcface_loss`             |
| Method              | `snake_case`        | `.encode_face`                     |
| Class / Protocol    | `PascalCase`        | `EnsembleTarget`, `RetinaFace`     |
| Constant            | `UPPER_SNAKE_CASE`  | `DEFAULT_LATENT_CHANNELS`          |
| Type alias          | `PascalCase`        | `FloatTensor`                      |
| Private / internal  | leading `_`         | `_align_landmarks`                 |

Do not abbreviate. `attack_success_rate`, not `asr`. `configuration`, not
`cfg`. Exceptions: well-known ML acronyms (`vae`, `pgd`, `clip`, `eot`).

---

## Type hints

- Every public function has a full type signature. No implicit `Any`.
- Use PEP 585 built-in generics (`list[int]`, `dict[str, Tensor]`).
- Use PEP 604 union syntax (`int | None`), not `Optional[int]`.
- Prefer `Protocol` for interfaces; prefer `TypedDict` or dataclasses for
  structured records. Avoid inheritance for interface reuse.
- `from __future__ import annotations` is the first import in every file.
- Type-only imports go inside `if TYPE_CHECKING:`.

---

## Docstrings

Every public class, function, and method has a docstring in Google style.
The first line is a single sentence summary. Args, Returns, Raises are
each documented when non-obvious.

```python
def blend_masks(a: Tensor, b: Tensor, weight: float) -> Tensor:
    """Blend two 4D masks with a fixed weight.

    Args:
        a: The first mask, shape ``(N, 1, H, W)``.
        b: The second mask, same shape as ``a``.
        weight: Blend fraction in ``[0.0, 1.0]``. ``0`` returns ``a``.

    Returns:
        The blended mask with the same shape as the inputs.
    """
```

Do not write docstrings for private (`_`-prefixed) functions unless the
behavior is non-obvious.

---

## Imports

```python
# 1. __future__
from __future__ import annotations

# 2. Standard library
import math
from pathlib import Path
from typing import TYPE_CHECKING

# 3. Third-party
import torch
from torch import Tensor, nn

# 4. Local (voidface.*), grouped by subsystem
from voidface.models.base import EnsembleTarget
from voidface.util.image import load_face_crop

if TYPE_CHECKING:
    from voidface.core.loss import LossWeights
```

ruff enforces this ordering. Do not hand-tweak it.

---

## Errors

- Raise the most specific built-in exception that fits, or a subclass of
  it. Never raise bare `Exception`.
- Error messages are complete sentences, ending in a period, describing
  what went wrong and — when useful — what the caller should do.
- Never swallow an exception silently. If you catch, you either re-raise
  or handle it with a comment explaining why.

---

## Configuration

- Hard-coded values are named constants at the top of the module (or in a
  `constants.py` inside the subsystem). No magic numbers scattered
  through function bodies.
- Runtime configuration is passed in explicitly — never via a global
  singleton, environment variable at import time, or module-level state.
- Configuration files use TOML. See `samples/configs/`.

---

## State

- Modules do not perform work at import time. Loading model weights,
  reading files, and touching the network happen inside functions.
- Prefer pure functions. When state is unavoidable, encapsulate it in a
  class whose instance is passed explicitly.
- No global mutable state. Full stop.

---

## Comments

- Explain *why*, not *what*. If the *what* needs a comment, the code is
  probably unclear.
- A short one-line comment above a subtle line is fine. A multi-paragraph
  comment inside a function almost always signals code that should have
  been split.
- Do not leave TODOs without a name: `# TODO(subsystem-owner): ...`.

---

## Tests

- Every non-trivial function has at least one unit test in `tests/unit/`.
- Integration tests live in `tests/integration/` and may hit real models.
- Tests use `pytest`; fixtures live in `tests/conftest.py` or subsystem
  `conftest.py`.
- Slow tests are marked `@pytest.mark.slow`. GPU tests
  `@pytest.mark.gpu`. Network tests `@pytest.mark.network`.

---

## Line length and formatting

- 100 columns.
- Formatting is done by `ruff format`. Do not hand-format. Do not fight it.
- `make format` reformats the whole tree. `make check-format` fails CI.

---

## Commits

- Subject line is imperative present: "Add ArcFace target", not
  "Added ArcFace target".
- Subject line is at most 72 characters.
- Body wraps at 72 characters and explains the *why*.
- Prefix the subject with a subsystem tag: `core: ...`, `models/detectors: ...`.
- Sign off non-trivial commits with `Signed-off-by:`.

---

## What is not enforced by tooling

The rules above are mechanically checkable. Two rules are not, and they
matter more than any of the above:

1. **Read the surrounding code before you write.** Match its style, its
   naming, its abstraction level. Do not import an unrelated pattern
   from a different subsystem just because you like it better.
2. **Delete more than you add.** Every patch that removes complexity is
   more valuable than one that adds it. When in doubt, do less.
