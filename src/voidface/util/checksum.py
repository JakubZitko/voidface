# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# File-integrity checks for downloaded weights.
#
# The R4 packaging critic flagged that Voidface loads several large
# torch pickles from community HuggingFace mirrors that can disappear,
# rebase, or in the worst case be replaced by a malicious payload. The
# defensive move is a SHA-256 gate: every downloaded weight file is
# hashed and compared against a pinned expected value before it is
# handed to torch.load. A mismatch raises loudly.
#
# The full R5 answer is to self-host every weight under a Voidface HF
# org with commit SHA pinning. This module is the first stone in that
# path — it lets us start pinning known-good hashes now.

"""File-integrity helpers for downloaded weights."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["IntegrityError", "compute_sha256", "verify_sha256"]

_READ_CHUNK = 1024 * 1024


class IntegrityError(RuntimeError):
    """Raised when a file's SHA-256 does not match the expected value."""


def compute_sha256(path: Path) -> str:
    """Return the lowercase-hex SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_READ_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> None:
    """Assert that the SHA-256 of ``path`` matches ``expected``.

    Args:
        path: The file to hash.
        expected: The lowercase-hex expected hash. If ``None``, the
            check is skipped (used during first-time bring-up when
            the hash is not yet known — every skip logs a warning
            from :func:`voidface.util.log.get_logger`).

    Raises:
        IntegrityError: When the actual hash differs from ``expected``.
    """
    if expected is None:
        import warnings

        warnings.warn(
            f"SHA-256 integrity check skipped for {path.name} — no expected hash "
            "provided. Add the computed value to the caller's known-good table.",
            stacklevel=2,
        )
        return
    actual = compute_sha256(path)
    if actual.lower() != expected.lower():
        msg = (
            f"SHA-256 mismatch on {path.name}: expected {expected}, got {actual}. "
            "The checkpoint has been modified since the pinned value was recorded. "
            "Refusing to load it. If this is a known upstream update, verify the "
            "new file out-of-band and update the pinned hash in the caller."
        )
        raise IntegrityError(msg)
