# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""SHA-256 integrity gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from voidface.util.checksum import IntegrityError, compute_sha256, verify_sha256


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def test_compute_sha256_matches_known_value(tmp_path: Path) -> None:
    file = tmp_path / "hello.bin"
    _write(file, b"hello world")
    # Known SHA-256 of "hello world".
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert compute_sha256(file) == expected


def test_verify_sha256_passes_on_match(tmp_path: Path) -> None:
    file = tmp_path / "hello.bin"
    _write(file, b"hello world")
    verify_sha256(
        file,
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
    )


def test_verify_sha256_case_insensitive(tmp_path: Path) -> None:
    file = tmp_path / "hello.bin"
    _write(file, b"hello world")
    verify_sha256(
        file,
        "B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9",
    )


def test_verify_sha256_raises_on_mismatch(tmp_path: Path) -> None:
    file = tmp_path / "hello.bin"
    _write(file, b"tampered")
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        verify_sha256(
            file,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )


def test_verify_sha256_none_skips_with_warning(tmp_path: Path) -> None:
    file = tmp_path / "hello.bin"
    _write(file, b"anything")
    with pytest.warns(UserWarning, match="integrity check skipped"):
        verify_sha256(file, None)
