# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface verify` — check a release bundle against CHECKSUMS.sha256."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voidface.util.checksum import compute_sha256


def run(args: argparse.Namespace) -> int:
    """Verify a release bundle's artifacts against its CHECKSUMS.sha256."""
    bundle = args.bundle_dir
    checksums_path = bundle / "CHECKSUMS.sha256"
    if not checksums_path.exists():
        print(
            f"error: {checksums_path} not found (not a voidface bundle?)",
            file=sys.stderr,
        )
        return 2

    text = checksums_path.read_text()
    mismatches: list[str] = []
    matches: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Format: "<sha>  <bundle-name>/<file>"
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_sha, name = parts
        # File is relative to the bundle_dir; name may include the
        # bundle directory prefix from `voidface package`.
        file_name = Path(name).name
        file_path = bundle / file_name
        if not file_path.exists():
            mismatches.append(f"  MISSING  {file_name}")
            continue
        actual_sha = compute_sha256(file_path)
        if actual_sha.lower() != expected_sha.lower():
            mismatches.append(
                f"  MISMATCH {file_name}  expected {expected_sha[:16]}...  "
                f"got {actual_sha[:16]}..."
            )
        else:
            matches.append(f"  OK       {file_name}")

    print(f"--- verify {bundle} ---")
    for line in matches + mismatches:
        print(line)
    if mismatches:
        print(f"FAIL: {len(mismatches)} artifact(s) failed verification")
        return 1
    print(f"OK: {len(matches)} artifact(s) verified")
    return 0
