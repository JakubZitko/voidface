# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Benchmark harness — scaffold. Populated after the eval subsystem lands.

"""End-to-end attack-success-rate and perceptual benchmark.

Run:

    uv run python scripts/benchmark.py --config samples/configs/bench.toml

Reports ASR per ensemble target and PSNR / SSIM / LPIPS aggregates.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "voidface benchmark scaffold — subsystem not implemented yet. "
        "See Documentation/architecture.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
