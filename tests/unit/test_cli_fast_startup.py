# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface --help` must not import torch.

The whole point of the R7.54-R7.65 delegator layout is that
subcommands import their heavy dependencies (torch, diffusers,
etc.) only when actually invoked. `voidface --help` and
`voidface --version` must stay lightweight.

This test guards that promise. If a future refactor accidentally
lifts a torch import to voidface_cli.main's top level, `voidface
--help` will pay the ~1-2 second torch import cost every time,
and this test fires.
"""

from __future__ import annotations

import subprocess
import sys


def test_help_does_not_import_torch() -> None:
    """Run `voidface --help` in a subprocess and check no torch load."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; "
         "import voidface_cli.main; "
         "print('torch' in sys.modules, 'torch_loaded_at_import')"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Line format: "True torch_loaded_at_import" if torch was imported.
    imported = result.stdout.strip().startswith("True")
    assert not imported, (
        "voidface_cli.main pulled in `torch` at module import. This slows "
        "`voidface --help` by ~1-2 seconds. Keep torch imports inside the "
        "individual `commands/*.py` modules and out of main.py's argparse "
        "wiring."
    )


def test_main_module_loads_fast() -> None:
    """A cold subprocess import of voidface_cli.main runs under 5s.

    This is a soft-guard sanity check. On a warm machine it's ~200ms.
    """
    import time
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", "import voidface_cli.main"],
        capture_output=True,
        text=True,
        check=True,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, (
        f"voidface_cli.main import took {elapsed:.2f}s (limit 5.0s). "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
