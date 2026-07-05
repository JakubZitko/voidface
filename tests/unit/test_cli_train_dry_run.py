# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface train --dry-run integration test.

Uses `voidface init smoke` to write a config, then runs `voidface
train --dry-run` and verifies it exits cleanly without actually
training. Catches CLI-level regressions in the config resolution
path.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def test_train_dry_run_smoke_config(tmp_path: Path) -> None:
    from voidface_cli.main import main

    # 1. Prepare a tiny dataset directory.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (32, 32), (i * 100, 100, 100)).save(data_dir / f"face_{i}.png")

    # 2. Generate a smoke config, override the [data].directory to
    # point at our synthetic dataset.
    cfg_path = tmp_path / "cfg.toml"
    rc = main(["init", "smoke", "-o", str(cfg_path)])
    assert rc == 0

    # Rewrite the directory line to point at tmp.
    text = cfg_path.read_text()
    text = text.replace(
        'directory = "samples/images"',
        f'directory = "{data_dir}"',
    )
    cfg_path.write_text(text)

    # 3. Run train --dry-run. All targets disabled in the smoke preset
    # so no external weights are fetched.
    rc = main(["train", str(cfg_path), "--device", "cpu", "--dry-run"])
    assert rc == 0
