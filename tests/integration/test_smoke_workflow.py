# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""End-to-end smoke workflow.

Walks a realistic newbie journey from init through info without ever
touching the network:

  1. voidface init smoke -o cfg.toml
  2. Patch the dataset directory to point at synthetic images.
  3. voidface config-check cfg.toml
  4. voidface train cfg.toml --dry-run --device cpu

Each step must exit 0.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def test_smoke_workflow_end_to_end(tmp_path: Path) -> None:
    from voidface_cli.main import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), (i * 60, 96, 128)).save(data_dir / f"f{i}.png")

    cfg_path = tmp_path / "cfg.toml"

    rc = main(["init", "smoke", "-o", str(cfg_path)])
    assert rc == 0, "voidface init smoke failed"

    text = cfg_path.read_text()
    text = text.replace(
        'directory = "samples/images"',
        f'directory = "{data_dir}"',
    )
    cfg_path.write_text(text)

    rc = main(["config-check", str(cfg_path)])
    assert rc == 0, "voidface config-check failed"

    rc = main(["train", str(cfg_path), "--dry-run", "--device", "cpu"])
    assert rc == 0, "voidface train --dry-run failed"
