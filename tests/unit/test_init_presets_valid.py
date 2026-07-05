# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Verify every `voidface init` preset produces a config that
`voidface config-check` accepts.

Regression guard: adding a preset (or a target/restorer name change)
without updating the corresponding preset TOML surfaces here.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("preset", ["smoke", "full", "detector-only", "recognizer-only"])
def test_preset_config_passes_check(preset: str, tmp_path: Path) -> None:
    from voidface_cli.main import main

    # Prep a synthetic data directory so [data].directory resolves
    # to a real path when we swap it in.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "keep.png").write_bytes(b"")

    cfg = tmp_path / "cfg.toml"
    rc = main(["init", preset, "-o", str(cfg)])
    assert rc == 0

    text = cfg.read_text()
    # Swap the placeholder ~/data/ffhq directory to the tmp dir so
    # config-check does not reject on missing data directory.
    text = text.replace('directory = "~/data/ffhq"', f'directory = "{data_dir}"')
    text = text.replace('directory = "samples/images"', f'directory = "{data_dir}"')
    cfg.write_text(text)

    rc = main(["config-check", str(cfg)])
    # 0 = OK; some presets legitimately trigger warnings but no
    # errors (WARN != FAIL). Assert we got a validate-clean exit.
    assert rc == 0
