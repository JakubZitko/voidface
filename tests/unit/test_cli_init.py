# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface init subcommand tests."""

from __future__ import annotations

from pathlib import Path


def test_init_smoke_writes_valid_toml(tmp_path: Path, capsys) -> None:
    import tomllib

    from voidface_cli.main import main

    out = tmp_path / "cfg.toml"
    rc = main(["init", "smoke", "-o", str(out)])
    assert rc == 0
    assert out.exists()
    parsed = tomllib.loads(out.read_text())
    assert parsed["experiment"]["name"] == "smoke"
    assert "restorers" in parsed


def test_init_full_writes_valid_toml(tmp_path: Path, capsys) -> None:
    import tomllib

    from voidface_cli.main import main

    out = tmp_path / "cfg.toml"
    rc = main(["init", "full", "-o", str(out)])
    assert rc == 0
    parsed = tomllib.loads(out.read_text())
    assert parsed["experiment"]["name"] == "full-ensemble"
    # Full config should enable multiple targets.
    assert parsed["targets"]["detector"]["enabled"] is True
    assert parsed["targets"]["recognizer"]["enabled"] is True
    assert parsed["restorers"]["gfpgan"] == 0.60


def test_init_to_stdout(tmp_path: Path, capsys) -> None:
    from voidface_cli.main import main

    rc = main(["init", "detector-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[targets.detector]" in out
    assert 'name = "detector-only"' in out
