# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface config-check subcommand tests."""

from __future__ import annotations

from pathlib import Path


def _write_config(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n")


def test_valid_config_returns_zero(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "keep.png").write_bytes(b"")

    cfg = tmp_path / "cfg.toml"
    _write_config(
        cfg,
        f"""
[experiment]
name = "test"
steps = 10

[data]
directory = "{data_dir}"
resolution = 64
batch_size = 1

[targets.detector]
enabled = true

[restorers]
identity = 1.0
""",
    )
    rc = main(["config-check", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_missing_directory_errors(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    cfg = tmp_path / "cfg.toml"
    _write_config(
        cfg,
        """
[experiment]
name = "test"

[data]
directory = "/tmp/definitely/does/not/exist/voidface-test"

[targets.detector]
enabled = true
""",
    )
    rc = main(["config-check", str(cfg)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "does not exist" in out


def test_no_targets_enabled_warns(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = tmp_path / "cfg.toml"
    _write_config(
        cfg,
        f"""
[data]
directory = "{data_dir}"
""",
    )
    rc = main(["config-check", str(cfg)])
    # No targets is a WARN, not an ERROR — training runs, it just
    # doesn't attack anything (TV-only, delta -> 0).
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN" in out


def test_unknown_target_name_errors(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = tmp_path / "cfg.toml"
    _write_config(
        cfg,
        f"""
[data]
directory = "{data_dir}"

[targets.deeftector]  # typo
enabled = true
""",
    )
    rc = main(["config-check", str(cfg)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "unknown [targets.deeftector]" in out


def test_missing_config_file_errors(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    rc = main(["config-check", str(tmp_path / "does_not_exist.toml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err
