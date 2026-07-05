# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""voidface info subcommand tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _write_checkpoint(path: Path, step: int = 12345) -> None:
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    config = VoidfaceConfig(base_channels=8)
    generator = Voidface(config).eval()
    torch.save({"step": step, "state_dict": generator.state_dict(), "config": config}, path)


def test_info_prints_human_summary(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt)
    rc = main(["info", str(ckpt)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "checkpoint info" in out
    assert "training step:  12345" in out
    assert "epsilon" in out


def test_info_json_output(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    ckpt = tmp_path / "gen.pt"
    _write_checkpoint(ckpt, step=99)
    rc = main(["info", str(ckpt), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["training_step"] == 99
    assert parsed["param_count"] > 0
    assert "epsilon" in parsed["config"]


def test_info_missing_checkpoint_errors(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    rc = main(["info", str(tmp_path / "does_not_exist.pt")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_info_diff_prints_side_by_side(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    from voidface_cli.main import main

    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    _write_checkpoint(a, step=100)
    _write_checkpoint(b, step=200)

    rc = main(["info", str(a), "--diff", str(b)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "diff" in out
    assert "step" in out
    # step differs; expect the marker.
    assert "*" in out
