# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Direct tests for the audit-trail JSON sidecar writer.

The sidecar shape is a public interface — downstream tooling and
research consumers read the fields directly. This locks the schema
so changes are deliberate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from voidface_cli.commands.protect import _write_output_json


def _fake_args(tmp_path: Path, *, use_generator: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        epsilon=12,
        seed=42,
        use_generator=use_generator,
        refine_steps=0,
        targets="detector,recognizer",
        restorers="identity",
        steps=100,
        no_lpips=False,
        face_mask=False,
        semantic_warp=0.0,
        output_json=tmp_path / "sidecar.json",
    )


def test_write_output_json_populates_all_public_fields(tmp_path: Path) -> None:
    clean = torch.zeros(1, 3, 8, 8)
    adv = torch.zeros(1, 3, 8, 8)
    adv[..., 0, 0] = 1.0

    args = _fake_args(tmp_path)
    _write_output_json(args, clean, adv, tmp_path / "out.png")

    payload = json.loads(args.output_json.read_text())
    assert payload["voidface_version"]
    assert payload["output"] == str(tmp_path / "out.png")
    assert isinstance(payload["psnr_db"], float)
    assert isinstance(payload["ssim"], float)
    assert isinstance(payload["l_inf"], float)
    assert payload["epsilon_int_over_255"] == 12
    assert payload["seed"] == 42

    flags = payload["flags"]
    for field in (
        "use_generator", "refine_steps", "targets", "restorers",
        "steps", "no_lpips", "face_mask", "semantic_warp",
    ):
        assert field in flags


def test_write_output_json_records_checkpoint_hash_when_present(tmp_path: Path) -> None:
    ckpt = tmp_path / "gen.pt"
    ckpt.write_bytes(b"fake checkpoint contents")
    args = _fake_args(tmp_path, use_generator=ckpt)

    clean = torch.zeros(1, 3, 4, 4)
    adv = torch.zeros(1, 3, 4, 4)
    _write_output_json(args, clean, adv, tmp_path / "out.png")

    payload = json.loads(args.output_json.read_text())
    assert "generator_checkpoint_sha256" in payload
    assert len(payload["generator_checkpoint_sha256"]) == 64
    assert payload["flags"]["use_generator"] == str(ckpt)


def test_write_output_json_no_hash_without_checkpoint(tmp_path: Path) -> None:
    args = _fake_args(tmp_path, use_generator=None)

    clean = torch.zeros(1, 3, 4, 4)
    adv = torch.zeros(1, 3, 4, 4)
    _write_output_json(args, clean, adv, tmp_path / "out.png")

    payload = json.loads(args.output_json.read_text())
    assert "generator_checkpoint_sha256" not in payload
    assert payload["flags"]["use_generator"] is None


def test_write_output_json_no_hash_when_checkpoint_missing(tmp_path: Path) -> None:
    """--use-generator points at a file that no longer exists."""
    ckpt = tmp_path / "ghost.pt"
    args = _fake_args(tmp_path, use_generator=ckpt)

    clean = torch.zeros(1, 3, 4, 4)
    adv = torch.zeros(1, 3, 4, 4)
    _write_output_json(args, clean, adv, tmp_path / "out.png")

    payload = json.loads(args.output_json.read_text())
    assert "generator_checkpoint_sha256" not in payload


def test_write_output_json_noop_on_non_tensor(tmp_path: Path) -> None:
    args = _fake_args(tmp_path)
    # Pass non-tensor to exercise the early-return guard.
    _write_output_json(args, "not a tensor", None, tmp_path / "out.png")  # type: ignore[arg-type]
    assert not args.output_json.exists()
