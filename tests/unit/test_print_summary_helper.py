# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Direct tests for the terminal-summary printer used by `voidface protect`.

Locks the human-readable output format so people who parse the
"--- summary ---" block (users piping into grep, wrappers, docs)
don't get surprised by silent field changes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from voidface_cli.commands.protect import _print_summary


def test_summary_prints_expected_labels(capsys) -> None:
    clean = torch.zeros(1, 3, 8, 8)
    adv = torch.zeros(1, 3, 8, 8)
    adv[..., 0, 0] = 0.1
    _print_summary(clean, adv, Path("/tmp/protected.png"))
    out = capsys.readouterr().out
    assert "--- summary ---" in out
    assert "output:" in out
    assert "PSNR:" in out
    assert "SSIM:" in out
    assert "L-inf:" in out
    assert "/tmp/protected.png" in out


def test_summary_noop_on_non_tensor(capsys) -> None:
    _print_summary("not a tensor", None, Path("/tmp/x.png"))  # type: ignore[arg-type]
    assert capsys.readouterr().out == ""


def test_summary_reports_correct_linf(capsys) -> None:
    clean = torch.zeros(1, 3, 4, 4)
    adv = clean.clone()
    # One-pixel spike of magnitude 5/255.
    adv[..., 0, 0] = 5.0 / 255.0
    _print_summary(clean, adv, Path("/tmp/x.png"))
    out = capsys.readouterr().out
    # L-inf field reports out_of_255 = 5.
    assert "L-inf:   5.00/255" in out or "L-inf:   5.0/255" in out
