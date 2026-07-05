# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Direct tests for the --restorers spec parser.

Guards the (name, weight) pair extraction logic that main.py's
_cmd_protect and the training loop both depend on. The parser used
to live in main.py; it was extracted to voidface_cli.commands.protect
in R7.65.
"""

from __future__ import annotations

from voidface_cli.commands.protect import _parse_restorer_spec


def test_single_name_default_weight() -> None:
    assert _parse_restorer_spec("identity") == [("identity", 1.0)]


def test_single_name_explicit_weight() -> None:
    assert _parse_restorer_spec("sd15-vae:0.7") == [("sd15-vae", 0.7)]


def test_multiple_names_mixed_defaults() -> None:
    result = _parse_restorer_spec("identity,sd15-vae:0.5,gfpgan:0.3")
    assert result == [
        ("identity", 1.0),
        ("sd15-vae", 0.5),
        ("gfpgan", 0.3),
    ]


def test_whitespace_between_tokens_is_stripped() -> None:
    assert _parse_restorer_spec(" identity ,  gfpgan:0.4 ") == [
        ("identity", 1.0),
        ("gfpgan", 0.4),
    ]


def test_empty_tokens_ignored() -> None:
    assert _parse_restorer_spec("identity,,gfpgan:0.2") == [
        ("identity", 1.0),
        ("gfpgan", 0.2),
    ]


def test_unknown_name_returns_none() -> None:
    # 'lolcat' is not in ALLOWED_RESTORERS.
    assert _parse_restorer_spec("identity,lolcat") is None


def test_non_numeric_weight_returns_none() -> None:
    assert _parse_restorer_spec("identity:heavy") is None


def test_empty_input_returns_none() -> None:
    assert _parse_restorer_spec("") is None
    assert _parse_restorer_spec("   ") is None
    assert _parse_restorer_spec(",,,") is None


def test_all_three_allowed_names() -> None:
    result = _parse_restorer_spec("identity,sd15-vae,gfpgan")
    assert result is not None
    names = {name for name, _ in result}
    assert names == {"identity", "sd15-vae", "gfpgan"}
