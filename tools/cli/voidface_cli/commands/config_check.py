# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface config-check` — validate a training TOML before a real run.

Catches typos and missing directories before the user waits for an
expensive checkpoint fetch to notice their [data].directory points
at ``~/typo/ffhq/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voidface.util.config import load_config
from voidface_cli.common import ALLOWED_RESTORERS, ALLOWED_TARGETS


def run(args: argparse.Namespace) -> int:
    """Parse a training config, validate its shape, print a summary."""
    if not args.config.exists():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: could not parse TOML: {exc}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    data_section = config.get("data", {})
    if "directory" not in data_section:
        errors.append("[data].directory is required")
    else:
        data_dir = Path(str(data_section["directory"])).expanduser()
        if not data_dir.exists():
            errors.append(f"[data].directory does not exist: {data_dir}")
        elif not data_dir.is_dir():
            errors.append(f"[data].directory is not a directory: {data_dir}")

    targets_conf = config.get("targets", {})
    for name in targets_conf:
        if name not in ALLOWED_TARGETS:
            errors.append(
                f"unknown [targets.{name}]; allowed: {sorted(ALLOWED_TARGETS)}"
            )

    restorers_conf = config.get("restorers", {})
    for name in restorers_conf:
        if name not in ALLOWED_RESTORERS:
            errors.append(
                f"unknown [restorers].{name}; allowed: {sorted(ALLOWED_RESTORERS)}"
            )

    enabled_targets = [
        name for name, t in targets_conf.items() if t.get("enabled", False)
    ]
    if not enabled_targets:
        warnings.append(
            "no [targets.*].enabled = true — training will only optimize the TV "
            "regularizer, which drives delta to zero (nothing to attack). Enable "
            "at least one target."
        )

    if "gfpgan" in restorers_conf and float(restorers_conf["gfpgan"]) > 0 \
            and not targets_conf.get("detector", {}).get("enabled", False):
        warnings.append(
            "[restorers].gfpgan > 0 but [targets.detector].enabled = false — "
            "the CLI train path will auto-load RetinaFace for the aligner "
            "but the ensemble will not score detection suppression."
        )

    if "sd15-vae" in restorers_conf and float(restorers_conf["sd15-vae"]) > 0 \
            and not targets_conf.get("vae", {}).get("enabled", False):
        errors.append(
            "[restorers].sd15-vae > 0 requires [targets.vae].enabled = true "
            "(the restorer shares the encoder with the target)."
        )

    experiment = config.get("experiment", {})
    print("--- config check ---")
    print(f"path:              {args.config}")
    print(f"name:              {experiment.get('name', '(unset)')}")
    print(f"seed:              {experiment.get('seed', 0)}")
    print(f"steps:             {experiment.get('steps', '(default)')}")
    if data_section:
        print(f"data.directory:    {data_section.get('directory', '(unset)')}")
        print(f"data.resolution:   {data_section.get('resolution', 256)}")
        print(f"data.batch_size:   {data_section.get('batch_size', 4)}")
    print(f"enabled targets:   {enabled_targets or '(none — nothing to attack)'}")
    print(f"restorers:         {list(restorers_conf) or '(identity only)'}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1
    print("OK")
    return 0
