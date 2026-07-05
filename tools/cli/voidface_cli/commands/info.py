# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""`voidface info` — print a checkpoint's metadata (with optional --diff)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from voidface.generator.architecture import Voidface, VoidfaceConfig


def _load_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        stored = payload.get("config")
        config = stored if isinstance(stored, VoidfaceConfig) else VoidfaceConfig()
        step = payload.get("step")
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
        config = VoidfaceConfig()
        step = None
    net = Voidface(config)
    net.load_state_dict(state_dict)
    return {
        "path": str(path),
        "size_mb": path.stat().st_size / 1024 / 1024,
        "step": step,
        "params": sum(p.numel() for p in net.parameters()),
        "epsilon": config.epsilon,
        "base_channels": config.base_channels,
        "num_stages": config.num_stages,
        "attention": config.attention_at_bottleneck,
    }


def _run_diff(a: Path, b: Path) -> int:
    """Print a side-by-side comparison of two checkpoints."""
    for path in (a, b):
        if not path.exists():
            print(f"error: checkpoint not found: {path}", file=sys.stderr)
            return 2

    left = _load_summary(a)
    right = _load_summary(b)

    def _fmt(key: str, l_val: object, r_val: object) -> str:
        marker = "  " if l_val == r_val else " *"
        return f"{marker}{key:22s}  {l_val!s:32s}  {r_val!s}"

    print(f"--- diff ---   A: {left['path']}\n              B: {right['path']}")
    print(f"{'':2}{'field':22s}  {'A':32s}  {'B'}")
    for key in (
        "size_mb", "step", "params", "epsilon", "base_channels",
        "num_stages", "attention",
    ):
        print(_fmt(key, left[key], right[key]))
    print("(* marks a difference)")
    return 0


def run(args: argparse.Namespace) -> int:
    """Print a checkpoint's metadata."""
    if args.diff is not None:
        return _run_diff(args.checkpoint, args.diff)

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored = payload.get("config")
        config = stored if isinstance(stored, VoidfaceConfig) else VoidfaceConfig()
        step = payload.get("step")
    else:
        state_dict = payload
        config = VoidfaceConfig()
        step = None

    generator = Voidface(config)
    generator.load_state_dict(state_dict)
    param_count = sum(p.numel() for p in generator.parameters())
    trainable_param_count = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    info = {
        "path": str(args.checkpoint),
        "file_size_bytes": args.checkpoint.stat().st_size,
        "training_step": step,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "config": {
            "epsilon": config.epsilon,
            "base_channels": config.base_channels,
            "num_stages": config.num_stages,
            "attention_at_bottleneck": config.attention_at_bottleneck,
        },
    }

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("--- checkpoint info ---")
        print(f"path:           {info['path']}")
        print(f"file size:      {info['file_size_bytes'] / 1024 / 1024:.2f} MB")
        step_display = "unknown" if step is None else str(step)
        print(f"training step:  {step_display}")
        print(f"params:         {param_count:,}")
        print(f"trainable:      {trainable_param_count:,}")
        print(f"epsilon:        {config.epsilon:.6f}  (~{config.epsilon * 255:.1f}/255)")
        print(f"base channels:  {config.base_channels}")
        print(f"num stages:     {config.num_stages}")
        print(f"attention:      {config.attention_at_bottleneck}")
    return 0
