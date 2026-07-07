#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Regenerate voidface paper figures. Run from anywhere; the script cd's to repo root.
#
# First time:  chmod +x Documentation/paper/figures/generate.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."
uv run python Documentation/paper/figures/generate_figures.py
