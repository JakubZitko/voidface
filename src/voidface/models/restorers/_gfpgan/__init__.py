# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2021 Tencent ARC
# Copyright (c) 2026 Voidface contributors (modifications)

"""Vendored GFPGAN v1-clean architecture.

Splits the upstream files into ``stylegan2_clean.py`` and
``gfpgan_clean.py``, both with a small shim for the two basicsr
coupling points (``ARCH_REGISTRY.register`` no-op and an inlined
``default_init_weights``). No functional changes to the arch.

See LICENSES/Apache-2.0-gfpgan.txt for the full upstream license.
"""

from __future__ import annotations

__all__: list[str] = []
