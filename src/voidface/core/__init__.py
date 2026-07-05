# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Training loop, optimizer, loss composition, and EOT.

Public API:

    :mod:`voidface.core.train`   the top-level training entry point.
    :mod:`voidface.core.loss`    composite loss functions.
    :mod:`voidface.core.optim`   signed-gradient PGD variants.
    :mod:`voidface.core.eot`     expectation over transformation.
    :mod:`voidface.core.pgd`     reference per-image PGD (evaluation only).

See ``Documentation/training/overview.md`` for the high-level picture.
"""

from __future__ import annotations

__all__: list[str] = []
