# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Face detector surrogates for the adversarial ensemble.

Each concrete detector lives in its own module:

    :mod:`voidface.models.detectors.retinaface`
    :mod:`voidface.models.detectors.scrfd`
    :mod:`voidface.models.detectors.yunet`
    :mod:`voidface.models.detectors.mtcnn`

See ``Documentation/models/detectors.md`` for the ensemble weighting
and attack-surface analysis.
"""

from __future__ import annotations

__all__: list[str] = []
