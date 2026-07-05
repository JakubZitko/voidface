# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Identity restorer. Included in the restorer sample distribution so
# that the generator still learns to defeat un-restored pipelines,
# not only pipelines that pass through a heavy face-restorer.
#
# The identity restorer is a valid default even before real restorers
# land — it lets the bilevel training loop compose the ``targets(
# restorer(x + delta))`` shape without special-casing "no restorer".

"""Identity pass-through restorer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from voidface.models.restorers.base import RestorerSpec

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["IdentityRestorer"]


class IdentityRestorer:
    """The identity map. Preserves ``image`` bit-for-bit."""

    spec = RestorerSpec(name="identity", expects_face_crop=False)

    def __call__(self, image: Tensor) -> Tensor:
        return image
