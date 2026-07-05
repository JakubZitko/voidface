# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""IdentityRestorer passes tensors through unchanged and satisfies the
Restorer protocol."""

from __future__ import annotations

import torch

from voidface.models.restorers.base import Restorer, RestorerSpec
from voidface.models.restorers.identity import IdentityRestorer


def test_identity_restorer_is_a_restorer() -> None:
    restorer = IdentityRestorer()
    assert isinstance(restorer, Restorer)
    assert isinstance(restorer.spec, RestorerSpec)
    assert restorer.spec.name == "identity"


def test_identity_restorer_preserves_input() -> None:
    restorer = IdentityRestorer()
    x = torch.rand(2, 3, 16, 16)
    y = restorer(x)
    assert torch.equal(x, y)


def test_identity_restorer_is_differentiable() -> None:
    restorer = IdentityRestorer()
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    y = restorer(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.equal(x.grad, torch.ones_like(x))
