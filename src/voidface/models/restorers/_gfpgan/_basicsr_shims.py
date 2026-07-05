# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2021 XPixelGroup (BasicSR)
# Copyright (c) 2026 Voidface contributors (modifications)
#
# Two shims that let us vendor GFPGAN's clean-arch files WITHOUT
# adding basicsr as a runtime dependency. basicsr transitively pulls
# in facexlib which is blocked on Python 3.12; the two symbols we
# actually need are trivial to inline:
#
#   * ARCH_REGISTRY.register() — a decorator that registers a class in
#     a global registry. We do not use the registry; the decorator is
#     a pass-through.
#   * default_init_weights — a small helper that applies Kaiming
#     initialization to Conv2d, Linear, and BatchNorm layers.

"""BasicSR shim helpers used by the vendored GFPGAN architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

if TYPE_CHECKING:
    from typing import Any

__all__ = ["ARCH_REGISTRY", "default_init_weights"]


class _NoopRegistry:
    """Stand-in for :class:`basicsr.utils.registry.Registry`.

    The upstream registry lets basicsr look up architectures by name at
    training time. Voidface never uses that lookup path; the decorator
    is a plain pass-through.
    """

    def register(self, cls: Any = None) -> Any:  # noqa: ANN401
        def _decorator(inner: Any) -> Any:  # noqa: ANN401
            return inner

        if cls is None:
            return _decorator
        return cls


ARCH_REGISTRY = _NoopRegistry()


def default_init_weights(
    module_list: Any,  # noqa: ANN401 -- BasicSR API accepts Module or list[Module]
    scale: float = 1.0,
    bias_fill: float = 0.0,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Initialize network weights (inlined from BasicSR arch_util).

    Kaiming initialization for Conv2d and Linear; constant 1.0 for
    BatchNorm. Bias filled with ``bias_fill``. ``scale`` post-multiplies
    the initialized Conv/Linear weights (used by residual blocks).
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
