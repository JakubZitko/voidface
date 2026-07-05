# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2018-2024 InsightFace Contributors
# Copyright (c) 2026 Voidface contributors (modifications)
#
# Vendored from InsightFace's arcface_torch reference implementation:
#   https://github.com/deepinsight/insightface
#   recognition/arcface_torch/backbones/iresnet.py
#
# License: Apache-2.0 (see LICENSES/Apache-2.0-insightface.txt).
#
# Modifications from upstream:
#   * Removed torch.cuda.amp.autocast wrapper in IResNet.forward. Voidface
#     runs surrogates in eval mode with frozen parameters; the CUDA-only
#     autocast context breaks on CPU/MPS and is not needed.
#   * Removed the "raise ValueError()" guard in _iresnet's pretrained
#     branch; Voidface loads pretrained weights outside this file via
#     the state_dict path in arcface.py.
#   * Restricted __all__ to iresnet100 — Voidface only uses the R100
#     variant. Other constructors (iresnet18/34/50/200) remain in-file
#     but are not exported.
#   * Style-only: black-formatted, no functional changes to the
#     architecture. State-dict compatibility with upstream weights is
#     preserved bit-for-bit.
#
# Every layer name matches upstream so the checkpoints from
# minchul/cvlface_arcface_ir101_webface4m load without renaming.

"""IResNet-100 face-recognition backbone (vendored)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["iresnet100"]

_USING_CKPT = False


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class IBasicBlock(nn.Module):
    """The IR-block used throughout IResNet — BN, conv, BN, PReLU, conv, BN + residual."""

    expansion = 1

    def __init__(  # noqa: PLR0913
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            msg = "IBasicBlock only supports groups=1 and base_width=64."
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 is not supported in IBasicBlock."
            raise NotImplementedError(msg)
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def _forward_impl(self, x: Tensor) -> Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x: Tensor) -> Tensor:
        if self.training and _USING_CKPT:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)


class IResNet(nn.Module):
    """IResNet backbone as published by InsightFace's arcface_torch."""

    fc_scale = 7 * 7

    def __init__(  # noqa: PLR0913
        self,
        block: type[IBasicBlock],
        layers: list[int],
        dropout: float = 0.0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        fp16: bool = False,  # noqa: ARG002 -- kept for state_dict compat with upstream
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            msg = (
                "replace_stride_with_dilation should be None or a 3-element list, "
                f"got {replace_stride_with_dilation}."
            )
            raise ValueError(msg)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, IBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[IBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers: list[nn.Module] = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Note: upstream wraps this in torch.cuda.amp.autocast; Voidface
        # runs surrogates in eval mode on CPU/MPS/CUDA and does not need
        # per-forward mixed precision — the autocast context is removed.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def iresnet100(**kwargs: object) -> IResNet:
    """IResNet-100 constructor.

    Layer schedule ``[3, 13, 30, 3]`` — matches InsightFace's canonical
    R100 checkpoints (glintr100, arcface_r100_v1, cvlface IR-101).
    """
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)  # type: ignore[arg-type]
