# SPDX-License-Identifier: MIT
# Copyright (c) 2019 Jiankang Deng and biubug6 contributors
# Copyright (c) 2026 Voidface contributors (modifications)
#
# Vendored from biubug6/Pytorch_Retinaface:
#   https://github.com/biubug6/Pytorch_Retinaface
#   models/retinaface.py + models/net.py (SSH, FPN blocks only)
#
# License: MIT (see LICENSES/MIT-biubug6-retinaface.txt).
#
# Modifications from upstream:
#   * Removed the MobileNetV1(0.25) backbone path — Voidface only ships
#     the ResNet-50 variant; keep the wrapper minimal.
#   * Removed the `phase` argument from ``RetinaFace.forward``. Voidface
#     ALWAYS returns raw pre-softmax classification logits so the
#     naming ``TargetOutputs.logits`` is truthful and the R4
#     correctness-critic softmax-vs-logits ambiguity goes away. The
#     downstream suppression loss applies its own softmax if needed.
#   * torch.utils.data.IntermediateLayerGetter dropped in favour of
#     ``torchvision.models._utils.IntermediateLayerGetter`` (same class,
#     upstream just re-imported it).
#   * Type-annotated, ruff-formatted. No functional changes to the arch;
#     state dicts from yakhyo/retinaface-pytorch load bit-for-bit.

"""RetinaFace-ResNet50 face-detector architecture (vendored)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import torchvision.models._utils as tv_utils
from torch import Tensor, nn
from torchvision.models import resnet50

if TYPE_CHECKING:
    pass

__all__ = ["RetinaFaceR50Arch"]


def _conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def _conv_bn_1x1(inp: int, oup: int, stride: int = 1, leaky: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class _SSH(nn.Module):
    """SSH (single-stage headless) context module."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        if out_channel % 4 != 0:
            msg = f"SSH out_channel must be divisible by 4, got {out_channel}."
            raise ValueError(msg)
        leaky = 0.1 if out_channel <= 64 else 0.0
        self.conv3X3 = _conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = _conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = _conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5_1 = self.conv5X5_1(x)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        return F.relu(torch.cat([conv3x3, conv5x5, conv7x7], dim=1))


class _FPN(nn.Module):
    """Feature-Pyramid Network across three ResNet stages."""

    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = _conv_bn_1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = _conv_bn_1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = _conv_bn_1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = _conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = _conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, features: dict[str, Tensor]) -> list[Tensor]:
        values = list(features.values())
        output1 = self.output1(values[0])
        output2 = self.output2(values[1])
        output3 = self.output3(values[2])
        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)
        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class _ClassHead(nn.Module):
    """Anchor-classification head. Returns raw logits, not softmax."""

    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class _BboxHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class _LandmarkHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


_CFG_RESNET50 = {
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
    "num_anchors": 2,
}


class RetinaFaceR50Arch(nn.Module):
    """RetinaFace-ResNet50 architecture (biubug6 config).

    Forward returns ``(bbox_regressions, classifications, ldm_regressions)``
    concatenated across the three FPN levels. Classifications are RAW
    pre-softmax logits with shape ``(N, num_anchors_total, 2)``. The
    R4 correctness critic requested this over the phase-gated softmax
    of upstream so the downstream loss knows exactly what it is looking
    at.
    """

    def __init__(self) -> None:
        super().__init__()
        cfg = _CFG_RESNET50
        backbone = resnet50(weights=None)  # weights loaded via state_dict path
        self.body = tv_utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = _FPN(in_channels_list, out_channels)
        self.ssh1 = _SSH(out_channels, out_channels)
        self.ssh2 = _SSH(out_channels, out_channels)
        self.ssh3 = _SSH(out_channels, out_channels)
        self.ClassHead = self._make_head(_ClassHead, out_channels, cfg["num_anchors"])
        self.BboxHead = self._make_head(_BboxHead, out_channels, cfg["num_anchors"])
        self.LandmarkHead = self._make_head(_LandmarkHead, out_channels, cfg["num_anchors"])

    @staticmethod
    def _make_head(
        head_cls: type[nn.Module], inchannels: int, num_anchors: int
    ) -> nn.ModuleList:
        return nn.ModuleList(head_cls(inchannels, num_anchors) for _ in range(3))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        # ALWAYS return raw logits. No phase gating, no upstream softmax.
        return bbox_regressions, classifications, ldm_regressions
