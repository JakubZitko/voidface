# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Voidface generator G — a small U-Net that predicts an adversarial
# delta. This is the SHIPPED model. Every user of Voidface downloads a
# trained version of this network (~5-15 MB after 8-bit quantization),
# loads it into memory, and applies it to their photo. That single
# forward pass replaces the ~100-300-step PGD loop the training system
# uses.
#
# Design targets:
#
#   * Parameter count: 5-10 M float32 -> <=15 MB fp32 -> <=4 MB int8.
#   * Latency: <=500 ms per 1024x1024 image on Apple Neural Engine
#     via CoreML. <=3 s on Intel Mac CPU.
#   * Deterministic: no stochastic components at inference.
#   * Input: (N, 3, H, W) float in [0, 1] RGB, any resolution >= 256
#     shortest side.
#   * Output: same shape, same range, values in [0, 1] with
#     ||G(x) - x||_inf <= epsilon (the L-infinity budget).
#
# See Documentation/models/generator.md.

"""Voidface generator G — adversarial delta predictor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["Voidface", "VoidfaceConfig"]


@dataclass(frozen=True)
class VoidfaceConfig:
    """Static configuration for the generator network.

    Attributes:
        epsilon: Default L-infinity delta budget as a fraction of the
            unit interval (e.g. ``12 / 255``). Bound in the forward
            call; trained-in default that the user can override at
            inference.
        base_channels: Number of channels at the outermost encoder
            stage. Doubled at each downsample stage. 16 gives a
            ~5.5 M-parameter network — the R5 target size. Must be
            divisible by 8 (GroupNorm constraint).
        num_stages: Number of encoder/decoder stages. 4 is the sweet
            spot for 256-1024 px face crops.
        attention_at_bottleneck: Include a self-attention block at the
            bottleneck. Off by default in R5.1 to keep the network
            fully convolutional (better CoreML export). Turn on in
            R5.2 experiments if the training-side signal is weak.
    """

    epsilon: float = 12.0 / 255.0
    base_channels: int = 16
    num_stages: int = 4
    attention_at_bottleneck: bool = False


class Voidface(nn.Module):
    """The Voidface generator G.

    Predicts an adversarial delta ``d = tanh(g(x)) * epsilon`` and
    returns the clamped protected image ``clip(x + d, 0, 1)``.
    """

    def __init__(self, config: VoidfaceConfig | None = None) -> None:
        super().__init__()
        self._config = config or VoidfaceConfig()
        base = self._config.base_channels
        stages = self._config.num_stages

        # Encoder: 4 stages of double-channel + downsample.
        encoder_channels = [base * (2**i) for i in range(stages + 1)]
        # Cap the largest channel width so we do not blow up param count.
        encoder_channels = [min(c, 384) for c in encoder_channels]

        self.stem = nn.Conv2d(3, encoder_channels[0], kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(stages):
            self.encoder_blocks.append(
                _ResidualBlock(encoder_channels[i], encoder_channels[i + 1])
            )
            self.downsamples.append(
                nn.Conv2d(encoder_channels[i + 1], encoder_channels[i + 1], kernel_size=2, stride=2)
            )

        # Bottleneck.
        self.bottleneck = nn.Sequential(
            _ResidualBlock(encoder_channels[-1], encoder_channels[-1]),
            _ResidualBlock(encoder_channels[-1], encoder_channels[-1]),
        )

        if self._config.attention_at_bottleneck:
            self.attention: nn.Module | None = _SelfAttention(encoder_channels[-1])
        else:
            self.attention = None

        # Decoder: mirror of encoder with skip connections.
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in reversed(range(stages)):
            self.upsamples.append(
                nn.ConvTranspose2d(
                    encoder_channels[i + 1],
                    encoder_channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
            )
            # skip cat -> 2*channels input to the block
            self.decoder_blocks.append(
                _ResidualBlock(2 * encoder_channels[i + 1], encoder_channels[i])
            )

        self.head = nn.Sequential(
            nn.GroupNorm(min(8, encoder_channels[0]), encoder_channels[0]),
            nn.GELU(),
            nn.Conv2d(encoder_channels[0], 3, kernel_size=3, padding=1),
        )

    @property
    def config(self) -> VoidfaceConfig:
        return self._config

    def forward(self, image: Tensor, epsilon: float | None = None) -> Tensor:
        """Protect ``image`` and return the same-shape adversarial output.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``.
                Height and width must be divisible by ``2^num_stages``
                (16 for the default 4 stages). Callers should reflect-pad
                or resize non-conforming inputs; the CLI handles this.
            epsilon: L-infinity budget override for this forward. When
                omitted, the config-default is used.

        Returns:
            A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.
        """
        # Shape validation is skipped during tracing so ONNX export
        # does not fire TracerWarnings on the boolean `if` guards.
        # The caller (or the CLI's protect / batch paths) is expected
        # to have padded to the right shape by the time we get here.
        if not torch.jit.is_tracing():
            if image.dim() != 4 or image.size(1) != 3:
                msg = f"Expected (N, 3, H, W) input, got {tuple(image.shape)}."
                raise ValueError(msg)
            divisor = 1 << self._config.num_stages
            if image.shape[-1] % divisor != 0 or image.shape[-2] % divisor != 0:
                msg = (
                    f"Input side lengths must be divisible by {divisor}. "
                    f"Got {tuple(image.shape[-2:])}. Pad or resize before calling."
                )
                raise ValueError(msg)

        skips: list[Tensor] = []
        feat = self.stem(image)
        for block, downsample in zip(self.encoder_blocks, self.downsamples, strict=True):
            feat = block(feat)
            skips.append(feat)
            feat = downsample(feat)

        feat = self.bottleneck(feat)
        if self.attention is not None:
            feat = feat + self.attention(feat)

        for i, (upsample, block) in enumerate(zip(self.upsamples, self.decoder_blocks, strict=True)):
            feat = upsample(feat)
            skip = skips[-(i + 1)]
            feat = torch.cat([feat, skip], dim=1)
            feat = block(feat)

        raw = self.head(feat)
        # Delta is tanh-projected to (-1, 1), then scaled by epsilon.
        eps = epsilon if epsilon is not None else self._config.epsilon
        delta = torch.tanh(raw) * eps
        return (image + delta).clamp(0.0, 1.0)


# --- internal blocks --------------------------------------------------------


class _ResidualBlock(nn.Module):
    """Two conv + GroupNorm + GELU with a 1x1 residual if channels differ."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, in_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        out_groups = min(8, out_channels)
        self.norm2 = nn.GroupNorm(out_groups, out_channels)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.residual(x)


class _SelfAttention(nn.Module):
    """Multi-head self-attention over a 2D feature map.

    Reserved for R5.2 experiments; disabled in the R5.1 default.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(n, self.num_heads, c // self.num_heads, h * w)
        k = k.view(n, self.num_heads, c // self.num_heads, h * w)
        v = v.view(n, self.num_heads, c // self.num_heads, h * w)
        attn = torch.softmax(
            (q.transpose(-1, -2) @ k) / (c // self.num_heads) ** 0.5, dim=-1
        )
        out = (v @ attn.transpose(-1, -2)).view(n, c, h, w)
        return self.out(out)
