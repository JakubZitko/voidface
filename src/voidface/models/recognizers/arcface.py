# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# ArcFace / IResNet-100 identity encoder surrogate.
#
# This replaces the Phase R1 FaceNet stand-in with the real ArcFace
# family that every off-the-shelf face-swap and personalization
# pipeline (SimSwap, InSwapper, FaceShifter, HifiFace, InstantID,
# PhotoMaker, IP-Adapter) uses as its identity conditioner. See
# Documentation/models/recognizers.md.
#
# The architecture lives in _iresnet.py (Apache-2.0 vendored from
# InsightFace). This wrapper handles:
#
#   * Weight fetch from huggingface_hub (minchul/cvlface_arcface_ir101_
#     webface4m).
#   * State-dict key remapping (strip ``model.``/``backbone.`` prefixes
#     inherited from the cvlface wrapper).
#   * Input adaptation: (N, 3, H, W) [0, 1] RGB -> resize-with-antialias
#     to 112 -> BGR flip (upstream trained on cv2 pixels which are BGR)
#     -> normalize to [-1, 1].
#   * Output: L2-normalized 512-D embedding, plugs into the existing
#     arcface_identity_loss in voidface.core.loss.

"""ArcFace identity encoder surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.recognizers._iresnet import iresnet100

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Arcface"]

_INPUT_SIZE = 112
_MODEL_ID = "minchul/cvlface_arcface_ir101_webface4m"
_WEIGHTS_FILENAME = "model.pt"


class Arcface(nn.Module):
    """Differentiable ArcFace / IResNet-100 identity encoder.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="arcface",
        family="recognizers",
        input_resolution=_INPUT_SIZE,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self._net = _load_iresnet100_with_weights(_MODEL_ID, self._device)

    def forward(self, image: Tensor) -> TargetOutputs:
        """Run the encoder on ``image``.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``
                RGB order. Inputs are antialias-resized to 112.

        Returns:
            :class:`TargetOutputs` with ``embedding`` set to an
            L2-normalized ``(N, 512)`` identity vector.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got shape {tuple(image.shape)}."
            raise ValueError(msg)

        resized = (
            image
            if image.shape[-2:] == (_INPUT_SIZE, _INPUT_SIZE)
            else F.interpolate(
                image,
                size=_INPUT_SIZE,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
        # arcface_torch was trained on cv2 pixels which are BGR. Feeding
        # RGB into a BGR-trained tower silently gives a different-but-
        # convergent embedding that will NOT match downstream ArcFace
        # verifiers. Explicit channel flip is the fix (correctness
        # critic, R4 workflow).
        bgr = resized.flip(dims=(1,))
        normalized = bgr.sub(0.5).mul(2.0)                      # -> [-1, 1]
        raw = self._net(normalized)
        embedding = F.normalize(raw, dim=-1)
        return TargetOutputs(embedding=embedding)

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]


def _load_iresnet100_with_weights(model_id: str, device: torch.device):  # noqa: ANN202
    """Download ArcFace R100 weights and load into a vendored IResNet.

    The cvlface checkpoint bundles the arch under a wrapper module, so
    real weight keys are prefixed with ``model.`` or ``backbone.``.
    We strip the prefix inline (same pattern as the VAE legacy-key
    remap in models/vaes/_diffusers_loader.py).
    """
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id=model_id, filename=_WEIGHTS_FILENAME)
    # cvlface ships a torch pickle; safetensors mirror not published
    # for this exact repo. We accept the .pt with weights_only=False
    # explicitly — the alternative (raw safetensors mirror) is on the
    # R5 self-hosting TODO. See risks in Phase R4 CHANGELOG.
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(raw)
    normalized = _normalize_arcface_state_dict(state_dict)

    net = iresnet100().to(device)
    net.eval()
    missing, unexpected = net.load_state_dict(normalized, strict=False)
    if len(unexpected) > 0:
        # Loud but non-fatal — a partial load still gives approximate
        # ArcFace geometry, which is better than crashing.
        import warnings

        warnings.warn(
            f"Arcface: {len(unexpected)} unexpected state_dict keys skipped "
            f"(first three: {unexpected[:3]}). "
            f"Missing keys: {len(missing)} (first three: {list(missing)[:3]}).",
            stacklevel=2,
        )
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def _extract_state_dict(raw: object) -> Mapping[str, Tensor]:
    """Peel off common wrapping (Lightning, cvlface) around a state dict."""
    if isinstance(raw, dict):
        for candidate in ("state_dict", "model_state_dict", "network"):
            inner = raw.get(candidate)
            if isinstance(inner, dict):
                return inner
        # Might already BE a state_dict.
        return raw  # type: ignore[return-value]
    msg = f"Unrecognized checkpoint shape: {type(raw).__name__}."
    raise ValueError(msg)


def _normalize_arcface_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Strip the cvlface wrapper prefixes and align to iresnet100's layer names."""
    stripped: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key
        for prefix in ("model.net.", "model.backbone.", "model.", "backbone.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        stripped[new_key] = tensor
    return stripped
