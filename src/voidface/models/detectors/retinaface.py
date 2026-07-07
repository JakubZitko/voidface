# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# RetinaFace-ResNet50 face-detector surrogate.
#
# This replaces the Phase R1 MTCNN-PNet stand-in with real RetinaFace,
# the SSH/FPN detector that InsightFace ships (buffalo_l) and that
# every mainstream face-swap pipeline reaches for when it wants
# reliable face localization. Suppressing RetinaFace's face-present
# logit is Voidface's single most valuable adversarial target — the
# detector is the one pipeline stage no face restorer can compensate
# for; if the detector says "no face here," GFPGAN never fires.
#
# The architecture lives in _retinaface_arch.py (MIT vendored). This
# wrapper handles:
#
#   * Weight fetch via huggingface_hub (yakhyo/retinaface-pytorch).
#   * BGR channel flip + ImageNet-style pixel-mean subtraction (upstream
#     was trained on cv2 pixels with per-channel mean [104, 117, 123]
#     and NO /255 divide).
#   * Antialias-preserving resize on any input.
#   * Returns raw pre-softmax logits with shape (N, anchors, 2).
#
# See Documentation/models/detectors.md.

"""RetinaFace-ResNet50 face-detector surrogate."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voidface.models.base import TargetOutputs, TargetSpec
from voidface.models.detectors._retinaface_arch import RetinaFaceR50Arch

__all__ = ["RetinaFace"]

_INPUT_SIZE = 640
_MODEL_ID = "akhaliq/RetinaFace-R50"
_WEIGHTS_FILENAME = "RetinaFace-R50.pth"
# biubug6/InsightFace pretrained mean, per-channel, BGR order,
# NO /255 divide — matches upstream training pipeline exactly.
_PIXEL_MEAN = (104.0, 117.0, 123.0)


class RetinaFace(nn.Module):
    """Differentiable RetinaFace-R50 face detector.

    Attributes:
        spec: Static specification consumed by the training loop.
    """

    spec = TargetSpec(
        name="retinaface-r50",
        family="detectors",
        input_resolution=_INPUT_SIZE,
        weight_url=None,
    )

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)
        self._net = _load_retinaface_r50_with_weights(_MODEL_ID, self._device)
        mean = torch.tensor(_PIXEL_MEAN, device=self._device).view(1, 3, 1, 1)
        self.register_buffer("_pixel_mean", mean, persistent=False)

    def forward(self, image: Tensor) -> TargetOutputs:
        """Run the detector on ``image``.

        Args:
            image: A ``(N, 3, H, W)`` float tensor in ``[0.0, 1.0]``,
                RGB order. Non-square inputs are antialias-resized to
                ``_INPUT_SIZE x _INPUT_SIZE``; larger images downsample,
                smaller upsample. The choice of a fixed square keeps
                anchor counts stable across a batch.

        Returns:
            :class:`TargetOutputs` with:
              * ``logits``: RAW pre-softmax classification, shape
                ``(N, num_anchors_total, 2)``. Face-present logit is
                ``[..., 1]``; suppression drives it below the background
                logit ``[..., 0]``.
              * ``aux``: ``{"bbox_regressions": ..., "landmarks": ...}``.
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
        # RGB [0, 1] -> BGR [0, 255] -> subtract per-channel mean.
        scaled = resized.mul(255.0)
        bgr = scaled.flip(dims=(1,))
        prepped = bgr - self._pixel_mean

        bbox_regressions, classifications, ldm_regressions = self._net(prepped)
        return TargetOutputs(
            logits=classifications,
            aux={"bbox_regressions": bbox_regressions, "landmarks": ldm_regressions},
        )

    def __call__(self, image: Tensor) -> TargetOutputs:  # type: ignore[override]
        return super().__call__(image)  # type: ignore[no-any-return]


def _load_retinaface_r50_with_weights(model_id: str, device: torch.device):
    """Download RetinaFace-R50 weights and load them into the vendored arch."""
    from huggingface_hub import hf_hub_download

    weights_path = hf_hub_download(repo_id=model_id, filename=_WEIGHTS_FILENAME)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # Common wrapping prefix from DataParallel training runs.
    stripped = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    net = RetinaFaceR50Arch().to(device)
    net.eval()
    missing, unexpected = net.load_state_dict(stripped, strict=False)
    if unexpected:
        import warnings

        warnings.warn(
            f"RetinaFace: {len(unexpected)} unexpected state_dict keys skipped "
            f"(first three: {unexpected[:3]}). "
            f"Missing keys: {len(missing)} (first three: {list(missing)[:3]}).",
            stacklevel=2,
        )
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
