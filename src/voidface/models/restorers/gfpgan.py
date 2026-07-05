# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# GFPGAN v1.4 restorer — R4 HEADLINE.
#
# This is the real face restorer in the bilevel training loop. Every
# off-the-shelf face-swap and nudify pipeline in 2025-2026 runs GFPGAN
# or CodeFormer as a mandatory final step; a delta that survives it is
# a delta that survives the actual attack, not just the pre-restorer
# proxy Voidface has been optimizing against through R3.
#
# Pipeline:
#
#     x  -->  RetinaFace  --landmarks-->  align_faces  --crop-->
#         --> GFPGANv1Clean -->  (image_-1_1, _)  -->  denorm+clamp
#         --> unalign_paste  -->  restored image at input resolution
#
# The RetinaFace instance is passed in rather than constructed here so
# we reuse the detector target already loaded by the training loop
# (its landmarks come free with the classification forward we're
# already doing for the detector suppression loss).
#
# See Documentation/models/restorers.md and
# Documentation/training/bilevel-adversarial.md.

"""GFPGAN v1.4 face restorer for the bilevel training loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from voidface.data.align import align_faces, unalign_paste
from voidface.models.restorers._gfpgan.gfpgan_clean import GFPGANv1Clean
from voidface.models.restorers.base import RestorerSpec

if TYPE_CHECKING:
    from voidface.models.detectors.retinaface import RetinaFace

__all__ = ["GfpganRestorer"]

_OUT_SIZE = 512
_MODEL_ID = "TencentARC/GFPGANv1"
_WEIGHTS_FILENAME = "GFPGANv1.4.pth"
# Pinned SHA-256 of the upstream GFPGANv1.4.pth release. When None,
# the gate warns but proceeds — used during bring-up before the real
# hash has been captured out-of-band. R5 flips this to a real hash
# and self-hosts under a Voidface HF org.
_WEIGHTS_SHA256: str | None = None


class GfpganRestorer:
    """GFPGAN v1.4 face restorer wrapped as a Voidface Restorer.

    Args:
        detector: A :class:`voidface.models.detectors.retinaface.RetinaFace`
            instance. Reused for landmark detection so we do not load
            two copies of the same network.
        device: Torch device the restorer parameters live on.
        detector_score_threshold: Face-confidence threshold for the
            single top-scoring anchor picked out of RetinaFace's
            output. If no anchor exceeds this, the restorer falls
            back to the identity map for that batch element and logs
            a warning.
    """

    spec = RestorerSpec(
        name="gfpgan-v1.4",
        expects_face_crop=False,
        weight_url=None,
    )

    def __init__(
        self,
        detector: RetinaFace,
        device: torch.device | str = "cpu",
        detector_score_threshold: float = 0.5,
    ) -> None:
        self._detector = detector
        self._device = torch.device(device)
        self._score_threshold = detector_score_threshold
        self._net = _load_gfpgan_v1_4_weights(_MODEL_ID, self._device)

    def __call__(self, image: Tensor) -> Tensor:
        """Restore ``image`` through the aligned GFPGAN forward.

        Args:
            image: A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.

        Returns:
            A ``(N, 3, H, W)`` restored tensor in ``[0.0, 1.0]``,
            same shape and range as the input.
        """
        if image.dim() != 4 or image.size(1) != 3:
            msg = f"Expected (N, 3, H, W) input, got {tuple(image.shape)}."
            raise ValueError(msg)

        # 1) Detect landmarks. RetinaFace returns per-anchor scores +
        # bbox regressions + landmarks; for each batch element we pick
        # the anchor with the highest face-present logit and use its
        # landmark regression. We do this in no-grad because landmark
        # regression at the peak anchor is a discrete argmax — not
        # differentiable — but the subsequent alignment IS
        # differentiable given the landmark coordinates.
        with torch.no_grad():
            det_out = self._detector(image)
            landmarks = _pick_top_landmarks(det_out, image, self._score_threshold)

        # If no face found for a batch element, fall back to the
        # identity map for that element.
        if landmarks is None:
            return image

        # 2) Align to 512x512 FFHQ template. Differentiable.
        aligned = align_faces(image, landmarks, output_size=_OUT_SIZE)

        # 3) Forward through GFPGAN. Expects [-1, 1] normalized input.
        crop_neg1_1 = aligned.crop.mul(2.0).sub(1.0)
        restored_image, _ = self._net(crop_neg1_1, randomize_noise=False)
        restored_01 = restored_image.add(1.0).div(2.0).clamp(0.0, 1.0)

        # 4) Paste the restored crop back onto the source image
        # coordinates via inverse similarity transform + feathered
        # blend. Differentiable.
        composited = unalign_paste(
            image, restored_01, aligned.transform, feather_pixels=32
        )
        return composited


def _pick_top_landmarks(
    outputs, image: Tensor, threshold: float
) -> Tensor | None:
    """Extract the top-scoring anchor's landmarks per batch element.

    Landmarks come from the detector as offsets from anchor centers;
    the RetinaFace surrogate already returns the raw regressions in
    ``TargetOutputs.aux["landmarks"]``. For a Phase-R4.5.2b-quality
    aligner it is sufficient to take the top-confidence anchor's
    landmarks and treat them as absolute image pixels. A more
    principled anchor-decode lands in R5.
    """
    aux = outputs.aux
    if aux is None or "landmarks" not in aux or outputs.logits is None:
        return None
    logits = outputs.logits                     # (N, A, 2)
    landmark_reg = aux["landmarks"]             # (N, A, 10)
    face_score = torch.softmax(logits, dim=-1)[..., 1]
    peak = face_score.argmax(dim=-1)            # (N,)
    max_conf = face_score.gather(1, peak.unsqueeze(-1)).squeeze(-1)
    if (max_conf < threshold).all():
        return None
    # Index the (N, A, 10) landmarks tensor at peak.
    idx = peak.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 10)
    top_landmarks = landmark_reg.gather(1, idx).squeeze(1)   # (N, 10)
    # Reshape to (N, 5, 2). Coordinates from RetinaFace's regression are
    # already in image-pixel space (the wrapper resizes to 640, the
    # detector returns anchor-offset landmarks in pixels of that 640
    # canvas). Map back to the original image's coordinate frame.
    _, _, h, w = image.shape
    landmarks_640 = top_landmarks.view(-1, 5, 2)
    scale_xy = torch.tensor(
        [w / 640.0, h / 640.0], device=image.device, dtype=image.dtype
    )
    return landmarks_640 * scale_xy


def _load_gfpgan_v1_4_weights(model_id: str, device: torch.device):
    """Fetch and load GFPGAN v1.4 weights."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from voidface.util.checksum import verify_sha256

    weights_path = Path(hf_hub_download(repo_id=model_id, filename=_WEIGHTS_FILENAME))
    verify_sha256(weights_path, expected=_WEIGHTS_SHA256)
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = raw.get("params_ema", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw

    net = GFPGANv1Clean(
        out_size=_OUT_SIZE,
        num_style_feat=512,
        channel_multiplier=2,
        fix_decoder=True,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    ).to(device)
    net.eval()
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if unexpected:
        import warnings

        warnings.warn(
            f"GFPGAN: {len(unexpected)} unexpected state_dict keys skipped "
            f"(first three: {unexpected[:3]}). "
            f"Missing keys: {len(missing)} (first three: {list(missing)[:3]}).",
            stacklevel=2,
        )
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net
