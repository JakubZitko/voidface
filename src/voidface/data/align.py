# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Differentiable 5-point face alignment.
#
# The R4 CEO critic flagged that face restorers (GFPGAN, CodeFormer,
# Real-ESRGAN) are trained on tightly-cropped FFHQ-aligned face crops.
# Feeding them off-distribution unaligned inputs produces gradient
# signal that does not reflect what an attacker's real pipeline
# actually sees. This module gives the GfpganRestorer in R4.5.2b a
# differentiable path from a full RGB image + 5 detected landmarks to
# an FFHQ-canonical 512x512 aligned crop, and back.
#
# Everything is grid_sample-based (differentiable) so gradients flow
# through the alignment when the restorer is invoked from the PGD
# loop. This is what makes the "bilevel loss survives face restore"
# claim honest — the attacker's alignment step is inside our gradient
# graph, not treated as a fixed preprocessor.

"""Differentiable 5-point face alignment (RGB -> FFHQ 512 -> RGB)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "FFHQ_LANDMARKS_512",
    "AlignedFace",
    "align_faces",
    "estimate_similarity_transform",
    "unalign_paste",
]


# FFHQ canonical 5-point landmark template at 512x512 output. Order:
#   left eye, right eye, nose tip, left mouth corner, right mouth corner.
# Coordinates in pixel space of the 512x512 aligned crop. Values match
# what GFPGAN, CodeFormer, and Real-ESRGAN-face-enhance were trained on.
FFHQ_LANDMARKS_512: tuple[tuple[float, float], ...] = (
    (192.98138, 239.94708),  # left eye
    (318.90277, 240.19366),  # right eye
    (256.63416, 314.01935),  # nose tip
    (201.26117, 371.41043),  # left mouth corner
    (313.08905, 371.15118),  # right mouth corner
)


class AlignedFace(torch.NamedTuple if False else object):  # placeholder swap below
    """Container for one aligned face and its transform metadata.

    Attributes:
        crop: The aligned face crop, ``(N, 3, 512, 512)`` in ``[0, 1]``.
        transform: The 2x3 similarity transform matrix that maps
            unaligned landmark coordinates -> aligned template
            coordinates. Shape ``(N, 2, 3)``.
    """

    __slots__ = ("crop", "transform")

    def __init__(self, crop: Tensor, transform: Tensor) -> None:
        object.__setattr__(self, "crop", crop)
        object.__setattr__(self, "transform", transform)


def estimate_similarity_transform(
    source: Tensor, target: Tensor
) -> Tensor:
    """Least-squares fit of a 2D similarity transform.

    Solves for ``(s, R, t)`` that minimizes ``|| s R source + t - target ||^2``
    per batch element. Returns the concatenated ``(N, 2, 3)`` affine
    matrix ``[s R | t]``.

    Args:
        source: Source points in image space, shape ``(N, 5, 2)``.
        target: Target points in the aligned template's coordinate
            space, shape ``(N, 5, 2)``.

    Returns:
        A ``(N, 2, 3)`` similarity transform matrix.
    """
    if source.shape != target.shape:
        msg = (
            f"Shape mismatch: source={tuple(source.shape)} "
            f"target={tuple(target.shape)}"
        )
        raise ValueError(msg)
    if source.dim() != 3 or source.size(-1) != 2:
        msg = f"Expected (N, K, 2), got {tuple(source.shape)}."
        raise ValueError(msg)

    src_centroid = source.mean(dim=1, keepdim=True)
    tgt_centroid = target.mean(dim=1, keepdim=True)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    src_scale = src_centered.pow(2).sum(dim=(1, 2)).clamp_min(1e-12).sqrt()
    tgt_scale = tgt_centered.pow(2).sum(dim=(1, 2)).clamp_min(1e-12).sqrt()

    # SVD-based rotation solve, per-batch.
    covariance = torch.matmul(src_centered.transpose(-1, -2), tgt_centered)
    u, _, vt = torch.linalg.svd(covariance)
    # Correct for reflections so det(R) = +1.
    reflection_fix = torch.eye(2, device=source.device, dtype=source.dtype).expand(
        source.size(0), 2, 2
    ).clone()
    signs = torch.sign(torch.linalg.det(torch.matmul(vt.transpose(-1, -2), u.transpose(-1, -2))))
    reflection_fix[:, 1, 1] = signs
    rotation = torch.matmul(
        torch.matmul(vt.transpose(-1, -2), reflection_fix), u.transpose(-1, -2)
    )
    scale = (tgt_scale / src_scale).view(-1, 1, 1)
    scaled_rotation = scale * rotation

    translation = tgt_centroid.transpose(-1, -2) - torch.matmul(
        scaled_rotation, src_centroid.transpose(-1, -2)
    )
    return torch.cat([scaled_rotation, translation], dim=-1)


def align_faces(
    image: Tensor,
    landmarks: Tensor,
    output_size: int = 512,
    template: tuple[tuple[float, float], ...] = FFHQ_LANDMARKS_512,
) -> AlignedFace:
    """Warp each face in ``image`` to the canonical FFHQ template.

    Args:
        image: A ``(N, 3, H, W)`` tensor in ``[0.0, 1.0]``.
        landmarks: A ``(N, 5, 2)`` tensor of pixel-space (x, y) 5-point
            landmarks — left eye, right eye, nose, left mouth, right
            mouth.
        output_size: Side length of the aligned crop. Defaults to 512
            to match GFPGAN's expected input.
        template: The 5-point canonical template. Defaults to FFHQ.

    Returns:
        An :class:`AlignedFace` with the crop and the estimated
        transform.
    """
    if image.dim() != 4 or image.size(1) != 3:
        msg = f"Expected (N, 3, H, W) image, got {tuple(image.shape)}."
        raise ValueError(msg)
    if landmarks.shape[-2:] != (5, 2):
        msg = f"Expected (N, 5, 2) landmarks, got {tuple(landmarks.shape)}."
        raise ValueError(msg)

    n = image.size(0)
    device = image.device
    dtype = image.dtype

    template_tensor = torch.tensor(template, dtype=dtype, device=device)
    template_batch = template_tensor.unsqueeze(0).expand(n, 5, 2)

    transform = estimate_similarity_transform(landmarks, template_batch)

    # Build the inverse affine for torch.nn.functional.affine_grid.
    # affine_grid expects a normalized-coord (-1, 1) matrix that maps
    # output -> input. We have image->aligned; we want aligned->image
    # to sample the source pixels.
    inverse = _invert_affine_2x3(transform)
    inverse_normalized = _pixel_to_normalized(
        inverse,
        src_h=image.size(-2),
        src_w=image.size(-1),
        dst_h=output_size,
        dst_w=output_size,
    )
    grid = F.affine_grid(inverse_normalized, size=(n, 3, output_size, output_size), align_corners=False)
    crop = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return AlignedFace(crop=crop, transform=transform)


def unalign_paste(
    original: Tensor,
    restored_crop: Tensor,
    transform: Tensor,
    feather_pixels: int = 32,
) -> Tensor:
    """Paste an aligned restored crop back into the original image space.

    Uses the same similarity transform (inverted) to warp the
    restored crop back to the source image's coordinate frame, then
    alpha-blends over the original using a feathered mask so seams do
    not appear as sharp edges. All operations are differentiable.

    Args:
        original: ``(N, 3, H, W)`` original image in ``[0, 1]``.
        restored_crop: ``(N, 3, S, S)`` restored aligned crop in ``[0, 1]``.
        transform: ``(N, 2, 3)`` transform from :func:`align_faces`.
        feather_pixels: Softness of the alpha-mask edge, in pixels of
            the aligned space.

    Returns:
        A ``(N, 3, H, W)`` blended image.
    """
    n, _, out_h, out_w = original.shape
    _, _, crop_h, crop_w = restored_crop.shape

    forward_normalized = _pixel_to_normalized(
        transform,
        src_h=crop_h,
        src_w=crop_w,
        dst_h=out_h,
        dst_w=out_w,
    )
    grid = F.affine_grid(
        forward_normalized, size=(n, 3, out_h, out_w), align_corners=False
    )
    warped_face = F.grid_sample(
        restored_crop, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    # Feathered alpha mask defined in aligned space, then warped by
    # the same grid so it aligns pixel-for-pixel with warped_face.
    alpha_crop = _feathered_rect_mask(
        crop_h, crop_w, feather_pixels, device=original.device, dtype=original.dtype
    ).expand(n, 1, crop_h, crop_w)
    alpha_warped = F.grid_sample(
        alpha_crop, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    return original * (1.0 - alpha_warped) + warped_face * alpha_warped


# --- internals ---------------------------------------------------------------


def _invert_affine_2x3(matrix: Tensor) -> Tensor:
    """Invert a batched 2x3 affine matrix into its 2x3 inverse.

    Extends to 3x3 by appending ``[0, 0, 1]``, inverts, then drops the
    last row.
    """
    n = matrix.size(0)
    homo = torch.zeros(n, 3, 3, device=matrix.device, dtype=matrix.dtype)
    homo[:, :2, :3] = matrix
    homo[:, 2, 2] = 1.0
    inverted = torch.linalg.inv(homo)
    return inverted[:, :2, :3]


def _pixel_to_normalized(
    matrix: Tensor,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> Tensor:
    """Convert a pixel-space affine matrix to :func:`affine_grid` form.

    ``F.affine_grid`` expects the matrix in the normalized-device
    coordinate frame ``(x, y) in [-1, 1]``. Pixel-space matrix ``M``
    that maps ``src_pixel -> dst_pixel`` becomes ``T_dst @ M @ T_src^-1``
    where ``T`` maps normalized coords to pixel coords for each side.
    """
    device = matrix.device
    dtype = matrix.dtype

    def _norm_to_pixel(h: int, w: int) -> Tensor:
        return torch.tensor(
            [[(w - 1) / 2.0, 0.0, (w - 1) / 2.0], [0.0, (h - 1) / 2.0, (h - 1) / 2.0]],
            device=device,
            dtype=dtype,
        )

    src = _norm_to_pixel(src_h, src_w)
    dst_inv_3x3 = torch.zeros(3, 3, device=device, dtype=dtype)
    dst_inv_3x3[:2, :3] = _norm_to_pixel(dst_h, dst_w)
    dst_inv_3x3[2, 2] = 1.0
    dst_inv = torch.linalg.inv(dst_inv_3x3)[:2, :3]

    n = matrix.size(0)
    src_batch = src.unsqueeze(0).expand(n, 2, 3)
    dst_inv_batch = dst_inv.unsqueeze(0).expand(n, 2, 3)

    # Compose: dst_inv @ [M | 0; 0 1] @ [src | 0; 0 1]
    m_3x3 = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    m_3x3[:, :2, :3] = matrix
    m_3x3[:, 2, 2] = 1.0
    src_3x3 = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    src_3x3[:, :2, :3] = src_batch
    src_3x3[:, 2, 2] = 1.0
    dst_inv_3x3_batch = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    dst_inv_3x3_batch[:, :2, :3] = dst_inv_batch
    dst_inv_3x3_batch[:, 2, 2] = 1.0
    composed = torch.matmul(torch.matmul(dst_inv_3x3_batch, m_3x3), src_3x3)
    return composed[:, :2, :3]


def _feathered_rect_mask(
    h: int, w: int, feather: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """A soft-edged rectangular alpha mask, 1 in the center, 0 outside."""
    y = torch.linspace(0, 1, h, device=device, dtype=dtype).view(h, 1)
    x = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, w)
    feather_norm = feather / max(h, w)
    edge_dist = torch.min(
        torch.min(x, 1.0 - x), torch.min(y, 1.0 - y).expand(h, w)
    )
    alpha = (edge_dist / max(feather_norm, 1e-6)).clamp(0.0, 1.0)
    return alpha.view(1, 1, h, w)
