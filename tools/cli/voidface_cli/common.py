# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Shared helpers used by multiple CLI subcommands.
#
# Extracted from main.py so subcommand modules can import a single
# well-scoped surface instead of digging private helpers out of the
# dispatcher.

"""Shared CLI helpers for voidface_cli subcommands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import Tensor

    from voidface.generator.architecture import Voidface, VoidfaceConfig

__all__ = [
    "ALLOWED_RESTORERS",
    "ALLOWED_TARGETS",
    "load_generator_checkpoint",
    "renormalize_weights",
    "resolve_device",
    "run_generator_and_save",
]

ALLOWED_RESTORERS = frozenset({"identity", "sd15-vae", "gfpgan"})
ALLOWED_TARGETS = frozenset({"detector", "recognizer", "vae", "sdxl-vae", "openclip"})


def resolve_device(name: str) -> torch.device:
    """Resolve a device name to a torch.device.

    Args:
        name: 'auto' (default), 'cpu', 'cuda', 'mps'.

    Returns:
        A ``torch.device``. 'auto' picks CUDA if available, then MPS
        if available on Apple Silicon, then CPU.
    """
    import torch  # noqa: PLC0415

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def renormalize_weights(weights: dict[str, float]) -> None:
    """Rescale ``weights`` in place so they sum to 1.0.

    When a user selects a subset of targets we do not want the
    remaining weights to be, say, 0.4 total. Renormalizing keeps the
    per-family loss magnitude comparable across subsets.
    """
    total = sum(weights.values())
    if total <= 0:
        return
    for k in weights:
        weights[k] /= total


def load_generator_checkpoint(
    path: Path, device: torch.device | str, log: Any
) -> tuple[Voidface, VoidfaceConfig]:
    """Load a Voidface checkpoint into a fresh generator on ``device``.

    Args:
        path: Path to a .pt checkpoint produced by
            :func:`voidface.core.train.train_generator`.
        device: Any torch.device-like object.
        log: A structlog logger for status logging.

    Returns:
        A tuple ``(generator, config)`` where ``generator`` is a
        loaded and .eval()-ed :class:`Voidface` and ``config`` is
        the checkpoint's :class:`VoidfaceConfig`.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        RuntimeError: When the file cannot be interpreted as a
            torch pickle produced by voidface train.
    """
    import torch

    from voidface.generator.architecture import Voidface, VoidfaceConfig

    if not path.exists():
        msg = (
            f"checkpoint not found: {path}. "
            f"Produce one with `voidface train cfg.toml` or point at a "
            f"downloaded release .pt file."
        )
        raise FileNotFoundError(msg)

    log.info("generator.loading", path=str(path))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        msg = (
            f"could not load checkpoint at {path}. Expected a torch pickle "
            f"produced by voidface train. Original error: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        stored = payload.get("config")
        config = stored if isinstance(stored, VoidfaceConfig) else VoidfaceConfig()
    else:
        state_dict = payload
        config = VoidfaceConfig()
    generator = Voidface(config).to(device).eval()
    generator.load_state_dict(state_dict)
    log.info(
        "generator.loaded",
        params=sum(p.numel() for p in generator.parameters()),
    )
    return generator, config


def run_generator_and_save(
    generator: Voidface,
    config: VoidfaceConfig,
    clean: Tensor,
    output_path: Path,
    epsilon_int: int,
    face_mask: bool = False,
) -> Tensor:
    """Run G on ``clean``, apply optional face mask, save to disk.

    Extracted from _protect_via_generator so both single-image and
    batch paths share the same forward + save sequence.
    """
    import torch
    import torch.nn.functional as F

    from voidface.util.image import save_image

    divisor = 1 << config.num_stages
    original_hw = clean.shape[-2:]
    padded_h = (original_hw[0] + divisor - 1) // divisor * divisor
    padded_w = (original_hw[1] + divisor - 1) // divisor * divisor
    if (padded_h, padded_w) != original_hw:
        clean_padded = F.pad(
            clean,
            (0, padded_w - original_hw[1], 0, padded_h - original_hw[0]),
            mode="reflect",
        )
    else:
        clean_padded = clean
    with torch.no_grad():
        adversarial = generator(clean_padded, epsilon=epsilon_int / 255.0)
    adversarial = adversarial[..., : original_hw[0], : original_hw[1]]

    if face_mask:
        from voidface.util.facemask import face_region_mask

        mask = face_region_mask(clean.squeeze(0)).to(device=clean.device)
        delta = adversarial - clean
        adversarial = (clean + delta * mask.unsqueeze(0)).clamp(0.0, 1.0)

    save_image(adversarial.squeeze(0), output_path)
    return adversarial
