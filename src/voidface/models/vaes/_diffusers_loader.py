# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors
#
# Shared bypass loader for diffusers AutoencoderKL variants.
#
# diffusers 0.29's ``from_pretrained`` fails to load state dicts on
# torch 2.2 (Intel Mac constraint) — it surfaces as
# ``OSError: Unable to load weights from checkpoint file`` even though
# the safetensors file is valid. This helper works around it by:
#
#   1. Fetching config.json and diffusion_pytorch_model.safetensors
#      through huggingface_hub, which is well-behaved on our
#      constrained versions.
#   2. Instantiating :class:`AutoencoderKL` directly from the config.
#   3. Loading the state dict via safetensors.torch.load_file (which
#      works on our torch version).
#   4. Normalizing legacy attention key names to the current diffusers
#      layout.
#
# Every VAE surrogate (Sd15Vae, SdxlVae, and future Flux VAE) uses this
# helper instead of duplicating the workaround. The helper will become
# unnecessary when we can bump diffusers past 0.30 — which requires
# torch>=2.4, which requires dropping Intel Mac wheels.

"""Shared diffusers AutoencoderKL bypass loader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import safetensors.torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from torch import Tensor

__all__ = ["load_autoencoder_kl", "normalize_legacy_attention_keys"]


_LEGACY_ATTENTION_RENAMES: Mapping[str, str] = {
    "query.weight": "to_q.weight",
    "query.bias": "to_q.bias",
    "key.weight": "to_k.weight",
    "key.bias": "to_k.bias",
    "value.weight": "to_v.weight",
    "value.bias": "to_v.bias",
    "proj_attn.weight": "to_out.0.weight",
    "proj_attn.bias": "to_out.0.bias",
}


def load_autoencoder_kl(
    model_id: str,
    device: torch.device,
    *,
    subfolder: str | None = None,
    weights_filename: str = "diffusion_pytorch_model.safetensors",
    config_filename: str = "config.json",
):  # noqa: ANN201 -- returns diffusers AutoencoderKL
    """Load a diffusers :class:`AutoencoderKL` on our constrained stack.

    Args:
        model_id: A ``org/name`` Hugging Face model identifier.
        device: The torch device to move the model to.
        subfolder: If the VAE lives under a subfolder (e.g. ``vae``
            inside a full SD checkpoint), pass it here. Standalone VAE
            repos leave this ``None``.
        weights_filename: Overrides the default ``.safetensors`` name
            for the rare repo that ships weights under a different
            filename.
        config_filename: Overrides the default ``config.json`` name.

    Returns:
        A ``AutoencoderKL`` on ``device``, put in ``.eval()`` mode with
        parameters frozen (``requires_grad_(False)``).

    Raises:
        RuntimeError: When the loaded state dict does not match the
            architecture even after legacy-key normalization.
    """
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=model_id, filename=_prefix(subfolder, config_filename))
    weights_path = hf_hub_download(repo_id=model_id, filename=_prefix(subfolder, weights_filename))

    with open(config_path) as handle:
        config = json.load(handle)
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    vae = AutoencoderKL(**config).to(device)

    state_dict = safetensors.torch.load_file(weights_path, device=str(device))
    vae.load_state_dict(normalize_legacy_attention_keys(state_dict))
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def normalize_legacy_attention_keys(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Rename legacy VAE-attention keys to the current diffusers layout.

    Older Stability-published SD 1.5 VAE weights use the pre-refactor
    attention keys (``query``, ``key``, ``value``, ``proj_attn``).
    Current diffusers expects (``to_q``, ``to_k``, ``to_v``,
    ``to_out.0``). SDXL and Flux VAE weights are already published in
    the new layout — the rename is a no-op for them.

    The tensor values are unchanged; only key names are rewritten.
    """
    renamed: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key
        for old_suffix, new_suffix in _LEGACY_ATTENTION_RENAMES.items():
            if key.endswith(old_suffix):
                new_key = key[: -len(old_suffix)] + new_suffix
                break
        renamed[new_key] = tensor
    return renamed


def _prefix(subfolder: str | None, filename: str) -> str:
    return filename if subfolder is None else f"{subfolder}/{filename}"
