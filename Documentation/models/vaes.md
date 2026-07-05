# Diffusion VAE ensemble

Every Stable Diffusion / SDXL / Flux based generator encodes the input
image through a VAE into a compressed latent, denoises in latent space,
and decodes back. Nudify inpainting, img2img face-swap, and IP-Adapter
style personalization all pass through the VAE.

If we can drive the VAE latent toward a null target for a Voidface-cloaked
image, downstream generation decodes something structurally unrelated to
the source photo.

**Currently shipped (R4.1 - R4.2):** SD 1.5 VAE and SDXL VAE. Both
loaded through the shared `_diffusers_loader` bypass at
`src/voidface/models/vaes/_diffusers_loader.py` (the R4.1 refactor
that broke a diffusers 0.29 + torch 2.2 state-dict loading bug).
Legacy attention keys (`query`/`key`/`value`/`proj_attn`) renamed
to the modern layout (`to_q`/`to_k`/`to_v`/`to_out.0`).

Weights:
- SD 1.5 VAE: `stabilityai/sd-vae-ft-mse` (334 MB, standalone repo
  since the full SD 1.5 checkpoint has gated files).
- SDXL VAE: `madebyollin/sdxl-vae-fp16-fix` (334 MB, the community-
  standard fp16-stable variant).

| VAE           | Latent shape (512²) | Family                | Weight | Status         |
| ------------- | ------------------- | --------------------- | ------ | -------------- |
| SD 1.5 VAE    | 4 × 64 × 64         | Automatic1111 / SD    | 0.30   | ✅ shipped R2  |
| SDXL VAE      | 4 × 64 × 64 (256²)  | SDXL, InstantID base  | 0.25   | ✅ shipped R4.2|
| Flux VAE      | 16 × 32 × 32        | Flux, SD3             | 0.20   | roadmap        |

The attack objective per VAE is:

    L_vae = || E(x + delta) - z_target ||²

with `z_target = E(gray_0.5)` (a constant, flat, mid-gray latent).

Transfer between VAE families is poor. SD 1.5 → SDXL is 20–40%; SD 1.5
→ Flux is 10–25%. The Flux VAE's 16-channel latent has ~4× the capacity
of the SD 1.5 VAE, which is why achievable disruption at fixed epsilon
is smaller and its ensemble weight is lower.

Source lives in `src/voidface/models/vaes/`.
