# Diffusion VAE ensemble

Every Stable Diffusion / SDXL / Flux based generator encodes the input
image through a VAE into a compressed latent, denoises in latent space,
and decodes back. Nudify inpainting, img2img face-swap, and IP-Adapter
style personalization all pass through the VAE.

If we can drive the VAE latent toward a null target for a Voidface-cloaked
image, downstream generation decodes something structurally unrelated to
the source photo.

| VAE           | Latent shape (512²)  | Family                | Weight |
| ------------- | -------------------- | --------------------- | ------ |
| SD 1.5 VAE    | 4 × 64 × 64          | Automatic1111 / A11    | 0.30   |
| SDXL VAE      | 4 × 128 × 128        | SDXL, InstantID base  | 0.25   |
| Flux VAE      | 16 × 32 × 32         | Flux, SD3             | 0.20   |

The attack objective per VAE is:

    L_vae = || E(x + delta) - z_target ||²

with `z_target = E(gray_0.5)` (a constant, flat, mid-gray latent).

Transfer between VAE families is poor. SD 1.5 → SDXL is 20–40%; SD 1.5
→ Flux is 10–25%. The Flux VAE's 16-channel latent has ~4× the capacity
of the SD 1.5 VAE, which is why achievable disruption at fixed epsilon
is smaller and its ensemble weight is lower.

Source lives in `src/voidface/models/vaes/`.
