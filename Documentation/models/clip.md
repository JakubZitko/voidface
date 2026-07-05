# CLIP-family image encoder ensemble

Modern personalization pipelines — IP-Adapter, InstantID, PhotoMaker,
PuLID, ID-Adapter — extract identity via a CLIP-family image encoder and
cross-attend it into the diffusion UNet. Attacking these encoders breaks
the identity-conditioning path for a large share of the personalization
literature.

| Encoder         | Architecture | Pool style             | Weight |
| --------------- | ------------ | ---------------------- | ------ |
| OpenCLIP ViT-H  | ViT-H/14     | CLS token (+ patches)  | 0.20   |
| SigLIP ViT-L    | ViT-L/16     | MAP pooling            | 0.15   |
| DINOv2 ViT-L    | ViT-L/14     | Self-supervised CLS    | 0.15   |

Attack objectives:

- Maximize `1 - cos(g(x + delta), g(x))` for each encoder.
- For IP-Adapter's Plus / Perceiver-Resampler input, we also attack the
  257-token patch sequence directly, not only the pooled CLS.

Transfer:

- OpenCLIP → SigLIP: 40–60%. Attack directly.
- OpenCLIP → DINOv2: 20–40%. Attack directly.
- Attacking all three jointly transfers to EVA-CLIP variants at 60–80%.

DINOv2 is the least adversarially robust of the three (task-agnostic
PGD at 4/255 rotates the CLS token by ~90°) and provides cheap
cross-algorithm diversity in the ensemble.

Source lives in `src/voidface/models/clip/`.
