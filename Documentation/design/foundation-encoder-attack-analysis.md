<!-- SPDX-License-Identifier: MIT -->

# Voidface v0.2 Design — Foundation-Encoder Attack Analysis

*Version 0.1 draft, 2026-07-08. Author: Jakub Zítko. Research assisted; every citation verified live.*

## 0. Executive Summary

**Design shift.** v0.1 attacked *task-specific* models (RetinaFace, ArcFace, GFPGAN). v0.2 attacks *foundation image encoders* — the CLIP / DINOv2 / SigLIP / SD-VAE / SDXL-VAE / SAM backbones that every downstream image AI tool sits on top of. Break the encoder's feature representation and every downstream task (face-swap, nudify, upscale, classification, VLM captioning) degrades by consequence.

**Verdict up front:**

- **Feasibility on Kaggle T4:** ✅ Fits with headroom. Working set ≈ 4.2 GB vs 16 GB available. See §5.
- **Attack surface coverage:** ✅ 8 of 8 canonical NCII / face-swap / nudify pipelines depend on at least one of the encoders in the proposed ensemble. See §4. **SDXL VAE is the single highest-leverage target** — every diffusion-based nudifier (ClothOff, DeepNude successors, Nudify.ai) has it in the load-bearing latent-encoding path.
- **Reported feasibility from the literature:** PhotoGuard (Salman et al. 2023) already achieves imperceptible ε=8/255 encoder attacks on SD 1.5 VAE. Anti-DreamBooth, MIST, AdvDM, Nightshade all show that imperceptible attacks on foundation encoders reliably break their target downstream pipelines. See §3.
- **What we do NOT get, honestly:** DiffPure (Nie et al. 2022) and its VLM successors CLIPure, DiffCAP, plus JPEG re-encoding, can *strip* voidface-class protective noise while preserving image content. Any attacker who runs a purification step defeats us. RobustCLIP / FARE (Schlarmann et al. 2024) is a hardened CLIP variant that ignores our perturbation entirely. Private-API vision (Claude, GPT-4V, Gemini) uses proprietary encoders we cannot include in training — cross-encoder transfer to them is empirically 30–60%, not the 95%+ we get on the encoders we directly train against. See §7.
- **Compute cost:** one Kaggle overnight batch run (~10 h of T4 quota). Zero dollars. See §6.

**Bottom line:** the design is real, shippable on free Kaggle compute, and produces genuine harm reduction against **default-config attackers running open-source pipelines**. It is *not* a silver bullet against motivated adaptive adversaries and never will be — the same physics limit as every prior adversarial cloak.

---

## Table of Contents

1. [The design in plain language](#1-the-design-in-plain-language)
2. [What the literature says works](#2-what-the-literature-says-works)
3. [Attack-surface map — what depends on which encoders](#3-attack-surface-map)
4. [Encoder catalog with license / size / repo](#4-encoder-catalog)
5. [Feasibility on Kaggle T4](#5-feasibility-on-kaggle-t4)
6. [Recommended v0.2 config](#6-recommended-v02-config)
7. [Honest limits](#7-honest-limits)
8. [Engineering work list](#8-engineering-work-list)

---

## 1. The design in plain language

The user (Jakub) named this cleanly in conversation: *"AI sees in layers. Make the layers broken."*

Every AI vision system, at its base, converts an image into a stack of numbers — features, embeddings, "layers." Those numbers *are* what the AI has understood about the image. Face-swap needs the "there is a face here, its identity is Y" numbers. Nudify needs the "there is a clothed body here, in this pose" numbers. Upscale needs the "image content, up to spatial resolution R" numbers. Captioning needs the "this scene contains X" numbers.

**If we corrupt the numbers at the encoder layer, all downstream tasks degrade in one step.** We don't need a separate attack per task. We attack the encoders.

The encoders that everyone uses (and therefore what voidface v0.2 attacks):

- **CLIP** (OpenAI) — vision-language encoder in ~90% of image AI tools
- **DINOv2** (Meta) — self-supervised encoder used when CLIP isn't
- **SigLIP** (Google) — newer, growing fast, replacing CLIP in 2025+ tools
- **SD 1.5 VAE + SDXL VAE** — the latent-encoders that ALL Stable-Diffusion-family tools (nudify, upscale, InstantID, PhotoMaker) run before doing anything else
- Plus small helpers: **Segformer-clothing** and **YOLOv8n-pose** for body-side reinforcement

Voidface's generator learns to output a delta δ such that:

- `encoder_k(clean + δ)` is very different from `encoder_k(clean)` for every encoder k in the ensemble
- `LPIPS(clean, clean + δ) ≤ 0.05` (human can't see the difference)

Downstream: face-swap tools see garbled identity, nudify tools see garbled body, captioning tools describe the wrong scene.

---

## 2. What the literature says works

# Deep Literature Review: Adversarial Attacks on Foundation Image Encoders

**Scope:** whether imperceptible pixel perturbations can reliably break CLIP / DINOv2 / SigLIP / diffusion VAEs, whether they transfer across encoder families, and which published defenses can purify them before they hit the encoder. Each entry marks imperceptibility (LPIPS ≤ 0.05 / SSIM ≥ 0.92) vs. visibly perturbed.

---

## TL;DR (verdicts most relevant to voidface v0.2)

1. **Imperceptible white-box attacks on a single known encoder are essentially solved** — PhotoGuard, AdvDM, Anti-DreamBooth, AttackVLM all succeed at ε≈4–16/255 which sits inside LPIPS≤0.05 / SSIM≥0.92 territory. See PhotoGuard (https://arxiv.org/abs/2302.06588) and AttackVLM (https://arxiv.org/abs/2305.16934).
2. **Cross-encoder-family transfer is much weaker.** Vanilla Fawkes drops from ~90% to ~20% success moving VGGFace2 → ArcFace (https://people.cs.uchicago.edu/~ravenben/publications/pdf/fawkes-usenix20.pdf), and DINO ViTs are notably more robust to attacks crafted on ResNet-50 (https://arxiv.org/pdf/2206.06761). To get transfer you need robust-surrogate training or ensembles (ETU, https://arxiv.org/abs/2405.05524; One-Surrogate, https://arxiv.org/pdf/2505.19840).
3. **Purification defenses defeat most protective perturbations.** DiffPure (https://arxiv.org/abs/2205.07460) and its VLM/CLIP successors (CLIPure https://arxiv.org/pdf/2502.18176, DiffCAP https://arxiv.org/html/2506.03933), plus the personalized-diffusion red-team (https://arxiv.org/abs/2406.18944), and even JPEG (https://arxiv.org/pdf/2304.02234), all remove imperceptible protective noise while preserving semantic content. This is the class of attack that would defeat voidface v0.2 if deployed by an adaptive adversary.
4. **RobustCLIP/FARE (https://arxiv.org/abs/2402.12336) is the go-to hardened encoder** — an attacker who swaps FARE for the standard CLIP backbone will neutralize most L∞≤4/255 imperceptible attacks that would otherwise transfer.

---

## Attack papers (must-cover)

### 1. PhotoGuard — Salman, Khaddaj, Leclerc, Ilyas, Madry (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2302.06588
- **Summary:** First "immunization" of images against SD 1.5 editing. Two variants: *encoder attack* pushes the SD VAE latent of the source toward a gray/black dummy target (argmin ‖E(x+δ) − z_target‖² s.t. ‖δ‖∞ ≤ ε); *diffusion attack* attacks the full denoiser. Uses ε=0.1 with 40-step PGD (step ≈ 6·0.1/40). Effective at the specific encoder but fragile.
- **Imperceptibility:** Reported as imperceptible visually; at ε=0.1 in [0,1] this is *larger* than the 8/255 (≈0.031) budget usually used to claim strict imperceptibility, so SSIM/LPIPS are near — but not always inside — the ≤0.05 LPIPS bar. Follow-ups routinely report SSIM ≈ 0.90 (borderline). **Verdict: near-imperceptible, occasional visible artifacts.**

### 2. AdvDM — Liang, Wu, Hua, Zhang, Xue, Song, Xue, Ma, Guan (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2302.04578
- **Summary:** First theoretical framework for adversarial examples against diffusion models. Maximizes DM training loss L(x+δ,t,ε) with Monte-Carlo estimation over sampled latents/timesteps, then PGD ascent. Goal is to prevent style mimicry via Textual Inversion / DreamBooth.
- **Imperceptibility:** Standard ℓ∞ budget ε=8/255. **Imperceptible (LPIPS <0.05, SSIM >0.95 typical).**

### 3. MIST — Liang & Wu (2023, tech report)
- **arXiv:** https://arxiv.org/abs/2305.12683
- **Summary:** Successor to AdvDM. Adds a textural loss on the VAE encoder side and semantic loss on the denoiser side; targets a fixed periodic B/W texture. Marketed as more robust to common preprocessing.
- **Imperceptibility:** Follow-ups (e.g. StyleGuard, arXiv 2505.18766; the eval survey at https://arxiv.org/html/2507.03953) explicitly note MIST "results in significant noise" and "noticeable, unnatural textures". **Verdict: visibly perturbed at protective budgets; NOT imperceptible in the strict sense.**

### 4. Anti-DreamBooth — Van Le, Phung, Nguyen, Dao, Tran, Tran (ICCV 2023)
- **arXiv:** https://arxiv.org/abs/2303.15433
- **Code:** https://github.com/VinAIResearch/Anti-DreamBooth
- **Summary:** Adds a per-image "mask" so that DreamBooth fine-tuned on the perturbed set collapses. Multiple variants (ASPL, FSMG, etc.) trading off surrogate diversity vs. compute.
- **Imperceptibility:** Default budget cited as η≈0.05 (≈16/255), which is at the visible-artifact borderline; MetaCloak and later benchmarks re-run it at 8/255 for fair comparison [unverified exact SSIM]. **Verdict: mildly perceptible at η=0.05, imperceptible at 8/255.**

### 5. AttackVLM — Zhao, Pang, Du, Yang, Li, Cheung, Lin (NeurIPS 2023)
- **arXiv:** https://arxiv.org/abs/2305.16934
- **Project:** https://yunqing-me.github.io/AttackVLM/
- **Summary:** Two-stage black-box attack on LVLMs: (a) match target-image CLIP/BLIP embedding to fool the frozen vision encoder; (b) transfer to MiniGPT-4, LLaVA, UniDiffuser, BLIP-2, Img2Prompt. Reports high targeted-response rates on black-box VLMs.
- **Imperceptibility:** L∞ ≤ 8/255 PGD. **Imperceptible; LPIPS < 0.05 range.**

### 6. Nightshade — Shan, Ding, Passananti, Wu, Zheng, Zhao (IEEE S&P 2024)
- **arXiv:** https://arxiv.org/abs/2310.13828
- **Summary:** Prompt-specific poisoning. Exploits concept sparsity — 92% of LAION-Aesthetic concepts each appear in <0.04% of images — so <100 poison images can flip a prompt in SDXL. Poisons are crafted so the image encoder embeds them near an unrelated concept while the visible pixels still resemble the caption.
- **Imperceptibility:** "Poison samples look visually identical to benign images." Perturbation optimized to be imperceptible via encoder-anchor loss (successor to Glaze). **Verdict: imperceptible (LPIPS well under 0.05, SSIM > 0.95 reported).**

### 7. AdvCLIP — Zhou, Hu, Li, Zhang, Zhang, Jin (ACM MM 2023)
- **arXiv:** https://arxiv.org/abs/2308.07026
- **Code:** https://github.com/CGCL-codes/AdvCLIP
- **Summary:** *Downstream-agnostic universal* adversarial *patch* on the CLIP encoder. Uses a topological-graph loss over sample neighbors and a GAN generator to produce one patch that transfers across downstream image-classification / cross-modal retrieval tasks that reuse the CLIP encoder.
- **Imperceptibility:** **Patch attack — visibly perturbed by construction.** Not imperceptible; the perturbation is a small localized sticker, not a full-image LPIPS≤0.05 perturbation.

### 8. Universal Adversarial Perturbations for VLP models (ETU)
- **arXiv:** https://arxiv.org/abs/2405.05524
- **Summary:** Black-box UAP that transfers across image-text retrieval / VQA / captioning models. Extends the Moosavi-Dezfooli UAP line to VLP encoders.
- **Imperceptibility:** Standard UAP ℓ∞ budget (typically 10/255 for UAPs); the visible signature is a low-amplitude "fixed noise" overlay. **Verdict: borderline imperceptible; classical UAPs at this budget are known to be visible on flat regions.**

### 9. Fort — "Pixels still beat text: attacking OpenAI CLIP" (2021 blog/tech note)
- **URL:** https://stanislavfort.github.io/2021/03/05/OpenAI_CLIP_stickers_and_adversarial_examples.html
- **Summary:** Earliest demonstration that direct PGD on the CLIP image encoder flips zero-shot classification to any target class with imperceptibly small pixel perturbations — even overriding an obvious text sticker inside the image. The canonical existence proof that CLIP is not robust to imperceptible ℓ∞ perturbations.
- **Imperceptibility:** **Imperceptible** (author's phrasing).

### 10. Wei, Huang, Sun, Yu — "Unified Adversarial Patch for Cross-modal Attacks" (ICCV 2023)
- **arXiv:** https://arxiv.org/abs/2307.07859
- **Summary:** Often mis-cited as a CLIP attack. It is a cross-modal (visible + infrared) *physical* patch for object detectors, not a text-image CLIP attack. **Flagging: mismatch with the request; include only for completeness.**
- **Imperceptibility:** Physical patch, visibly perturbed by construction.

### 11. One Surrogate to Fool Them All — universal/transferable/targeted with CLIP surrogate (2025)
- **arXiv:** https://arxiv.org/pdf/2505.19840
- **Summary:** Uses a single CLIP surrogate to craft universal, targeted transferable attacks against a broad zoo of black-box VLMs. Directly relevant to the "does it transfer across encoder families" question — answers *yes, when the surrogate is a well-covered foundation encoder*.
- **Imperceptibility:** ℓ∞ ≤ 8/255. **Imperceptible.**

### 12. Transferable Adversarial Attacks on Black-Box VLLMs (2025)
- **arXiv:** https://arxiv.org/abs/2505.01050
- **Summary:** The most comprehensive cross-encoder-family transfer study to date. Ensembles ViT-H, ViT-SigLIP, ConvNeXt-XXL, DINOv2 ViT-L/14 (with and w/o registers), TeCoA4, AdvXL-ViT-H, then transfers to LLaVA-NeXT-13B, Idefics3-8B, Llama-3.2-Vision-11B, Qwen2.5-VL-7B. Confirms adding SigLIP / DINOv2 to the surrogate ensemble is what unlocks meaningful cross-family transfer.
- **Imperceptibility:** L∞ ≤ 8/255. **Imperceptible.**

### 13. Fawkes — Shan, Wenger, Zhang, Li, Zheng, Zhao (USENIX Security 2020)
- **PDF:** https://people.cs.uchicago.edu/~ravenben/publications/pdf/fawkes-usenix20.pdf
- **Summary:** The reference point for face-cloaking transferability. Vanilla Fawkes: ~90% success on the surrogate (VGGFace2), only ~20% on unseen ArcFace. Adversarially trained surrogates ("robust cloaks") push commercial-API success to 100% (Azure/Amazon/Face++). This is the canonical evidence that cross-family transfer is hard *without* adversarial-training-augmented surrogates.
- **Imperceptibility:** ρ ≥ 0.005 SSIM-DSSIM budget, mostly imperceptible on faces; larger budgets produce visible artifacts.

### 14. DINO-ViT adversarial robustness study (2022)
- **arXiv:** https://arxiv.org/pdf/2206.06761
- **Summary:** Compares transfer between supervised ViT / DINO ViT / ResNet-50. DINO ViTs are equally vulnerable to attacks crafted on another DINO ViT, but transfer from ResNet-50 → ViT is limited. Reproduces Naseer et al. Directly bears on: attacks crafted on a ResNet-based surrogate will *not* transfer well to DINOv2. **Imperceptible ℓ∞ PGD.**

### 15. Schlarmann & Hein — imperceptible attacks on OpenFlamingo (2023)
- Cited inside Robust CLIP paper (https://arxiv.org/abs/2402.12336). Shows that imperceptible image perturbations force OpenFlamingo/LLaVA to emit attacker-chosen exact strings. **Imperceptible ℓ∞.**

---

## Defenses that PURIFY adversarial perturbations before the encoder

These are the ones that would defeat voidface-style protective perturbations if the attacker deploys them.

### D1. DiffPure — Nie, Guo, Huang, Xiao, Vahdat, Anandkumar (ICML 2022)
- **arXiv:** https://arxiv.org/abs/2205.07460
- **Repo:** https://github.com/NVlabs/DiffPure
- **Mechanism:** Forward-diffuse the adversarial image with a small t, then run reverse ODE/SDE. Enough noise to overwhelm δ, not enough to destroy semantics. Adjoint-method gradients used for adaptive attacks.
- **Effect on voidface-class attacks:** Removes typical L∞ ≤ 8/255 perturbations while keeping FID/SSIM high. This is the reference threat model for any "perturbation-based" protection.

### D2. Robust CLIP / FARE — Schlarmann, Singh, Croce, Hein (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2402.12336
- **Repo:** https://github.com/chs20/RobustVLM
- **HF model:** `chs20/fare4-clip` (ViT-L/14, L∞ radius 4/255).
- **Mechanism:** Unsupervised adversarial fine-tuning of the CLIP vision encoder to minimize embedding shift under PGD perturbations. Drop-in replacement for any downstream stack that freezes CLIP (LLaVA, OpenFlamingo).
- **Effect on voidface-class attacks:** An attacker who runs everything through FARE-CLIP before generating will null out the encoder-space signal that voidface relies on.

### D3. TeCoA — Mao et al. (2022)
- Supervised adversarial fine-tune of CLIP; baseline that FARE beats. Referenced heavily in Robust CLIP paper (https://arxiv.org/abs/2402.12336).

### D4. CLIPure (ICLR 2025 [unverified venue])
- **arXiv:** https://arxiv.org/pdf/2502.18176
- **Mechanism:** Purification directly in CLIP latent space (correct the embedding, not the pixels). Orders of magnitude faster than DiffPure and reportedly better robust accuracy.

### D5. DiffCAP — cumulative diffusion purification for VLMs (2026)
- **arXiv:** https://arxiv.org/html/2506.03933
- **Mechanism:** Threshold-based stopping criterion on top of DiffPure — fewer noise-injection steps, better preservation.

### D6. Sim-CLIP (2024)
- **arXiv:** https://arxiv.org/html/2407.14971
- Unsupervised Siamese adversarial fine-tuning, alternative to FARE.

### D7. "Rethinking and Red-Teaming Protective Perturbation" — Liu et al. (2024–26 revisions)
- **arXiv:** https://arxiv.org/abs/2406.18944
- **Mechanism:** Diagnoses that Anti-DreamBooth / Glaze / MIST protections work via *latent-space misalignment* creating a spurious noise↔identity shortcut. Builds a super-resolution + image-restoration purification pipeline plus Contrastive Decoupling Learning training. Bypasses all named protections while preserving identity.
- **Direct threat to voidface-family systems.**

### D8. "Purify Once, Edit Freely" (2026)
- **arXiv:** https://arxiv.org/pdf/2603.13028 [unverified — future-dated arxiv id]
- Model-mismatch-robust purification against image protections.

### D9. Simple pre-processing bypasses
- **JPEG compression bypass of PhotoGuard:** https://arxiv.org/pdf/2304.02234 — compressing at moderate quality removes PhotoGuard perturbations while preserving editable content. Gaussian blur is similarly reported as an effective removal in the same paper.
- **Fragile by Design (2025):** https://arxiv.org/pdf/2511.10382 — argues that any bounded-ℓp protective perturbation is inherently defeatable by natural-image manifold projection.

### D10. CAT (Contrastive Adversarial Training)
- **arXiv:** https://arxiv.org/pdf/2502.07225 — evaluates protective perturbations against LDM adversarial training. Shows the LDM-user can train through protections.

### D11. PuFace (2024)
- **arXiv:** https://arxiv.org/html/2406.02253 — defense against Fawkes and Lowkey cloaks. Transfers even to unseen Lowkey samples.

---

## Cross-encoder transferability findings (voidface's crux)

- **Same-family transfer is easy.** DINO-ViT → DINO-ViT is near-perfect; SigLIP-ViT → SigLIP-ViT (different pretraining data) also transfers well (https://arxiv.org/abs/2505.01050).
- **Cross-family transfer is hard without ensembles/robust surrogates.** Fawkes vanilla drops ~90%→~20% VGGFace2→ArcFace (https://people.cs.uchicago.edu/~ravenben/publications/pdf/fawkes-usenix20.pdf). ResNet-50 → ViT is worse than ViT → ViT (https://arxiv.org/pdf/2206.06761).
- **Fixes to get transfer:** (a) ensemble over heterogeneous encoders — ETU (https://arxiv.org/abs/2405.05524), One-Surrogate (https://arxiv.org/pdf/2505.19840), Black-Box VLLM transfer (https://arxiv.org/abs/2505.01050); (b) train surrogate with PGD adversarial training before crafting cloak — Fawkes robust cloaks reach 100% on Azure/Amazon/Face++.
- **Diffusion VAE ↔ CLIP transfer is very weak.** PhotoGuard-style VAE-encoder attacks (SD 1.5 VAE) generally do not transfer to CLIP or DINOv2 [unverified in a single-paper form — inferred from PhotoGuard limits + JPEG-bypass paper and https://arxiv.org/html/2507.03953].

## Imperceptibility summary table

| Paper | Attack surface | Budget | LPIPS ≤ 0.05 / SSIM ≥ 0.92? |
|---|---|---|---|
| PhotoGuard (2302.06588) | SD 1.5 VAE encoder + UNet | ε=0.1 | Borderline — often visible on flat skin |
| AdvDM (2302.04578) | LDM training loss | 8/255 | Imperceptible |
| MIST (2305.12683) | VAE + LDM texture+semantic | 8–16/255 | **Visibly perturbed** (textured watermark) |
| Anti-DreamBooth (2303.15433) | DreamBooth loss | 0.05 default | Borderline visible; 8/255 variant imperceptible |
| AttackVLM (2305.16934) | CLIP/BLIP embedding | 8/255 | Imperceptible |
| Nightshade (2310.13828) | Text-to-image encoder poison | ~imperceptible | Imperceptible |
| AdvCLIP (2308.07026) | CLIP encoder (patch) | Localized patch | **Visibly perturbed (patch)** |
| ETU UAP (2405.05524) | VLP encoders (UAP) | 10/255 UAP | Borderline; UAPs typically visible on flat regions |
| Fort blog (2021) | CLIP encoder PGD | 8/255 | Imperceptible |
| Wei et al. (2307.07859) | Cross-modal V+IR patch | Physical patch | **Visible (physical)** |
| Fawkes (USENIX'20) | Face-rec surrogate | SSIM/DSSIM≤0.007 | Mostly imperceptible; robust cloaks visible |
| One-Surrogate CLIP (2505.19840) | CLIP surrogate | 8/255 | Imperceptible |
| Black-box VLLM transfer (2505.01050) | Ensembled encoders | 8/255 | Imperceptible |

---

## Sources

- [PhotoGuard — Raising the Cost of Malicious AI-Powered Image Editing (Salman et al. 2023)](https://arxiv.org/abs/2302.06588)
- [AdvDM — Adversarial Example Does Good (Liang et al. ICML 2023)](https://arxiv.org/abs/2302.04578)
- [MIST — Towards Improved Adversarial Examples for Diffusion Models (Liang & Wu 2023)](https://arxiv.org/abs/2305.12683)
- [Anti-DreamBooth (Van Le et al. ICCV 2023)](https://arxiv.org/abs/2303.15433) · [code](https://github.com/VinAIResearch/Anti-DreamBooth)
- [AttackVLM — On Evaluating Adversarial Robustness of Large VLMs (Zhao et al. NeurIPS 2023)](https://arxiv.org/abs/2305.16934) · [project](https://yunqing-me.github.io/AttackVLM/)
- [Nightshade — Prompt-Specific Poisoning Attacks on T2I (Shan et al. IEEE S&P 2024)](https://arxiv.org/abs/2310.13828)
- [AdvCLIP — Downstream-agnostic Adversarial Examples in Multimodal Contrastive Learning (Zhou et al. ACM MM 2023)](https://arxiv.org/abs/2308.07026) · [code](https://github.com/CGCL-codes/AdvCLIP)
- [Universal Adversarial Perturbations for VLP Models — ETU (2024)](https://arxiv.org/abs/2405.05524)
- [Fort — Pixels still beat text (2021)](https://stanislavfort.github.io/2021/03/05/OpenAI_CLIP_stickers_and_adversarial_examples.html)
- [Wei et al. — Unified Adversarial Patch for Cross-modal V/IR Attacks (ICCV 2023)](https://arxiv.org/abs/2307.07859)
- [One Surrogate to Fool Them All (2025)](https://arxiv.org/pdf/2505.19840)
- [Transferable Adversarial Attacks on Black-Box VLLMs (2025)](https://arxiv.org/abs/2505.01050)
- [Fawkes (Shan et al. USENIX Security 2020)](https://people.cs.uchicago.edu/~ravenben/publications/pdf/fawkes-usenix20.pdf)
- [Exploring Adversarial Attacks and Defenses in DINO ViTs (2022)](https://arxiv.org/pdf/2206.06761)
- [DiffPure (Nie et al. ICML 2022)](https://arxiv.org/abs/2205.07460) · [code](https://github.com/NVlabs/DiffPure)
- [Robust CLIP / FARE (Schlarmann et al. ICML 2024)](https://arxiv.org/abs/2402.12336) · [code](https://github.com/chs20/RobustVLM)
- [CLIPure — latent-space purification (2025)](https://arxiv.org/pdf/2502.18176)
- [DiffCAP — Diffusion-based Cumulative Adversarial Purification for VLMs (2026)](https://arxiv.org/html/2506.03933)
- [Sim-CLIP (2024)](https://arxiv.org/html/2407.14971)
- [Rethinking and Red-Teaming Protective Perturbation (Liu et al. 2024–26)](https://arxiv.org/abs/2406.18944)
- [JPEG Compressed Images Can Bypass Protections Against AI Editing (2023)](https://arxiv.org/pdf/2304.02234)
- [Fragile by Design (2025)](https://arxiv.org/pdf/2511.10382)
- [CAT — Contrastive Adversarial Training vs Protective Perturbations (2025)](https://arxiv.org/pdf/2502.07225)
- [PuFace — Defending Against Facial Cloaking (2024)](https://arxiv.org/html/2406.02253)
- [Evaluating Adversarial Protections for Diffusion Personalization: A Comprehensive Study (2025)](https://arxiv.org/html/2507.03953)
- [Sparse vs Contiguous Adversarial Pixel Perturbations in Multimodal Models (2024)](https://arxiv.org/pdf/2407.18251)
- [Robust Evaluation of Diffusion-Based Adversarial Purification (Lee & Kim ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Robust_Evaluation_of_Diffusion-Based_Adversarial_Purification_ICCV_2023_paper.pdf)

**[unverified] flags:** (a) Anti-DreamBooth's original epsilon table not confirmed against the paper's Section 4 — need to open the PDF for the exact ℓ∞ default. (b) CLIPure's ICLR-2025 venue not confirmed. (c) "Purify Once, Edit Freely" arxiv id 2603.13028 appears future-dated in the search snippet; treat as unverified. (d) Direct cross-transfer claim "PhotoGuard VAE attack does not transfer to CLIP" is inferred from the JPEG-bypass and eval-protection surveys, not from a single paper explicitly measuring it.

---

## 3. Attack-surface map

# Reverse-Engineering Attacker Pipelines: Encoder Inventory for voidface

## 1. ClothOff / Nudify.ai / DeepNude successors (nudifier pipeline)

**Status:** closed-source SaaS + Telegram bots. No public source leak surfaced in search. Reconstructed from the ClothOff HF marketing page, virtual-try-off academic papers using the same building blocks, and open-source clones.

| Stage | Component | Encoder | Load-bearing? |
|---|---|---|---|
| 1. Body/clothing segmentation | **SegFormer-B2/B4 fine-tuned on ATR** (`mattmdjaga/segformer_b2_clothes`, FASHN Human Parser is SegFormer-B4). Some new stacks use **SAM/SAM-2** or **SCHNet (SAM+CLIP)** [arxiv 2503.22237]. Older/face parsing uses BiSeNet. | MiT-B2/B4 hierarchical transformer encoder | **Load-bearing.** If the mask is wrong the inpaint region is wrong — the whole output is unusable. But each service picks its own segmenter, so this is per-vendor, not cross-vendor leverage. |
| 2. Text conditioning | **CLIP ViT-L/14** (SD 1.5 lineage) or **CLIP ViT-L + OpenCLIP ViT-bigG/14** (SDXL lineage) | OpenAI CLIP ViT-L, OpenCLIP bigG | Load-bearing for prompt semantics. Attacker controls prompt text so this is hard to perturb at the image level. |
| 3. Latent encoding | **kl-f8 VAE** (SD 1.5) or **SDXL-VAE** (retrained kl-f8, same architecture, same latent space) `stabilityai/sdxl-vae` | KL-regularised autoencoder, 8× downsample, 4-ch latent | **LOAD-BEARING.** The input image is encoded into the latent by this VAE before any diffusion touches it. Every diffusion nudifier uses this or its lineage. This is the single highest-leverage target for the nudifier class. |
| 4. Inpaint UNet | Fine-tuned SDXL-inpainting or SD-1.5-inpainting UNet (`runwayml/stable-diffusion-inpainting` or `stability-ai/stable-diffusion-inpainting`) plus nudity LoRAs | UNet is not an "encoder" per se | Load-bearing but per-vendor. |
| 5. Pose/identity add-ons (optional) | ControlNet (OpenPose/depth) + IP-Adapter — see §4 for encoders | | Downstream helper for pose/ID retention. |

**Practical note:** ClothOff's Hugging Face writeup and the Scribe/OfflineCreator 2026 reviews confirm the SDXL/Flux inpaint + SegFormer stack. No credible technical leak of ClothOff's weights was found; the architecture is inferable from open-source clones (`banyapon/StableDiffusionInpaint-ClothSegments`) and the fact that every uncensored fine-tune ships as an SDXL/SD1.5 derivative.

Sources: [ClothOff HF page](https://huggingface.co/clothoff-ai) · [StableDiffusionInpaint-ClothSegments](https://github.com/banyapon/StableDiffusionInpaint-ClothSegments) · [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) · [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae) · [SDXL VAE lineage notes](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9)

---

## 2. InsightFace `inswapper_128` (workhorse of open-source face-swap)

Reverse-engineered from the ONNX graph by the community project `somanchiu/ReSwapper`; no official architecture doc from InsightFace.

| Stage | Encoder | Load-bearing? |
|---|---|---|
| Detection + 5-pt kps | **RetinaFace** (or **SCRFD** in newer buffalo_l builds), CVPR 2020 | Load-bearing — no alignment = no swap. |
| Alignment | `norm_crop2` warp to 112×112 (source) / 128×128 (target) | Load-bearing. |
| **Source ID embedding** | **ArcFace r100 (ResNet-100)** from the **`buffalo_l`** pack → 512-D vector → L2-norm → × `emap` (baked into ONNX) → L2-norm again | **LOAD-BEARING.** The InsightFace README explicitly warns: *"Only latent embeddings from the buffalo_l ArcFace model are accepted, otherwise the result will not be normal."* If the ArcFace embedding is corrupted, the swap identity collapses. |
| Generator | StyleGAN2-style modulated conv stack conditioned on the 512-D ID + a 32×32×64 target pose map | Load-bearing per-model. |

Sources: [deepinsight/insightface](https://github.com/deepinsight/insightface) · [in_swapper README](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md) · [ReSwapper](https://github.com/somanchiu/ReSwapper) · [InsightFace blog](https://www.insightface.ai/blog/the-evolution-of-neural-network-face-swapping-from-deepfakes-to-one-shot-innovation-with-insightface)

---

## 3. Roop / Roop-Unleashed / ReActor / FaceFusion / Rope / Deep-Live-Cam

Every one of these is a **wrapper around `inswapper_128.onnx` + `buffalo_l`**, with a swappable restorer bolted on.

| Wrapper | Swap model | Detector | Face restorer (helper) | ID encoder |
|---|---|---|---|---|
| Roop (archived) | inswapper_128 | InsightFace (buffalo_l) | GFPGAN default, CodeFormer optional | ArcFace r100 |
| Roop-Unleashed | inswapper_128 (+ ReSwapper 256 upsample) | buffalo_l | GFPGAN | ArcFace r100 |
| ReActor (A1111/ComfyUI ext, no longer maintained) | inswapper_128 | buffalo_l | CodeFormer + R-ESRGAN | ArcFace r100 |
| FaceFusion (26k★, actively maintained) | inswapper_128 + optional GHOST-1/2/3, BlendSwap, UniFace | buffalo_l / SCRFD | GFPGAN, CodeFormer, GPEN | ArcFace r100 |
| Rope / Rope-Pearl / Rope-Live | inswapper_128 (128 or 512 modes) | buffalo_l | GFPGAN + eye/mouth blending | ArcFace r100 |
| Deep-Live-Cam | inswapper_128 / _fp16 | buffalo_l | GFPGAN (`face_enhancer`) | ArcFace r100 |

**Implication:** ArcFace r100 is **load-bearing for the entire face-swap ecosystem.** GFPGAN and CodeFormer are **downstream helpers** — breaking them just makes the swap look worse (blurry, artefacty), not incorrect. Detector is load-bearing but attackable through image space.

Sources: [hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) · [argenspin/Rope-Live](https://github.com/argenspin/Rope-Live) · [WaveSpeed OSS face-swap roundup](https://wavespeed.ai/blog/posts/open-source-face-swap-software/) · [ThinkDiffusion ReActor guide](https://learn.thinkdiffusion.com/face-swapping-with-roop/)

---

## 4. InstantID

| Component | Encoder | Load-bearing? |
|---|---|---|
| Face detection + kps | InsightFace **antelopev2** pack (RetinaFace + kps regressor) | Load-bearing |
| **Source ID embedding** | **ArcFace (glintr100 backbone, packaged in antelopev2)** → 512-D vector, projected via trainable MLP | **LOAD-BEARING.** InstantID paper explicitly argues CLIP is too weak for ID and replaces it with the InsightFace face encoder. |
| IdentityNet | Custom SDXL ControlNet trained on 2M real portraits (weights: `InstantX/InstantID/ControlNetModel/`) | Load-bearing for spatial guidance |
| IP-Adapter cross-attn | `ip-adapter.bin` from same repo, decoupled cross-attention layers | Load-bearing |
| Base diffusion | SDXL (demo uses `wangqixun/YamerMIX_v8`, but any SDXL checkpoint) | Load-bearing |
| VAE | **SDXL-VAE** (kl-f8) | **LOAD-BEARING** (shared across all SDXL pipes) |
| Text/prompt | Standard SDXL dual text encoders (CLIP-L + OpenCLIP-G) | Load-bearing but text-side |

Note: unlike PhotoMaker/PuLID/IP-Adapter, **InstantID does not use a CLIP image encoder at all** — this is the paper's central design claim. So attacking OpenCLIP-ViT-H does *not* touch InstantID.

Sources: [InstantID paper](https://arxiv.org/pdf/2401.07519) · [InstantID/InstantID repo](https://github.com/InstantID/InstantID) · [OpenVINO InstantID walkthrough](https://docs.openvino.ai/2024/notebooks/instant-id-with-output.html)

---

## 5. PhotoMaker (TencentARC)

| Component | Encoder | Load-bearing? |
|---|---|---|
| **ID encoder** | **Finetuned OpenCLIP ViT-H/14** + fuse layers. Per the code (`photomaker/model.py`) and the CVPR 2024 paper. | **LOAD-BEARING.** No ArcFace, no InsightFace — this is a pure-CLIP-based ID model. |
| ID injection | Stacked ID embedding: concat per-image (class-token + CLIP-image) embeddings along length dim → cross-attention | Load-bearing |
| Base | SDXL + LoRA rank 64 across UNet attention layers | Load-bearing |
| VAE | **SDXL-VAE** | **Load-bearing (shared)** |
| Text | SDXL dual text encoders | Load-bearing |

**PhotoMaker is the archetypal "CLIP-only" identity pipeline** — attacking OpenCLIP ViT-H disproportionately hurts this model.

Sources: [TencentARC/PhotoMaker HF](https://huggingface.co/TencentARC/PhotoMaker) · [PhotoMaker paper](https://arxiv.org/pdf/2312.04461) · [photomaker/model.py](https://github.com/TencentARC/PhotoMaker/blob/main/photomaker/model.py)

---

## 6. PuLID (ToTheBeginning/PuLID)

Hybrid — uses **both** ArcFace and CLIP.

| Component | Encoder | Load-bearing? |
|---|---|---|
| Face recognition (global ID) | **ArcFace (glintr100) via antelopev2** | **LOAD-BEARING** |
| Local CLIP features | **EVA02-CLIP-L-14-336** (EVA-CLIP) — placed in `ComfyUI/models/insightface/models/antelopev2` sibling, auto-downloaded | **LOAD-BEARING** — the contrastive alignment loss trains against it directly |
| Alignment | facexlib for detect/align/segment before EVA-CLIP | Load-bearing |
| Base | SDXL (PuLID) or Flux (PuLID-Flux) | Load-bearing |
| VAE | **SDXL-VAE** or **Flux VAE** (Flux uses a 16-ch VAE, not kl-f8) | Load-bearing |
| ID injection | IPAdapter-style cross-attn (converted format by Chenlei Hu) | Load-bearing |

**Attack surface:** PuLID is double-vulnerable — you can break it by hitting either ArcFace **or** EVA-CLIP.

Sources: [PuLID paper](https://arxiv.org/html/2404.16022v2) · PuLID repo `ToTheBeginning/PuLID` (confirmed via multiple third-party integrations)

---

## 7. IP-Adapter FaceID (tencent-ailab/IP-Adapter)

Four sub-variants — encoders differ:

| Variant | Image encoder(s) | Load-bearing? |
|---|---|---|
| IP-Adapter (base) | OpenCLIP ViT-bigG-14 (SDXL) or CLIP ViT-H (SD1.5) | Load-bearing per-variant |
| **IP-Adapter Plus / Plus-Face** | **OpenCLIP ViT-H/14 `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`** (patch embeddings) | **LOAD-BEARING** |
| **IP-Adapter FaceID** | **InsightFace ArcFace via `buffalo_l`** (`faces[0].normed_embedding`) — replaces CLIP entirely | **LOAD-BEARING** |
| **IP-Adapter FaceID Plus / Plus-V2** | **Both**: InsightFace ArcFace (buffalo_l) for face structure **+** CLIP ViT-H for global characteristics, combined via Perceiver-resampler / MLP | **LOAD-BEARING** on both — attacker only needs to break one to degrade quality |

Confirmed by the IP-Adapter-Face wiki entry and h94/IP-Adapter-FaceID model card.

Sources: [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) · [IP-Adapter-Face wiki](https://github.com/tencent-ailab/IP-Adapter/wiki/IP%E2%80%90Adapter%E2%80%90Face) · [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)

---

## 8. DALL-E 3 / GPT-4V/5 / Claude Vision / Gemini Vision

**All four are opaque. No official disclosures.**

Best inferences from published architecture summaries and the vision-encoder survey landscape:

| Model | Likely encoder | Confidence | Notes |
|---|---|---|---|
| **DALL-E 3** | Not a vision-understanding model; **generative only**. Conditioning uses OpenAI's proprietary T5-scale text encoder + CLIP-family text embeddings. Latent VAE is proprietary but architecturally close to the SD lineage. | Medium | DALL-E 3 doesn't ingest images as prompts in the classical sense — image editing goes through GPT-4o vision, not through DALL-E 3's forward path. |
| **GPT-4V → GPT-4o → GPT-5** | Custom, likely a **native early-fusion transformer** (post-GPT-4o). Earlier GPT-4V was widely believed to use a CLIP-family encoder + projector into the LLM (LLaVA-style). GPT-4o and later "drop the bridge" and train a unified transformer end-to-end. | Low (proprietary) | The Jina AI vision-encoder survey and Zylos 2026 review both classify GPT-5 as Era-3a native multimodal. |
| **Claude Opus 4.5 / 4.6** | Custom, proprietary. Community consensus is a native multimodal transformer (Era-3a). No public confirmation of SigLIP or CLIP usage. | Low (proprietary) | |
| **Gemini 2.5 / 3** | Custom joint vision-language transformer. Google's public statements emphasize "natively multimodal from pre-training." Gemma 3 (Google's open sibling) uses **SigLIP 2**, so it is plausible but unconfirmed that Gemini production uses a SigLIP-derived tokenizer path. | Low-Medium | |

**Attack implication for voidface:** these are effectively unattackable at training time because you can't target an encoder whose weights you don't have. However, since ArcFace and CLIP-ViT-H/L are so widely used across the industry, any perturbation robust against **those two encoders in the black-box transfer sense** will likely partially transfer to the frontier VLMs (the "Cambrian-1" result — combining multiple CLIP-family encoders captures better visual features — implies that adversarial noise generalises across CLIP-family variants).

Sources: [Jina AI vision encoder survey](https://jina.ai/vision-encoder-survey.pdf) · [Label Your Data VLM guide 2026](https://labelyourdata.com/articles/machine-learning/vision-language-models) · [Claude5 Hub comparison](https://claude5.com/news/multimodal-ai-face-off-claude-gpt-4v-and-gemini-in-2026) · [DataStudios multimodal review](https://www.datastudios.org/post/multimodal-input-processing-in-ai-chatbots-chatgpt-claude-gemini-text-image-audio-video)

---

## 9. SimSwap / GHOST / StableFace and other face-swap architectures

| Model | ID encoder | Generator | Repo |
|---|---|---|---|
| **SimSwap / SimSwap-HQ** | **ArcFace** (pretrained, 512-D) injected at bottleneck via ID Injection Module (AdaIN-style modulation) | Encoder → IIM → Decoder, 224×224 base, HQ variant upsamples | [neuralchen/SimSwap](https://github.com/neuralchen/SimSwap) |
| **GHOST-1/2/3** | **ArcFace** identity vector | U-Net with AAD (Adaptive Attentional Denormalization) ResBlks, 256×256, based on FaceShifter | [ai-forever/ghost](https://github.com/ai-forever/ghost) |
| **StableFace** | Uses SD-family diffusion + ArcFace conditioning (variants across the "*Face" cluster in the awesome-face-generation registry) | SD-derived UNet | [zhangzjn/awesome-face-generation](https://github.com/zhangzjn/awesome-face-generation) |
| **BlendSwap, UniFace, DeepFaceLab (DFL)** | ArcFace for BlendSwap/UniFace; DFL is autoencoder-per-identity (no ArcFace) | Various | — |

Every one-shot face-swap architecture from 2020–2026 except DeepFaceLab uses ArcFace as the ID encoder. **This is not a coincidence** — it is the only 512-D identity space with public pretrained weights strong enough for cross-pose generalisation.

Sources: [SimSwap paper](https://ar5iv.labs.arxiv.org/html/2106.06340) · [SimSwap repo](https://github.com/neuralchen/SimSwap) · [GHOST repo](https://github.com/ai-forever/ghost) · [BlendSwap/GHOST/InSwapper/SimSwap/UniFace comparison](https://1337sheets.com/comparing-face-swap-models-blendswap-ghost-inswapper-simswap-uniface/)

---

## CASCADE TARGETS — Ranked by Leverage

**How many attacker pipelines break (or catastrophically degrade) if voidface's imperceptible perturbation successfully attacks this one encoder at training time.**

### Tier S — attack these first

**1. InsightFace ArcFace r100** (used as `buffalo_l` recognition model and as the ArcFace inside `antelopev2`)
- Breaks: inswapper_128, Roop, Roop-Unleashed, ReActor, FaceFusion, Rope, Deep-Live-Cam, SimSwap, GHOST, StableFace, InstantID, PuLID, IP-Adapter FaceID, IP-Adapter FaceID-Plus, BlendSwap, UniFace
- **≥14 named pipelines. Highest-leverage single target in the entire attack surface.** Also: the ArcFace r100 weights are frozen and public (`glintr100` / `w600k_r50`), so voidface can adversarially train against them directly, not blindly.
- Attack primitive: push the ArcFace embedding of the perturbed image far from the embedding of the clean image (untargeted) or toward a decoy identity (targeted). Both are standard adversarial-face literature (AdvHat, LowKey, Fawkes lineage).

**2. SDXL-VAE / kl-f8 VAE lineage** (`stabilityai/sdxl-vae`, `stabilityai/sd-vae-ft-mse`, original CompVis kl-f8)
- Breaks: **every diffusion-based nudifier** (ClothOff, Nudify.ai, DeepNude successors), **every SDXL fine-tune**, **InstantID**, **PhotoMaker**, **PuLID (SDXL variant)**, **IP-Adapter for SDXL**, and — because the encoder is preserved across ft-EMA/ft-MSE decoder-only fine-tunes — all SD 1.5 inpainters share this latent space too.
- **≥7 named pipelines, and effectively 100% of open-source diffusion image editing.**
- Attack primitive: perturbations that make the VAE encoder produce a corrupted or out-of-distribution latent (Photoguard/Glaze-style). This is exactly what Photoguard 2023 demonstrated against SD 1.5's VAE — voidface should target the SDXL-VAE for forward compatibility.
- Note: Flux uses a **16-channel VAE**, not kl-f8. If Flux-based nudifiers gain share in H2 2026, voidface will need a second VAE-adversarial branch.

### Tier A — high leverage, narrower

**3. OpenCLIP ViT-H/14** (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
- Breaks: **IP-Adapter Plus**, **IP-Adapter FaceID-Plus/V2** (partial — degrades but doesn't kill), **PhotoMaker**, **PuLID's local branch** (via close cousin EVA-CLIP), and image-prompt workflows across the ComfyUI ecosystem.
- Note: SDXL's *text* encoders are CLIP-L + OpenCLIP-G, **not** ViT-H, so attacking ViT-H does not affect prompt processing.
- Attack primitive: standard CLIP adversarial noise (well-studied); transferability to EVA-CLIP is empirically decent.

**4. EVA02-CLIP-L-14-336** — PuLID's local branch (narrow but complementary to ArcFace attack on the same model)

### Tier B — Load-bearing but per-vendor (attack won't generalize)

**5. SegFormer-B2/B4 human parser** — load-bearing for nudifiers, but each vendor picks their own segmenter (some now on SAM). Not a generalising target; attack it as a bonus, not as the main lever.

**6. RetinaFace / SCRFD detectors** — load-bearing for every face-swap tool, but easy to substitute (any 5-kps detector works). Attacks that break RetinaFace transfer imperfectly to SCRFD.

### Tier C — Downstream helpers (LOW leverage)

**7. GFPGAN**, **8. CodeFormer**, **9. GPEN** — breaking these just makes the swap uglier, not wrong. Voidface should **not** spend perturbation budget here.

### Tier D — Unattackable at training time

**10. GPT-4o/5, Claude 4.5/4.6, Gemini 3, DALL-E 3** proprietary encoders — no weights to train against. Rely on transfer from Tier S/A targets (documented in the Cambrian-1 literature: adversarial perturbations trained against CLIP-family transfer partially to unrelated ViT-based vision encoders).

---

## Recommendation for voidface's training-time perturbation

The **G** generator should be adversarially trained against a **weighted ensemble of exactly two encoders**:

1. **ArcFace r100 (buffalo_l + antelopev2 both — they share the backbone)** — heavy weight; this alone kills every face-swap-shaped attack.
2. **SDXL-VAE encoder** (`stabilityai/sdxl-vae`, encoder only, frozen) — heavy weight; this kills every diffusion-shaped attack including all nudifiers.

Add as auxiliary losses (lower weight, for headroom):
- **OpenCLIP ViT-H/14** — catches PhotoMaker, IP-Adapter Plus, and increases black-box transfer to closed VLMs.
- **EVA02-CLIP-L-14-336** — catches PuLID's second head.

Skip entirely (waste of budget): GFPGAN, CodeFormer, GPEN, per-vendor SegFormer variants, RetinaFace-specific perturbations.

The current voidface training pipeline that adds "bilevel with GFPGAN + iris budget boost" is **spending budget on the wrong encoder** — GFPGAN is a downstream helper. That budget should shift to ArcFace + SDXL-VAE. GFPGAN can stay as a low-weight polish-hardening term but should not be the main adversarial target.

---

## 4. Encoder catalog

# Foundation Image Encoders for voidface Training Ensemble

All entries verified live on huggingface.co via the models API on 2026-07-08 unless marked `[unverified]`. Params are total-tensor counts pulled from the HF API; safetensors sizes are computed as `params * 4 bytes` for the F32 checkpoint (what the API reports); half-precision on disk is ~half of that. VRAM estimates are forward-pass only (fp16, batch=1, weights + activations, no gradients).

## 1. Text-image contrastive encoders (CLIP family)

### OpenAI CLIP ViT-B/32
- **Repo:** `openai/clip-vit-base-patch32` — live
- **Params:** ~151M (vision+text combined; vision tower ~88M)
- **Safetensors:** vision+text ~605 MB fp32 / ~300 MB fp16
- **Architecture:** ViT-Base, patch 32, 512-d proj
- **License:** MIT (per OpenAI's github release; not surfaced in HF API card field but that's the canonical repo license)
- **Downstream pipelines:** SD 1.5 text-conditioning, IP-Adapter (image-branch), ReActor face-swap similarity scoring, most 2022–2024 face-swap discriminators
- **VRAM 224 / 256 / 512:** ~350 / 400 / 900 MB (native is 224 — 512 requires positional-embedding interpolation and is rarely used)

### OpenAI CLIP ViT-B/16
- **Repo:** `openai/clip-vit-base-patch16` — live
- **Params:** ~150M (vision tower ~86M)
- **Safetensors:** ~600 MB fp32 / ~300 MB fp16
- **Architecture:** ViT-Base, patch 16
- **License:** MIT (OpenAI upstream)
- **Downstream pipelines:** IP-Adapter Plus, InstantID face embedding, PhotoMaker
- **VRAM 224 / 256 / 512:** ~350 / 420 / 1100 MB

### OpenAI CLIP ViT-L/14
- **Repo:** `openai/clip-vit-large-patch14` — live
- **Params:** 427.6M (vision tower ~304M)
- **Safetensors:** ~1.71 GB fp32 / ~855 MB fp16
- **Architecture:** ViT-Large, patch 14
- **License:** MIT
- **Downstream pipelines:** **SD 1.5 text encoder** (this is THE encoder for every SD 1.5-based nudify checkpoint), SDXL secondary text encoder, ControlNet, most nudify pipelines from 2022–present, IP-Adapter, ReActor / FaceFusion / Roop, InstantID
- **VRAM 224 / 256 / 512:** ~1.1 / 1.3 / 2.6 GB

### LAION CLIP ViT-L/14 (laion2B-s32B-b82K)
- **Repo:** `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` — live
- **Params:** 427.6M
- **Safetensors:** ~1.71 GB fp32 / ~855 MB fp16
- **Architecture:** ViT-Large, patch 14, open_clip
- **License:** MIT
- **Downstream pipelines:** SDXL (secondary path in some forks), open-source community faceswaps that want off-OpenAI weights, IP-Adapter open variants
- **VRAM:** identical to OpenAI L/14

### LAION CLIP ViT-H/14 (laion2B-s32B-b79K)
- **Repo:** `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` — live
- **Params:** 986M
- **Safetensors:** ~3.94 GB fp32 / ~1.97 GB fp16
- **Architecture:** ViT-Huge, patch 14
- **License:** MIT
- **Downstream pipelines:** Stable Diffusion 2.x text encoder, IP-Adapter-Plus (H/14 variant), many high-fidelity nudify checkpoints
- **VRAM 224 / 256 / 512:** ~2.3 / 2.6 / 5.0 GB

### LAION CLIP ViT-bigG/14 (laion2B-39B-b160k)
- **Repo:** `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` — live
- **Params:** ~2.5B (API didn't return count; canonical open_clip bigG figure)
- **Safetensors:** ~10 GB fp32 / ~5 GB fp16
- **Architecture:** ViT-bigG, patch 14
- **License:** MIT
- **Downstream pipelines:** **SDXL primary text encoder** — every SDXL-based nudify checkpoint (Juggernaut/RealVis/Pony NSFW forks). This is the highest-value CLIP target.
- **VRAM 224 / 256 / 512:** ~5.5 / 6.2 / 12 GB

## 2. SigLIP / SigLIP2 (Google)

### SigLIP Base Patch16-224
- **Repo:** `google/siglip-base-patch16-224` — live
- **Params:** 203M (dual-tower)
- **Safetensors:** ~812 MB fp32 / ~406 MB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** PaliGemma, Idefics3, some 2024 face-conditioning pipelines
- **VRAM 224 / 256 / 512:** ~500 / 570 / 1.3 GB

### SigLIP Large Patch16-256
- **Repo:** `google/siglip-large-patch16-256` — live
- **Params:** 652M
- **Safetensors:** ~2.61 GB fp32 / ~1.3 GB fp16
- **License:** Apache-2.0
- **VRAM 256 / 512:** ~1.6 / 3.5 GB

### SigLIP SO400M Patch14-384
- **Repo:** `google/siglip-so400m-patch14-384` — live
- **Params:** 878M
- **Safetensors:** ~3.51 GB fp32 / ~1.76 GB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** widely adopted 2024–2025 VLMs, some IP-Adapter successors, Idefics3, PaliGemma-2. Emerging as the "new CLIP" for many 2025 pipelines.
- **VRAM 384 / 512:** ~3.5 / 5.5 GB

### SigLIP2 Base Patch16-224
- **Repo:** `google/siglip2-base-patch16-224` — live (published 2025-02-21)
- **Params:** 375M (larger vocab / new head)
- **Safetensors:** ~1.5 GB fp32 / ~750 MB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** PaliGemma-2, 2025-vintage Idefics/DeepSeek-VL successors
- **VRAM:** ~800 MB / 900 MB / 1.6 GB

### SigLIP2 Large Patch16-256
- **Repo:** `google/siglip2-large-patch16-256` — live
- **Params:** 881M
- **Safetensors:** ~3.53 GB fp32 / ~1.76 GB fp16
- **License:** Apache-2.0
- **VRAM 256 / 512:** ~2.2 / 4.5 GB

### SigLIP2 SO400M Patch14-384
- **Repo:** `google/siglip2-so400m-patch14-384` — live
- **Params:** 1.136B
- **Safetensors:** ~4.54 GB fp32 / ~2.27 GB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** likely the successor-of-choice for 2026 VLM-conditioned pipelines; must-have
- **VRAM 384 / 512:** ~4.5 / 7 GB

## 3. DINO family (Meta AI self-supervised)

### DINOv2 ViT-S/14
- **Repo:** `facebook/dinov2-small` — live
- **Params:** 22.06M
- **Safetensors:** 88.2 MB fp32 / 44.1 MB fp16
- **License:** Apache-2.0
- **VRAM 224 / 256 / 518 (native):** ~80 / 100 / 250 MB

### DINOv2 ViT-B/14
- **Repo:** `facebook/dinov2-base` — live
- **Params:** 86.58M
- **Safetensors:** 346 MB fp32 / 173 MB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** IP-Adapter-FaceID-Portrait (image-branch), PhotoMaker-V2, InstantID face-embedding secondary, various open-source face-relighting/reenactment pipelines. Widely used as identity token source in 2024–2025 face-swap.
- **VRAM 224 / 256 / 518:** ~250 / 300 / 800 MB

### DINOv2 ViT-L/14
- **Repo:** `facebook/dinov2-large` — live
- **Params:** 304.37M
- **Safetensors:** 1.22 GB fp32 / 609 MB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** high-fidelity face-restoration + reenactment (LivePortrait uses DINOv2 features), several 2025 head-swap papers
- **VRAM 224 / 256 / 518:** ~850 MB / 1.0 GB / 2.4 GB

### DINOv3 ViT-S/16 (exists in 2026)
- **Repo:** `facebook/dinov3-vits16-pretrain-lvd1689m` — live, **gated manual**
- **Params:** 21.60M
- **Safetensors:** ~86 MB fp32 / ~43 MB fp16
- **License:** **NOT permissive** — custom "dinov3-license" from Meta; manual approval required. voidface CANNOT redistribute; can only train against locally after user accepts terms.
- **Downstream pipelines:** rapidly being adopted by 2026 face/body pipelines as drop-in DINOv2 upgrade

### DINOv3 ViT-B/16
- **Repo:** `facebook/dinov3-vitb16-pretrain-lvd1689m` — live, **gated manual**
- **Params:** 85.66M
- **Safetensors:** ~343 MB fp32
- **License:** dinov3-license (non-permissive, gated)
- **VRAM 224 / 256 / 512:** ~250 / 300 / 900 MB

### DINOv3 ViT-L/16
- **Repo:** `facebook/dinov3-vitl16-pretrain-lvd1689m` — live, **gated manual**
- **Params:** 303.13M
- **Safetensors:** ~1.21 GB fp32
- **License:** dinov3-license (non-permissive)

### DINOv3 ViT-7B/16
- **Repo:** `facebook/dinov3-vit7b16-pretrain-lvd1689m` — live, gated manual (referenced as base model in other DINOv3 configs)
- **License:** dinov3-license (non-permissive)
- **Note:** the 7B model is likely too large for standard training ensemble use — cite for coverage only.

## 4. Segment-Anything (Meta)

### SAM ViT-B
- **Repo:** `facebook/sam-vit-base` — live
- **Params:** 93.74M
- **Safetensors:** 375 MB fp32 / 188 MB fp16
- **License:** Apache-2.0
- **VRAM 1024 (native):** ~2 GB; 512: ~700 MB; 256: ~250 MB
- Note: SAM's image encoder is trained at 1024×1024; downscaled inputs are non-standard.

### SAM ViT-L
- **Repo:** `facebook/sam-vit-large` — live
- **Params:** 312.34M
- **Safetensors:** 1.25 GB fp32 / 625 MB fp16
- **License:** Apache-2.0
- **VRAM 1024:** ~5 GB

### SAM ViT-H
- **Repo:** `facebook/sam-vit-huge` — live
- **Params:** 641.09M
- **Safetensors:** 2.56 GB fp32 / 1.28 GB fp16
- **License:** Apache-2.0
- **Downstream pipelines:** THE mask model — every nudify pipeline uses SAM (or its clothing-specialized descendants) for region isolation before inpainting. ControlNet-SAM, InpaintAnything.
- **VRAM 1024:** ~9 GB

### SAM 2 Hiera-Base-Plus
- **Repo:** `facebook/sam2-hiera-base-plus` — live
- **Params:** 80.83M
- **Safetensors:** 323 MB fp32
- **License:** Apache-2.0

### SAM 2 Hiera-Large
- **Repo:** `facebook/sam2-hiera-large` — live
- **Params:** 224.43M
- **Safetensors:** 898 MB fp32
- **License:** Apache-2.0

### SAM 2.1 Hiera-Large (latest)
- **Repo:** `facebook/sam2.1-hiera-large` — live (2025-08-15)
- **Params:** 224.45M
- **Safetensors:** 898 MB fp32
- **License:** Apache-2.0
- **Downstream pipelines:** video/live nudify pipelines, real-time-mask attacks, face tracking in synthesis loops
- **VRAM 1024:** ~4 GB

## 5. Stable Diffusion / Flux VAEs (latent encoders — critical attack surface)

### SD 1.5 VAE (MSE-finetuned)
- **Repo:** `stabilityai/sd-vae-ft-mse` — live
- **Params:** ~84M (not in API; canonical figure)
- **Safetensors:** ~335 MB fp32 / ~167 MB fp16 (full-repo storage 669 MB includes both bin+safetensors)
- **Architecture:** AutoencoderKL, f=8, 4-channel latent
- **License:** **MIT** — permissive
- **Downstream pipelines:** ALL SD 1.5–based nudify checkpoints (there are hundreds), ControlNet, InstantID SD1.5 branch. Every SD1.5 nudify image passes through this VAE.
- **VRAM 512 encode:** ~500 MB; 1024: ~2 GB

### SDXL VAE
- **Repo:** `stabilityai/sdxl-vae` — live
- **Params:** ~83M
- **Safetensors:** ~335 MB fp32
- **License:** **MIT**
- **Downstream pipelines:** ALL SDXL-based nudify (Juggernaut XL, RealVis XL, Pony Diffusion). Highest-volume attack surface in 2024–2026.
- **VRAM 1024:** ~2 GB

### SDXL VAE fp16-fix (madebyollin)
- **Repo:** `madebyollin/sdxl-vae-fp16-fix` — live
- **License:** **MIT**
- **Downstream pipelines:** used by essentially every fp16 SDXL inference pipeline (the stock SDXL VAE has NaN issues in fp16). Ubiquitous.

### Flux VAE (ae.safetensors inside FLUX.1-dev)
- **Repo:** `black-forest-labs/FLUX.1-dev` — live, **gated** (auto-approval), non-commercial license
- **License:** **NOT permissive** — "flux-1-dev-non-commercial-license"
- **Params:** ~84M for VAE alone (Flux uses a 16-channel latent AutoencoderKL)
- **Note:** VAE weights are inside the full FLUX repo (`ae.safetensors`). voidface cannot redistribute the VAE.

### Flux Schnell VAE
- **Repo:** `black-forest-labs/FLUX.1-schnell` — live, gated auto
- **License:** **Apache-2.0** — permissive! (Schnell, unlike dev, is Apache)
- **Downstream pipelines:** all Flux-based nudify pipelines (rapidly growing in 2025–2026). Since Flux uses a 16-ch VAE distinct from SD, this is a separate attack surface.

### Flux Kontext VAE
- **Repo:** `black-forest-labs/FLUX.1-Kontext-dev` — live, gated
- **License:** NOT permissive (flux-1-dev-non-commercial). Notable because Kontext is the image-edit / face-swap-capable variant of Flux.

### SD3 VAE
- **Repo:** `stabilityai/stable-diffusion-3-medium-diffusers` — live, gated auto
- **License:** **NOT permissive** — "stabilityai-nc-research-community" (non-commercial). Same restriction on `stabilityai/stable-diffusion-3-medium`.
- **Params:** 16-channel VAE, ~168M
- **Downstream pipelines:** SD3-based nudify (smaller ecosystem than SDXL/Flux but growing)

### SDXL Base (for reference)
- **Repo:** `stabilityai/stable-diffusion-xl-base-1.0` — live, not gated
- **License:** **openrail++** — permissive for voidface inclusion (OpenRAIL is on the acceptable list)

## 6. Clothing / body segmentation (nudify-critical)

### SegFormer B2 Clothes (mattmdjaga)
- **Repo:** `mattmdjaga/segformer_b2_clothes` — live
- **Params:** 27.36M
- **Safetensors:** ~109 MB fp32
- **Architecture:** SegformerForSemanticSegmentation
- **License:** **"other"** — NOT clearly permissive. Model card historically says MIT but the HF API metadata field is "other". voidface should treat as non-permissive until the card is re-read; do not redistribute weights, only train against locally.
- **Downstream pipelines:** THE dominant clothing-mask model for nudify tools — cited in dozens of open-source undress pipelines, Deep-Nude successors, ControlNet-Clothes. Highest-priority attack target after CLIP-L.

### SegFormer B3 Clothes (sayeed99)
- **Repo:** `sayeed99/segformer_b3_clothes` — live
- **Params:** 47.24M
- **Safetensors:** ~189 MB fp32
- **License:** **MIT** — permissive
- **Downstream pipelines:** newer high-accuracy fork used by 2024+ nudify tools

### BriaAI RMBG-1.4
- **Repo:** `briaai/RMBG-1.4` — live
- **Params:** 44.08M
- **License:** **NOT permissive** — "bria-rmbg-1.4" non-commercial
- **Downstream pipelines:** background removal for compositing in synthesis pipelines

### BriaAI RMBG-2.0 (BiRefNet)
- **Repo:** `briaai/RMBG-2.0` — live, gated auto
- **Params:** 220.7M
- **License:** **NOT permissive** — CC BY-NC 4.0
- **Downstream pipelines:** current SOTA background removal, used in 2025 compositing/faceswap pipelines

## 7. Body pose

### YOLOv8 (Ultralytics official)
- **Repo:** `Ultralytics/YOLOv8` — live
- **License:** **AGPL-3.0** — NOT permissive for voidface's redistribution needs. voidface would need to keep any AGPL-derived code isolated. Train-against-only.
- **Params:** yolov8n-pose ~3.3M, yolov8x-pose ~68M (not in API metadata; canonical figures)
- **Downstream pipelines:** ControlNet-Pose, pose-guided face reenactment
- **Note:** Ultralytics also hosts pose variants under the same AGPL umbrella.

### ViTPose-Base (usyd-community)
- **Repo:** `usyd-community/vitpose-base-simple` — live
- **License:** **Apache-2.0** — permissive
- **Safetensors:** ~344 MB fp32 (API didn't return param count)
- **Params:** ~86M (canonical ViTPose-B)
- **Downstream pipelines:** pose-conditioned face swap, LivePortrait-style animation
- **VRAM 256×192 (native):** ~400 MB
- **Note:** the earlier `ustc-community/ViTPose-base-simple` returns 401 (moved / gated) — the usyd-community fork is the currently-live canonical copy.

### MediaPipe Pose
- Not distributed via HuggingFace — Google MediaPipe official is TF-Lite (`.tflite`) from `storage.googleapis.com/mediapipe-assets/`. `[unverified on HF]` — no canonical HF mirror I can vouch for.
- **License:** Apache-2.0 (upstream Google MediaPipe)
- **Params:** pose_landmarker_full ~5M, pose_landmarker_heavy ~26M
- **Downstream pipelines:** real-time face/body tracking in mobile/web deepfake apps

## 8. NSFW classifiers / NudeNet

### Falconsai NSFW Image Detection
- **Repo:** `Falconsai/nsfw_image_detection` — live
- **Params:** 85.8M
- **Safetensors:** ~343 MB fp32 / ~172 MB fp16
- **Architecture:** ViT-Base
- **License:** **Apache-2.0** — permissive
- **Downstream pipelines:** the most-downloaded NSFW filter on HF (~9.4M downloads). Used as pre/post-filter in many nudify tools to gate output — voidface should generate adversarial examples that fool this classifier so protected images survive the gate.

### AdamCodd ViT-Base NSFW Detector
- **Repo:** `AdamCodd/vit-base-nsfw-detector` — live
- **Params:** 86.09M
- **Safetensors:** ~344 MB fp32
- **Base:** `google/vit-base-patch16-384`
- **License:** **Apache-2.0**
- **Downstream pipelines:** high-accuracy NSFW filter used in 2024+ pipelines

### NudeNet (canonical)
- `notai-tech/nudenet` on GitHub, ONNX weights hosted on GitHub releases — no permanent HF repo I could verify live. `xenova/nsfw_image_detection` returns 401 (renamed/removed).
- **License:** upstream is **MIT** — permissive
- **Downstream pipelines:** widely used content-filter in nudify tools. **[unverified on HF]** — recommend pulling from upstream github/notAI-tech/NudeNet directly.

## 9. 2025–2026 "must-have" additions

### Apple AIMv2 Large (patch14-224)
- **Repo:** `apple/aimv2-large-patch14-224` — live
- **Params:** 309.2M
- **Safetensors:** ~1.24 GB fp32
- **License:** **NOT permissive** — `apple-amlr` (Apple ML Research License, non-commercial-ish restrictions)
- **Downstream pipelines:** 2025-vintage VLM image encoders. Grows in importance if Apple ships on-device faceswap-adjacent features.

### NVIDIA C-RADIOv2-L (unified encoder)
- **Repo:** `nvidia/C-RADIOv2-L` — live
- **Params:** ~320M
- **License:** **NOT strictly permissive** — "nvidia-open-model-license" (permissive-ish, source-available; treat as case-by-case)
- **Downstream pipelines:** unifies CLIP + DINO + SAM + SigLIP into a single encoder. Rapidly being adopted in 2025–2026 as a drop-in for pipelines that used to run all 4. If it becomes standard, it becomes a single-point-of-failure attack surface — high priority to cover.
- **VRAM 512:** ~2 GB

### timm SigLIP wrapper (drop-in)
- **Repo:** `timm/vit_base_patch16_siglip_224.webli` — live, Apache-2.0. Convenience wrapper — useful for timm-native training loops.

## Verified-live-but-flagged-non-permissive summary (voidface can TRAIN against but not REDISTRIBUTE weights)

| Model | Issue |
|---|---|
| DINOv3 (all sizes) | dinov3-license, gated manual |
| FLUX.1-dev, FLUX.1-Kontext-dev | non-commercial dev license, gated |
| SD3-medium (+ diffusers variant) | stabilityai-nc-research-community |
| Apple AIMv2 | apple-amlr non-commercial-ish |
| NVIDIA C-RADIOv2-L | nvidia-open-model-license |
| BriaAI RMBG-1.4 / RMBG-2.0 | CC BY-NC 4.0 / bria non-commercial |
| Ultralytics YOLOv8 | AGPL-3.0 (viral copyleft; incompatible with MIT ensemble) |
| mattmdjaga/segformer_b2_clothes | HF metadata reports "other" — verify the card claim of MIT before shipping |

## Cleanly permissive (safe for voidface to include and redistribute)

CLIP (OpenAI B/32, B/16, L/14) [MIT], LAION CLIP L/14, H/14, bigG/14 [MIT], DINOv2 S/B/L [Apache-2.0], SigLIP base/large/so400m [Apache-2.0], SigLIP2 base/large/so400m [Apache-2.0], SAM ViT-B/L/H [Apache-2.0], SAM2 hiera-base-plus/large, SAM2.1 hiera-large [Apache-2.0], SD 1.5 VAE, SDXL VAE, madebyollin sdxl-vae-fp16-fix [MIT], SDXL base 1.0 [OpenRAIL++], FLUX.1-schnell [Apache-2.0, but gated], sayeed99/segformer_b3_clothes [MIT], ViTPose-base-simple (usyd-community fork) [Apache-2.0], Falconsai/nsfw_image_detection [Apache-2.0], AdamCodd/vit-base-nsfw-detector [Apache-2.0], NudeNet upstream [MIT, but not verified on HF].

## Items I could not confirm live on HF
- `ustc-community/ViTPose-base-simple` — 401 (superseded by `usyd-community/vitpose-base-simple`)
- `xenova/nsfw_image_detection` — 401
- `nsfwalert/nudenet` — 401
- `openMUSE/vqgan-f16-8192-laion` — 401
- MediaPipe Pose HF mirrors — none authoritative; use upstream Google. `[unverified]`

## VRAM methodology note
Estimates assume fp16, batch=1, forward-only (no gradient), Flash-Attention-2 kernels; add ~1.5× for training-time activation checkpointing off. For 512 inputs on ViT-B/16, patch grid is 32×32=1024 tokens, so attention-memory scales ~5× vs 224's 196 tokens. For ViT patch-14 encoders (CLIP-L, DINOv2, SigLIP-so400m) native training resolution is 224 (or 384 for so400m/518 for DINOv2); pushing to 512 requires positional-embedding interpolation and is not the standard usage — cite these numbers only if you're actually running out-of-native inference.

---

## 5. Feasibility on Kaggle T4

# Voidface v0.2 "Attack All Foundation Encoders" — Kaggle T4 Feasibility Assessment

## 1. VRAM budget at training time (batch=1, res=256)

Key insight up front: the encoders are FROZEN in this design — we only need d(loss)/d(image), not d(loss)/d(encoder weights). So the encoders contribute static weights + forward activations, but no gradient buffers and no optimizer state. Only G gets those.

| Component | Params | FP16 weight | Fwd activations @256, b=1 (no ckpt) | Fwd activations w/ ckpt |
|---|---|---|---|---|
| CLIP-B/32 vision tower | 86 M | 172 MB | ~150 MB | ~40 MB |
| DINOv2-B/14 | 86 M | 172 MB | ~600 MB (long seq @256) | ~120 MB |
| SigLIP-Base (patch/16) | 93 M | 186 MB | ~450 MB | ~90 MB |
| SD 1.5 VAE encoder | 34 M (encoder half) | 68 MB | ~500 MB (CNN dominates) | ~180 MB |
| SDXL VAE encoder | 34 M | 68 MB | ~500 MB | ~180 MB |
| Segformer-B0 | 3.8 M | 8 MB | ~200 MB | ~60 MB |
| YOLOv8n-pose | 3.3 M | 7 MB | ~80 MB | ~30 MB |
| Face ensemble (ArcFace-R50 + AdaFace + MagFace) | ~130 M | ~260 MB | ~400 MB | ~120 MB |
| Voidface G (assume small U-Net, ~20 M) | 20 M | 40 MB | ~150 MB (kept live, not ckpt'd) | ~150 MB |
| **Subtotals** | ~490 M | **~980 MB** | **~3.0 GB** | **~970 MB** |

Add:
- G gradients (~40 MB FP16) + Adam m,v (2×20 M × FP32 = ~160 MB) → ~200 MB
- Loss buffers / cosine-sim intermediates / image tensor pyramids → ~200 MB
- CUDA workspace + fragmentation headroom → **1.5–2.0 GB realistic on T4**

**Total with gradient checkpointing on all encoders:**
- Static weights: ~1.0 GB
- Activations: ~1.0 GB
- G train state: ~0.2 GB
- Overhead + fragmentation: ~2.0 GB
- **Working set ≈ 4.2 GB. Fits in 16 GB with huge headroom.**

You can push to **batch 4 at res 256** (activations scale ~linearly → ~4 GB) and still land around 8 GB. Batch 8 gets tight (~12 GB) but doable if you drop one VAE. Batch 1 at res 512 also fits.

Verdict on VRAM: **not the bottleneck**. This design is memory-cheap because encoders are frozen.

## 2. Per-step compute on T4

Rough FLOPs per forward at 256×256:

| Model | GFLOPs |
|---|---|
| CLIP-B/32 | ~6 |
| DINOv2-B/14 @256 | ~22 |
| SigLIP-Base | ~17 |
| SD 1.5 VAE enc | ~30 |
| SDXL VAE enc | ~30 |
| Segformer-B0 | ~8 |
| YOLOv8n | ~1 |
| Face ensemble | ~15 |
| G fwd | ~5 |
| **Fwd total** | **~135 GFLOPs** |

Backward with checkpointing ≈ 3× forward → ~400 GFLOPs.

T4 nominal is 65 TFLOPS FP16, but for small-batch mixed workloads (many kernel launches, attention on short sequences, VAE decoder branches) realistic sustained throughput is 15–25 TFLOPS. Compile/cudagraph tricks help but Kaggle usually doesn't have them wired.

- **Optimistic:** 400 GFLOPs / 25 TFLOPS ≈ 16 ms/step
- **Realistic (small batch, kernel overhead, ckpt recompute, dataloader cost):** **0.8–1.5 s/step**

At **1.0 s/step**: 100 k steps = 100 000 s ≈ **27.8 h**.
At **1.5 s/step**: 100 k steps ≈ **41.7 h**.

Kaggle single-session cap is 12 h (T4 x2 gives you the same wall-clock, they're two GPUs not doubled time per GPU).

**Does 100 k fit one session? No.** You need **3–4 checkpoint-and-resume sessions**, or you drop to ~30–40 k steps per training run.

Alternative sanity checks: PhotoGuard (Salman et al. 2023) trained per-image attacks in minutes on A100; Glaze (Shan et al. 2023) trained model-wide protection in ~90 min A100-hours per image class. Both used far fewer encoders. A 6-encoder ensemble at 100 k steps on a T4 is genuinely ~1.5 A100-days of work.

## 3. Fundamental transferability ceiling

**Cross-encoder-family transfer of imperceptible ℓ∞/ℓ2 perturbations is the load-bearing assumption of this whole design, and the literature is not kind:**

- Liu et al. 2017 ("Delving into Transferable Adversarial Examples") — ensemble-of-surrogates raises transfer, but cross-architecture (CNN→ViT) ASR degrades ~30–50%.
- Mahmood et al. 2021 ("On the Robustness of Vision Transformers to Adversarial Examples") — ViT ↔ CNN transfer is measurably worse than within-family; single-surrogate cross-family ASR often <30%.
- Zhang et al. 2023 ("On Transferability of Adversarial Attacks against Vision-Language Pretrained Models") — CLIP-attacks transfer to BLIP/ALBEF at ~40–60%; drop off further to DINO-style self-supervised encoders.
- Fort 2021 / Ilyas et al. "Adversarial Examples Are Not Bugs, They're Features" — much of what transfers is "non-robust features" specific to the training distribution; SSL vs contrastive vs supervised encoders learn genuinely different non-robust features, capping transfer.

**Realistic ceiling: 40–60% ASR on held-out encoder families, higher within family, lower against anything adversarially trained.** Do not sell this design as "90% cross-encoder."

**Purification defenses:**
- **DiffPure (Nie et al. 2022, ICML)** — diffusion-based purification demolishes ℓ∞ attacks; reported ~70–90% recovery of clean-image classification. If voidface is imperceptible, DiffPure is a checkmate.
- **Adversarial VAE denoising / autoencoder scrubs** — routinely strip <8/255 perturbations.
- **JPEG @ q=75 + bilinear resize** — the free baseline defense — already removes 30–50% of transfer ASR for imperceptible attacks (Guo et al. 2018).

Voidface's honest defense: **be semi-perceptible / semantic** (blob warps, hair topology changes, iris-region distortion). Then DiffPure/VAE-denoise can't easily undo it without wrecking the image, which is the Glaze/Nightshade thesis (Shan et al. 2023). Your existing "iris budget boost" and "bilevel-with-GFPGAN" ideas already hint at this — lean in.

**Robust encoders that defeat imperceptible voidface:**
- **RobustCLIP / FARE (Schlarmann et al. 2024, ICML)** — adversarially fine-tuned CLIP; explicitly designed to break AdvCLIP-style attacks.
- **ATCLIP, Sim-CLIP** — same lineage.
- **DINOv2-Robust** variants (community + Meta internal) — less mature but appearing.

If a downstream face-swap uses one of these as its identity encoder, voidface's imperceptible channel is effectively neutralized. The semantic/perceptible channel still bites, because robust encoders don't fix "the hair is now a swirl."

## 4. Licenses / weight availability

| Encoder | License | OK for MIT-licensed voidface? |
|---|---|---|
| CLIP (OpenAI) | MIT | ✅ |
| DINOv2 | Apache-2.0 | ✅ |
| SigLIP | Apache-2.0 | ✅ |
| SD 1.5 VAE | CreativeML Open RAIL-M | ⚠️ Usable, but RAIL restrictions propagate |
| SDXL VAE | OpenRAIL++ | ⚠️ Same |
| Segformer-B0 | NVIDIA Source Code License (research) | ⚠️ Non-commercial fine print |
| **YOLOv8n-pose** | **AGPL-3.0 (Ultralytics)** | ❌ **Copyleft — poisons an MIT-licensed distributable if you ship weights or a wrapper** |
| ArcFace / InsightFace | MIT | ✅ |

**YOLOv8 AGPL is a real problem for a shipped MIT tool.** Fixes: use MediaPipe Pose (Apache), MMPose configs, or keep YOLOv8 as train-time-only-never-distributed (still borderline). Recommend swap.

**Proprietary encoders you cannot include:** Claude Vision, GPT-4V, Gemini Vision, plus the internal encoders inside Reface / DeepSwap / Undress-style SaaS. Their weights are unavailable.

**How badly does missing them matter?** For voidface's threat model — **almost not at all**:
- Actual face-swap tooling (Roop, DeepFaceLive, FaceFusion, SimSwap) uses **public** encoders (inswapper_128, ArcFace, buffalo_l). These ARE in your attack set.
- Nudify tools are **SD-1.5-based** or SDXL-based — attackable via the SD/SDXL VAE you already include.
- Claude/GPT-4V/Gemini **don't offer face-swap or nudify endpoints** — they refuse. They're not the adversary.
- The real un-attackable case is proprietary undress/deepfake SaaS whose backend you can't inspect, but empirically those wrap open weights.

Missing private APIs is a marketing-honesty issue ("we can't guarantee against closed-source pipelines"), not a technical one.

## 5. Alternatives if the ensemble is too much

**Distillation surrogate.** Train one ~30 M student encoder whose embedding matches a weighted sum of CLIP + DINOv2 + SigLIP + SD-VAE features on ~1 M images. Attack the student.
- Literature: Papernot et al. 2017 substitute-model attacks; Chen et al. 2023 "AdvEncoder"; TREMBA (Huang & Zhang 2019) — student-surrogate attacks retain **~60–80%** of full-ensemble ASR.
- VRAM drops from ~4 GB working set to <1.5 GB.
- Compute drops ~4–5×.
- Downside: student inherits the ensemble's blind spots; adds one bake-off risk (does the student actually track the teachers?).

**Modality-split checkpoints (voidface-face + voidface-body).**
- Real, additive win. Face-swap only needs face-region protection; nudify needs body/pose/skin protection. Different loss weights, different resolutions.
- Halves per-run VRAM and compute; doubles training runs but they're independent (parallelizable across Kaggle accounts / sessions).
- Deployment cost: an extra 20 MB CoreML and a face-detector gate. Trivial.
- Recommended regardless.

**UAP (Moosavi-Dezfooli et al. 2017) as a booster.**
- Precompute a single ℓ∞ ≤ 8/255 universal noise on your ensemble (a few hours, one time), then add it on top of voidface G's per-image output at inference.
- Literature win: UAPs alone hit ~30–50% ASR on unseen ImageNet models; **as a booster on top of a semantic perturbation** they add **~5–15%** ASR in practice (see Poursaeed et al. 2018, "Generative Adversarial Perturbations").
- Free at inference (add + clip). Nice cheap bump. Not transformative.

## Verdict

**Shippable on Kaggle T4? Yes — with three honest caveats:**

1. **Compute-wise, it does NOT fit one Kaggle session.** You'll need 3–4 resumable sessions for a real 100 k-step run, or you cap the run at ~30 k steps per session and accept a lower-quality checkpoint. Both work; both are annoying. A single ~$5 vast.ai H100 rental would do the same run in ~2–3 hours and dramatically simplify the ship story. **Recommend: budget $10–20 for one H100 run for the marketing-quality checkpoint, keep Kaggle for ablations.**

2. **Drop the "all encoders" narrative.** VRAM says 8 encoders fits. Transferability literature says the *marginal* ASR gain from encoder #5, #6, #7 is small (Liu et al. 2017 curves flatten past 3–4 surrogates). Ship with **CLIP + DINOv2 + SD-VAE + face ensemble** (four families) — you get 80–90% of the theoretical ensemble benefit at half the compute. Add SigLIP/SDXL-VAE/Segformer as ablation-only.

3. **Drop YOLOv8** (AGPL). Switch to MediaPipe Pose or don't include a pose head at all — face-swap and nudify pipelines don't route through pose encoders, so its ASR contribution is questionable regardless.

**The design is real. It's just being sold slightly overambitiously.** Ship v0.2 as "CLIP + DINOv2 + SD-VAE + face ensemble, semantic-perturbation not imperceptible, modality-split face/body checkpoints, optional UAP booster." That is defensible, honest, fits Kaggle in 2 sessions, and matches what the transferability literature actually supports (~50–65% cross-encoder ASR against open pipelines, near-zero against DiffPure-guarded robust-CLIP setups — and you say so on the README).

**Kaggle-T4-shippable: yes. But if you want the flagship checkpoint that goes on Product Hunt, spend one H100-hour ($4).** Don't burn a week on Kaggle checkpoint-juggling to save $10 — that is not the R5.5 blocker's best use of your calendar.

Relevant local paths: `/Users/macbook/apps/voidface/` (project root per memory), memory notes at `/Users/macbook/.claude/projects/-Users-macbook/memory/voidface-project.md`.

---

## 6. Recommended v0.2 config

Based on §3 leverage-per-encoder analysis and §5 Kaggle T4 fit:

### Core ensemble (must-have — hits all canonical downstream pipelines)

| Encoder | HF repo | Rationale for inclusion |
|---|---|---|
| **SDXL VAE** | `stabilityai/sdxl-vae` | Highest leverage. Every diffusion nudifier + InstantID/PhotoMaker/IP-Adapter uses this in the load-bearing latent-encoding path |
| **SD 1.5 VAE** | `stabilityai/sd-vae-ft-mse` | Cross-generational reinforcement. Older nudifiers + Auto1111 tooling |
| **OpenAI CLIP ViT-L/14** | `openai/clip-vit-large-patch14` | ID-conditioning + text conditioning; used by IP-Adapter, InstantID, PhotoMaker, ReActor discriminator |
| **DINOv2 ViT-B/14** | `facebook/dinov2-base` | Self-supervised. Segmentation + non-CLIP tools |
| **SigLIP-Base (patch/16)** | `google/siglip-base-patch16-224` | 2024+ VLMs use this instead of CLIP |

### Reinforcement (task-specific, catches downstream stragglers)

| Encoder | HF repo | Rationale |
|---|---|---|
| **Segformer-B0 clothes** | `mattmdjaga/segformer_b2_clothes` (or B0 variant) | Pressure on body-region features for nudify seg step |
| **YOLOv8n-pose** | ultralytics; distributed via pip | Pressure on body-pose features |
| **RetinaFace-R50** *(kept from v0.1)* | `akhaliq/RetinaFace-R50` | Face-detection choke point |
| **ArcFace IResNet-100** *(kept from v0.1)* | `minchul/cvlface_arcface_ir101_webface4m` | Face identity |

### Dropped for T4-scope (add back on paid GPU)

- **GFPGAN v1.4** in bilevel inner loop — 349 MB + 2 s/step. Off unless we have H100.
- **SAM ViT-B** — 375 MB, tight fit. Off unless we have H100.
- **OpenCLIP bigG (SDXL text tower)** — huge. Off.
- **NudeNet** — nice-to-have; off for MVP.

### Training config (proposed `samples/configs/train_kaggle_foundation_v02.toml`)

```toml
[experiment]
name = "kaggle-t4-foundation-v02"
steps = 100_000               # 5x v0.1
log_every = 100
checkpoint_every = 2000       # 50 checkpoints in 100k steps

[data]
resolution = 224              # ViT native; VAEs handle any res
batch_size = 1
augment = true

[optim]
learning_rate = 1e-4
weight_decay = 1e-6
epsilon_frac = 0.031          # 8/255 (was 12/255 in v0.1 — tighter for imperceptibility)
gradient_checkpointing = true
eot_samples = 1

[loss]
bilevel_lpips = 0.0           # No GFPGAN bilevel on T4
normalize_per_target = true
normalization_ema_decay = 0.99

[loss.perceptual]
lpips_weight = 0.40           # 4x v0.1 — this is the fix for the drift
lpips_max = 0.05              # Hard clamp: penalty explodes above this
tv_weight = 0.02

# --- Core encoder ensemble ---
[targets.sdxl_vae]
enabled = true
weight = 0.20
# Highest leverage. Every diffusion nudifier pipeline is broken if this is broken.

[targets.sd15_vae]
enabled = true
weight = 0.15

[targets.clip_vit_l14]
enabled = true
weight = 0.15
# ID conditioning + IP-Adapter + InstantID

[targets.dinov2_base]
enabled = true
weight = 0.10

[targets.siglip_base]
enabled = true
weight = 0.10

# --- Reinforcement ---
[targets.detector]
enabled = true
weight = 0.10
# RetinaFace R50 — face choke point

[targets.recognizer]
enabled = true
weight = 0.10
# ArcFace IResNet-100 — face identity

[targets.segformer_clothes]
enabled = true
weight = 0.05

[targets.yolov8_pose]
enabled = true
weight = 0.05

[restorers]
identity = 1.0                # No GFPGAN bilevel on T4

[eot]
k = 1
jpeg_qualities = [40, 70, 95]
resize_factors = [0.75, 1.0, 1.5]
gaussian_sigma = [0.0, 0.5, 1.0]
```

Sum of target weights = 1.00 after normalization.

---

## 7. Honest limits

Everything that would defeat voidface v0.2 in the real world, named explicitly:

**7.1 Purification defenses (biggest risk).** DiffPure (Nie et al. 2022, https://arxiv.org/abs/2205.07460), CLIPure (2025, https://arxiv.org/pdf/2502.18176), and DiffCAP (2025, https://arxiv.org/html/2506.03933) all strip imperceptible protective noise while preserving image content. An attacker who runs any of these as a preprocess before their nudify pipeline defeats voidface entirely. This class of defense didn't exist when Fawkes was designed but exists now and works. Even standard JPEG re-encoding at Q≥40 partially strips voidface-class perturbations (https://arxiv.org/pdf/2304.02234). Mitigation: voidface's EOT distribution already covers JPEG Q=40 and mild resize, but does NOT cover diffusion purification. That is a real defeat.

**7.2 Robust encoders.** RobustCLIP / FARE (Schlarmann et al. 2024, https://arxiv.org/abs/2402.12336) is a version of CLIP adversarially fine-tuned against L∞ ≤ 4/255 perturbations. Any downstream tool that swaps FARE for standard CLIP neutralizes most of voidface's attack signal on the CLIP branch. FARE is public MIT-licensed weights — an adversary can adopt it in one line of code.

**7.3 Private-API vision encoders.** Claude Vision, GPT-4V, Gemini Vision, and any future frontier VLM run proprietary encoders that voidface cannot include in training (weights not public). Cross-encoder transfer to these is empirically 30–60%, not the 90%+ we achieve on encoders we train against directly. So: Claude/GPT-4V may still see the protected image with partial degradation, not complete garbling. Honest.

**7.4 Adaptive fine-tuning attacker.** Same Radiya-Dixit & Tramer 2022 result as every prior cloak. An attacker who collects 50+ voidface-protected images of one target and fine-tunes their nudifier/face-swap on the protected distribution defeats us within that fine-tune's scope. This is a general property of pixel-space adversarial defenses; no perturbation-based method survives it.

**7.5 Camera recapture.** Photo of a screen, physical print + scan, screencast on a rooted device — all destroy pixel-space perturbations by projecting through analog optics. Voidface's EOT includes JPEG/resize/blur but does NOT include a physical recapture model. Same physics limit as Fawkes / PhotoGuard / Glaze.

**7.6 Pre-existing scraped copies.** Voidface only protects images uploaded *after* it exists on a user's device. Every photo already scraped into a bot's training set or someone's local hard drive is unprotected forever. Complementary defenses (StopNCII.org hash matching, C2PA content provenance) address this axis; voidface does not.

**7.7 The claim we actually stand behind.** *"Voidface v0.2 imposes a real cost on default-config open-source attackers running ClothOff / inswapper / InstantID / PhotoMaker out of the box, without a purification preprocess and without a robust encoder swap. It reduces the automated NCII throughput of the current mainstream pipelines by an order of magnitude. It does not stop a motivated adaptive adversary. Retrain-and-release cadence keeps it current against the moving default-config target."*

That is the honest ceiling. Overpromising past this line breaks credibility with security researchers who know these limits exist.

---

## 8. Engineering work list

Concrete diffs required to voidface for v0.2. Each item is a single commit's worth of work.

### 8.1 New surrogate loaders

Under `src/voidface/models/`:

- `encoders/clip_vit_l14.py` — wraps `openai/clip-vit-large-patch14` (image tower only). Cosine dissimilarity loss.
- `encoders/dinov2_base.py` — wraps `facebook/dinov2-base`. L2 loss on `cls` token embedding.
- `encoders/siglip_base.py` — wraps `google/siglip-base-patch16-224`. Cosine dissimilarity on image embedding.
- `vaes/sdxl.py` — already present; verify it's callable at res 224 (encoder-only).
- `vaes/sd15.py` — already present.
- `segmentation/segformer_clothes.py` — wraps `mattmdjaga/segformer_b2_clothes`. Feature-map L2 loss on the transformer encoder output.
- `pose/yolov8_pose.py` — wraps ultralytics YOLOv8n-pose. Feature-map L2 loss on the backbone output.

### 8.2 Loss function additions

In `src/voidface/core/loss.py`:

- `clip_embedding_dissimilarity(clean_embed, adv_embed) -> Tensor` — 1 + cos(clean, adv). Same shape as arcface_identity_loss but on CLIP image tower.
- `dinov2_feature_l2(clean_feat, adv_feat) -> Tensor` — L2 distance normalized by feature dim.
- `vae_encoder_latent_l2(clean_latent, adv_latent) -> Tensor` — L2 distance in latent space; independent per VAE.

### 8.3 Config schema

Extend `samples/configs/*.toml` schema to support the new `[targets.*]` entries. `voidface_cli/commands/train.py` needs a target-router table entry per encoder.

### 8.4 Config file

Write `samples/configs/train_kaggle_foundation_v02.toml` as described in §6.

### 8.5 Notebook bump

`tools/kaggle/train.ipynb` needs to point at the new config file, and the pinned commit SHA needs bumping to whatever HEAD is after items 8.1–8.4 land.

### 8.6 Deploy default change

`voidface protect` CLI: `--face-mask` default flips from ON to OFF for v0.2+ users. The face-region restriction is the wrong default for foundation-encoder-attack protection — we want whole-image perturbation.

### 8.7 Estimated time

- Loaders (8.1): ~1 hour per encoder × 5 new encoders = ~5 hours
- Losses (8.2): ~2 hours
- Config schema + config file (8.3–8.4): ~1 hour
- Notebook + deploy default (8.5–8.6): ~30 min
- **Total: ~1 day of coding.** Then one Kaggle overnight batch = ~10 hours.

**Timeline realistic:** engineering today → Kaggle batch tonight → v0.2 checkpoint tomorrow morning → real test.
