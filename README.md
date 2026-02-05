# Drifting Models (ImageNet) — Reference Training + Inference Code

This is a *reference* PyTorch implementation of **Generative Modeling via Drifting** (Deng et al., 2026).
It follows the paper's pseudocode/appendix for:

- Drifting field `V` with *double-softmax* normalization (Alg. 2)
- Training objective with `stopgrad` target (Eq. 6 / Eq. 13 / Eq. 26)
- Feature and drift normalization + multi-temperature aggregation (Appendix A.6)
- CFG via unconditional real negatives with weighting derived from α (Appendix A.7)
- A queue-based sampler for positives / unconditional data (Appendix A.8)

It supports two modes:
1. **Pixel-space** generator: noise → 256×256 RGB
2. **Latent-space** generator (recommended): noise → 32×32×4 SD-VAE latents, then decode to pixels for feature extraction and/or saving.
   Latent mode requires a Stable-Diffusion-style VAE (e.g., from `diffusers`).

> NOTE: This repo is engineered to be *complete and runnable*, but it is not the authors' exact internal codebase.
> Matching the paper's best numbers requires the same generator architecture, MAE feature encoder, and large-scale training setup.

---

## 1) Data

Expect ImageNet in ImageFolder layout:

```
IMAGENET_ROOT/
  train/
    n01440764/xxx.JPEG
    ...
  val/
    ...
```

---

## 2) Install

```bash
pip install -r requirements.txt
# For latent training:
pip install diffusers transformers accelerate
```

---

## 3) Train (single-node, multi-GPU)

### Pixel-space (simpler, no VAE)
```bash
torchrun --nproc_per_node=8 train_imagenet_drifting.py \
  --data_root /path/to/IMAGENET_ROOT/train \
  --mode pixel \
  --out_dir ./runs/pixel_b16 \
  --image_size 256 \
  --num_classes 1000
```

### Latent-space (paper-style, needs SD VAE)
```bash
torchrun --nproc_per_node=8 train_imagenet_drifting.py \
  --data_root /path/to/IMAGENET_ROOT/train \
  --mode latent \
  --vae_id stabilityai/sd-vae-ft-mse \
  --out_dir ./runs/latent_b2 \
  --latent_h 32 --latent_w 32 --latent_c 4 \
  --num_classes 1000
```

---

## 4) Sample / Inference (1-NFE)

```bash
python sample_imagenet_drifting.py \
  --ckpt /path/to/checkpoint.pt \
  --out_dir ./samples \
  --mode latent \
  --vae_id stabilityai/sd-vae-ft-mse \
  --num_classes 1000 \
  --n 64 \
  --alpha 1.0
```

---

## 5) What to tweak to match the paper closer

- Use a **DiT-like** generator with adaLN-Zero + register tokens and style embedding (paper Appendix A.2).
- Use a strong **self-supervised feature encoder** (MoCo/SimCLR in pixel space, or the paper's latent-MAE).
- Use the paper's batch allocation: `Nc`, `Npos`, `Nneg`, `Nuncond`, and α sampling (Table 8).

See defaults in `train_imagenet_drifting.py` and CLI args.
