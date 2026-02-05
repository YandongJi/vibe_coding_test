import argparse
import os
import math
from typing import Optional

import torch
import torchvision
from torchvision.utils import save_image

from models.dit import DiTConfig, DiTGenerator


try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None


def load_vae(vae_id: str, device: torch.device):
    if AutoencoderKL is None:
        raise RuntimeError("diffusers not installed. `pip install diffusers transformers accelerate`")
    vae = AutoencoderKL.from_pretrained(vae_id).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def decode_from_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    img = vae.decode(latents / sf).sample
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--mode", type=str, choices=["pixel","latent"], default="latent")
    p.add_argument("--vae_id", type=str, default="", help="required for latent mode")

    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--n", type=int, default=64, help="number of samples")
    p.add_argument("--alpha", type=float, default=1.0, help="CFG strength α (paper uses 1..3.5 search)")
    p.add_argument("--class_id", type=int, default=-1, help="if >=0, sample this class; else random per sample")

    # model shape (must match training)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--in_channels", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--register_tokens", type=int, default=16)
    p.add_argument("--style_tokens", type=int, default=32)
    p.add_argument("--style_codebook", type=int, default=64)

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})

    # infer config from checkpoint if possible
    if args.mode == "latent":
        image_size = int(train_args.get("latent_h", 32))
        in_channels = int(train_args.get("latent_c", 4))
    else:
        image_size = int(train_args.get("image_size", args.image_size))
        in_channels = 3

    cfg = DiTConfig(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=int(train_args.get("patch_size", args.patch_size)),
        hidden_dim=int(train_args.get("hidden_dim", args.hidden_dim)),
        depth=int(train_args.get("depth", args.depth)),
        n_heads=int(train_args.get("n_heads", args.n_heads)),
        num_classes=int(train_args.get("num_classes", args.num_classes)),
        n_register_tokens=int(train_args.get("register_tokens", args.register_tokens)),
        n_style_tokens=int(train_args.get("style_tokens", args.style_tokens)),
        style_codebook=int(train_args.get("style_codebook", args.style_codebook)),
        cond_dim=int(train_args.get("hidden_dim", args.hidden_dim)),
    )
    model = DiTGenerator(cfg).to(device)

    # load EMA if available
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    model.eval()

    vae = None
    if args.mode == "latent":
        if not args.vae_id:
            # try from train args
            args.vae_id = train_args.get("vae_id", "")
        if not args.vae_id:
            raise RuntimeError("latent mode requires --vae_id (e.g., stabilityai/sd-vae-ft-mse)")
        vae = load_vae(args.vae_id, device)

    # sample
    if args.mode == "latent":
        eps = torch.randn((args.n, in_channels, image_size, image_size), device=device)
    else:
        eps = torch.randn((args.n, 3, image_size, image_size), device=device)

    if args.class_id >= 0:
        c = torch.full((args.n,), int(args.class_id), device=device, dtype=torch.long)
    else:
        c = torch.randint(0, cfg.num_classes, (args.n,), device=device)

    a = torch.full((args.n,), float(args.alpha), device=device)
    style_ids = torch.randint(0, cfg.style_codebook, (args.n, cfg.n_style_tokens), device=device)

    with torch.no_grad():
        out = model(eps, c, a, style_ids=style_ids)

    if args.mode == "latent":
        img = decode_from_latents(vae, out)
    else:
        img = out

    # save a grid
    grid = torchvision.utils.make_grid(img, nrow=int(math.sqrt(args.n)))
    save_image((grid + 1.0) * 0.5, os.path.join(args.out_dir, "grid.png"))
    print("Saved:", os.path.join(args.out_dir, "grid.png"))

    # save individual images
    for i in range(args.n):
        save_image((img[i] + 1.0) * 0.5, os.path.join(args.out_dir, f"{i:05d}.png"))


if __name__ == "__main__":
    main()
