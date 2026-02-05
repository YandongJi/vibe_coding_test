import argparse
import os
import math
import time
from copy import deepcopy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from drifting.loss import drifting_loss_multi_group, CFGAlphaSchedule, sample_powerlaw_alpha, cfg_uncond_weight
from models.dit import DiTConfig, DiTGenerator
from models.feature_resnet import PixelFeatureEncoder, FeatureGroupsConfig
from data.imagenet_queue import LatentQueue, QueueConfig


# Optional: latent tokenizer (Stable Diffusion VAE)
try:
    from diffusers import AutoencoderKL
except Exception:
    AutoencoderKL = None


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    return rank, world, local


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=(1 - decay))


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def build_imagenet_loader(data_root: str, batch_size: int, num_workers: int, rank: int, world: int):
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                 # [0,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),  # -> [-1,1]
    ])
    ds = ImageFolder(root=data_root, transform=tfm)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return ds, loader, sampler


def build_generator(args) -> DiTGenerator:
    if args.mode == "latent":
        # latent is treated as an "image" of size latent_h*8? Actually latent is 32x32.
        # We'll set image_size to latent_h (assumed square) for simplicity.
        image_size = args.latent_h
        in_ch = args.latent_c
        patch = args.patch_size   # e.g., 2 like DiT/2 for latent
    else:
        image_size = args.image_size
        in_ch = 3
        patch = args.patch_size   # e.g., 16 like DiT/16 for pixel

    cfg = DiTConfig(
        image_size=image_size,
        in_channels=in_ch,
        patch_size=patch,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        n_heads=args.n_heads,
        num_classes=args.num_classes,
        n_register_tokens=args.register_tokens,
        n_style_tokens=args.style_tokens,
        style_codebook=args.style_codebook,
        cond_dim=args.hidden_dim,
    )
    return DiTGenerator(cfg)


def load_vae(args, device: torch.device):
    if args.mode != "latent":
        return None
    if AutoencoderKL is None:
        raise RuntimeError("diffusers not installed. `pip install diffusers transformers accelerate` for latent mode.")
    if args.vae_id:
        vae = AutoencoderKL.from_pretrained(args.vae_id).to(device)
    else:
        raise RuntimeError("latent mode requires --vae_id (e.g., stabilityai/sd-vae-ft-mse)")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_to_latents(vae, images_m11: torch.Tensor, sample: bool = False) -> torch.Tensor:
    # images_m11 in [-1,1]
    if vae is None:
        raise RuntimeError("VAE is None")
    # diffusers expects [-1,1] already
    enc = vae.encode(images_m11)
    if sample:
        z = enc.latent_dist.sample()
    else:
        z = enc.latent_dist.mean
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    return z * sf


def decode_from_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    # latents are scaled by scaling_factor
    if vae is None:
        raise RuntimeError("VAE is None")
    sf = getattr(vae.config, "scaling_factor", 0.18215)
    img = vae.decode(latents / sf).sample
    return img


def save_checkpoint(path: str, model: nn.Module, ema: nn.Module, opt: torch.optim.Optimizer, step: int, args):
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "opt": opt.state_dict(),
        "step": step,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="ImageNet train folder (ImageFolder)")
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--mode", type=str, choices=["pixel", "latent"], default="latent")

    # latent params
    p.add_argument("--vae_id", type=str, default="", help="diffusers VAE id or path (latent mode)")
    p.add_argument("--latent_h", type=int, default=32)
    p.add_argument("--latent_w", type=int, default=32)
    p.add_argument("--latent_c", type=int, default=4)

    # generator params
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--register_tokens", type=int, default=16)
    p.add_argument("--style_tokens", type=int, default=32)
    p.add_argument("--style_codebook", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=1000)

    # drifting loss / batching (Table 8)
    p.add_argument("--Nc", type=int, default=128)
    p.add_argument("--Nneg", type=int, default=64)
    p.add_argument("--Npos", type=int, default=128)
    p.add_argument("--Nuncond", type=int, default=32)
    p.add_argument("--taus", type=float, nargs="+", default=[0.02, 0.05, 0.2])

    # CFG alpha sampling (Table 8)
    p.add_argument("--alpha_min", type=float, default=1.0)
    p.add_argument("--alpha_max", type=float, default=4.0)
    p.add_argument("--alpha_power", type=float, default=5.0)
    p.add_argument("--p_alpha_eq_1", type=float, default=0.0)

    # optimizer (Table 8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # real batch to refresh queues (Appendix A.8 says push 64 new reals)
    p.add_argument("--real_batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)

    # AMP
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # checkpointing
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="")

    args = p.parse_args()

    rank, world, local = setup_ddp()
    device = torch.device("cuda", local)

    os.makedirs(args.out_dir, exist_ok=True)

    # data
    ds, loader, sampler = build_imagenet_loader(args.data_root, args.real_batch, args.workers, rank, world)
    real_iter = infinite_loader(loader)

    # models
    gen = build_generator(args).to(device)
    ema = deepcopy(gen).to(device)
    for p_ in ema.parameters():
        p_.requires_grad_(False)

    vae = load_vae(args, device) if args.mode == "latent" else None

    # feature encoder: ResNet50 pooled groups (you can swap with your MAE/SimCLR/MoCo)
    feat_encoder = PixelFeatureEncoder(group_cfg=FeatureGroupsConfig(grids=(1,2,4), include_std=True)).to(device)

    # optimizer
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd)

    # resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        gen.load_state_dict(ckpt["model"])
        if ckpt.get("ema") and ema is not None:
            ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        if rank == 0:
            print(f"Resumed from {args.resume} @ step {start_step}")

    gen = DDP(gen, device_ids=[local], output_device=local, broadcast_buffers=False, find_unused_parameters=False)

    # queues (latent mode stores latents; pixel mode stores images downsampled? We'll store images as latents-like just for API consistency.)
    if args.mode == "latent":
        qcfg = QueueConfig(num_classes=args.num_classes, per_class_size=128, global_size=1000,
                           latent_shape=(args.latent_c, args.latent_h, args.latent_w), dtype=torch.float16)
    else:
        # WARNING: storing full 256x256 images per class is huge; for pixel mode, we store smaller (3,64,64) thumbnails as a demo.
        qcfg = QueueConfig(num_classes=args.num_classes, per_class_size=64, global_size=512,
                           latent_shape=(3, 64, 64), dtype=torch.float16)

    queue = LatentQueue(qcfg, device=torch.device("cpu"))

    # alpha schedule
    alpha_sched = CFGAlphaSchedule(args.alpha_min, args.alpha_max, args.alpha_power, args.p_alpha_eq_1)

    # AMP
    use_amp = args.amp and torch.cuda.is_available()
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # warmup the queue
    if rank == 0:
        print("Warming up queues with real data...")
    for _ in range(200):  # ~200 * real_batch images per rank
        imgs, labels = next(real_iter)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.mode == "latent":
            lat = encode_to_latents(vae, imgs, sample=False)
            queue.push(lat.cpu(), labels.cpu())
        else:
            thumb = torch.nn.functional.interpolate(imgs, size=(64,64), mode="bilinear", align_corners=False)
            queue.push(thumb.cpu(), labels.cpu())

    if rank == 0:
        print("Start training...")

    t0 = time.time()

    for step in range(start_step, args.max_steps):
        sampler.set_epoch(step)

        # === Update queues with fresh real samples (Appendix A.8) ===
        imgs, labels = next(real_iter)
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.mode == "latent":
            with torch.no_grad():
                lat = encode_to_latents(vae, imgs, sample=False)  # [B,4,32,32]
                queue.push(lat.cpu(), labels.cpu())
        else:
            with torch.no_grad():
                thumb = torch.nn.functional.interpolate(imgs, size=(64,64), mode="bilinear", align_corners=False)
                queue.push(thumb.cpu(), labels.cpu())

        # === Sample Nc classes and per-class CFG alpha ===
        class_ids = torch.randperm(args.num_classes, device=device)[:args.Nc]
        alphas = sample_powerlaw_alpha(args.Nc, alpha_sched, device=device)  # [Nc]

        # === Generate negatives: Nneg per class (Appendix A.9) ===
        if args.mode == "latent":
            eps = torch.randn((args.Nc * args.Nneg, args.latent_c, args.latent_h, args.latent_w), device=device)
        else:
            eps = torch.randn((args.Nc * args.Nneg, 3, args.image_size, args.image_size), device=device)

        c = class_ids[:, None].expand(args.Nc, args.Nneg).reshape(-1)
        a = alphas[:, None].expand(args.Nc, args.Nneg).reshape(-1)

        # optionally style ids for reproducibility
        style_ids = torch.randint(0, args.style_codebook, (eps.shape[0], args.style_tokens), device=device)

        # LR warmup + cosine decay
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / args.warmup_steps
        else:
            # cosine to 10% of base lr
            t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))
        for pg in opt.param_groups:
            pg["lr"] = lr

        gen.train()
        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            x = gen(eps, c, a, style_ids=style_ids)  # [Nc*Nneg, ...]
            if args.mode == "latent":
                x_lat = x
                # decode to pixels for feature encoder
                x_img = decode_from_latents(vae, x_lat)
            else:
                x_img = x

            # reshape per class
            x_img = x_img.view(args.Nc, args.Nneg, *x_img.shape[1:])

            # sample positives + uncond from queues, then decode if needed
            y_pos_lat = []
            y_unc_lat = []
            for i in range(args.Nc):
                cid = int(class_ids[i].item())
                y_pos_lat.append(queue.sample_class(cid, args.Npos, device=device))
                if args.Nuncond > 0:
                    y_unc_lat.append(queue.sample_uncond(args.Nuncond, device=device))
            y_pos_lat = torch.stack(y_pos_lat, dim=0)  # [Nc,Npos,C,H,W] (latent or thumb)
            y_unc_lat = torch.stack(y_unc_lat, dim=0) if args.Nuncond > 0 else None

            if args.mode == "latent":
                y_pos_img = decode_from_latents(vae, y_pos_lat.view(-1, args.latent_c, args.latent_h, args.latent_w))
                y_pos_img = y_pos_img.view(args.Nc, args.Npos, *y_pos_img.shape[1:])
                if args.Nuncond > 0:
                    y_unc_img = decode_from_latents(vae, y_unc_lat.view(-1, args.latent_c, args.latent_h, args.latent_w))
                    y_unc_img = y_unc_img.view(args.Nc, args.Nuncond, *y_unc_img.shape[1:])
                else:
                    y_unc_img = None
            else:
                # thumbnails are stored in queue; upsample to 256 for feature encoder
                y_pos_img = torch.nn.functional.interpolate(y_pos_lat.view(-1,3,64,64), size=(args.image_size,args.image_size), mode="bilinear", align_corners=False)
                y_pos_img = y_pos_img.view(args.Nc, args.Npos, 3, args.image_size, args.image_size)
                if args.Nuncond > 0:
                    y_unc_img = torch.nn.functional.interpolate(y_unc_lat.view(-1,3,64,64), size=(args.image_size,args.image_size), mode="bilinear", align_corners=False)
                    y_unc_img = y_unc_img.view(args.Nc, args.Nuncond, 3, args.image_size, args.image_size)
                else:
                    y_unc_img = None

            # === Feature extraction ===
            # flatten across classes for encoder, then reshape back
            xg = feat_encoder(x_img.view(-1, *x_img.shape[2:]))       # list of groups, each [Nc*Nneg, L, D]
            pg = feat_encoder(y_pos_img.view(-1, *y_pos_img.shape[2:]))
            ug = feat_encoder(y_unc_img.view(-1, *y_unc_img.shape[2:])) if y_unc_img is not None else None

            # reshape groups to [Nc, N*, L, D]
            x_groups = [g.view(args.Nc, args.Nneg, g.shape[1], g.shape[2]) for g in xg]
            p_groups = [g.view(args.Nc, args.Npos, g.shape[1], g.shape[2]) for g in pg]
            u_groups = [g.view(args.Nc, args.Nuncond, g.shape[1], g.shape[2]) for g in ug] if ug is not None else None

            # === Drifting loss per class (Appendix A.9: per label independently) ===
            loss = 0.0
            for i in range(args.Nc):
                alpha_i = float(alphas[i].item())
                w_unc = cfg_uncond_weight(alpha_i, args.Nneg, args.Nuncond)

                xi = [gg[i] for gg in x_groups]  # list of [Nneg, L, D]
                pi = [gg[i] for gg in p_groups]  # list of [Npos, L, D]
                ui = None if u_groups is None else [gg[i] for gg in u_groups]  # list of [Nuncond, L, D]

                loss = loss + drifting_loss_multi_group(
                    xi, pi, uncond_groups=ui, uncond_w=w_unc, taus=args.taus
                )

            loss = loss / args.Nc

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), args.grad_clip)
            opt.step()

        # EMA
        ema_update(ema, gen.module, decay=args.ema_decay)

        # logging
        if step % 50 == 0:
            # average loss across ranks
            loss_val = loss.detach()
            if world > 1:
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                loss_val = loss_val / world
            if rank == 0:
                dt = time.time() - t0
                print(f"step {step:06d} | loss {loss_val.item():.6f} | lr {lr:.2e} | {dt/60:.1f} min")
                t0 = time.time()

        # checkpoint
        if rank == 0 and (step + 1) % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{step+1:07d}.pt")
            save_checkpoint(ckpt_path, gen.module, ema, opt, step + 1, args)
            print(f"Saved {ckpt_path}")

    if rank == 0:
        save_checkpoint(os.path.join(args.out_dir, "ckpt_last.pt"), gen.module, ema, opt, args.max_steps, args)
        print("Done.")


if __name__ == "__main__":
    main()
