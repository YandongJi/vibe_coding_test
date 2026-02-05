from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class QueueConfig:
    num_classes: int = 1000
    per_class_size: int = 128      # Appendix A.8
    global_size: int = 1000        # Appendix A.8
    latent_shape: Tuple[int, int, int] = (4, 32, 32)  # (C,H,W)
    dtype: torch.dtype = torch.float16


class LatentQueue:
    """
    Ring-buffer queues for:
      - per-class positives
      - global unconditional pool

    Stores latents on CPU to conserve GPU memory.
    """
    def __init__(self, cfg: QueueConfig, device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        C, H, W = cfg.latent_shape
        self.class_buf = torch.empty((cfg.num_classes, cfg.per_class_size, C, H, W), dtype=cfg.dtype, device=device)
        self.class_ptr = torch.zeros((cfg.num_classes,), dtype=torch.long, device=device)
        self.class_count = torch.zeros((cfg.num_classes,), dtype=torch.long, device=device)

        self.global_buf = torch.empty((cfg.global_size, C, H, W), dtype=cfg.dtype, device=device)
        self.global_ptr = torch.zeros((), dtype=torch.long, device=device)
        self.global_count = torch.zeros((), dtype=torch.long, device=device)

    @torch.no_grad()
    def push(self, latents: torch.Tensor, labels: torch.Tensor):
        """latents: [B,C,H,W] (CPU or GPU), labels: [B]"""
        latents = latents.detach().to(device=self.class_buf.device, dtype=self.cfg.dtype)
        labels = labels.detach().to(device=self.class_buf.device)

        B = latents.shape[0]
        for i in range(B):
            c = int(labels[i].item())
            p = int(self.class_ptr[c].item())
            self.class_buf[c, p].copy_(latents[i])
            self.class_ptr[c] = (p + 1) % self.cfg.per_class_size
            self.class_count[c] = min(self.cfg.per_class_size, int(self.class_count[c].item()) + 1)

            # global
            gp = int(self.global_ptr.item())
            self.global_buf[gp].copy_(latents[i])
            self.global_ptr = (gp + 1) % self.cfg.global_size
            self.global_count = min(self.cfg.global_size, int(self.global_count.item()) + 1)

    @torch.no_grad()
    def sample_class(self, class_id: int, n: int, device: torch.device) -> torch.Tensor:
        """Return [n,C,H,W]"""
        cnt = int(self.class_count[class_id].item())
        if cnt <= 0:
            raise RuntimeError(f"class queue empty for class {class_id}")
        maxn = self.cfg.per_class_size if cnt >= self.cfg.per_class_size else cnt
        if cnt >= n:
            idx = torch.randperm(maxn, device=self.class_buf.device)[:n]
        else:
            idx = torch.randint(0, maxn, (n,), device=self.class_buf.device)
        out = self.class_buf[class_id, idx].to(device=device, dtype=torch.float32)
        return out

    @torch.no_grad()
    def sample_uncond(self, n: int, device: torch.device) -> torch.Tensor:
        cnt = int(self.global_count.item())
        if cnt <= 0:
            raise RuntimeError("global queue empty")
        maxn = self.cfg.global_size if cnt >= self.cfg.global_size else cnt
        if cnt >= n:
            idx = torch.randperm(maxn, device=self.global_buf.device)[:n]
        else:
            idx = torch.randint(0, maxn, (n,), device=self.global_buf.device)
        out = self.global_buf[idx].to(device=device, dtype=torch.float32)
        return out
