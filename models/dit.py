import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, T, C], shift/scale: [B, C]
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0):
        super().__init__()
        hidden = int(dim * hidden_mult)
        # 2x for gate/value
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLU(dim, hidden_mult=mlp_mult)

        # adaLN-Zero: produce (shift1, scale1, gate1, shift2, scale2, gate2)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C], cond: [B, C]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)

        x1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out

        x2 = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2[:, None, :] * self.mlp(x2)
        return x


@dataclass
class DiTConfig:
    image_size: int = 256
    in_channels: int = 3
    patch_size: int = 16
    hidden_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    num_classes: int = 1000

    # in-context conditioning tokens (paper uses 16 register tokens)
    n_register_tokens: int = 16

    # random style embedding (paper: 32 tokens from codebook of 64)
    n_style_tokens: int = 32
    style_codebook: int = 64

    # conditioning embedding dim (same as hidden_dim)
    cond_dim: int = 768


class DiTGenerator(nn.Module):
    """
    A lightweight DiT-like generator for one-step generation:
      f(ε, c, α, style) -> x

    - Patchify ε (noise image / latent) into tokens
    - Add register tokens
    - Transformer blocks with adaLN-Zero conditioning
    - Unpatchify to output image/latent
    """
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.image_size % cfg.patch_size == 0
        self.grid = cfg.image_size // cfg.patch_size
        self.n_patches = self.grid * self.grid
        self.token_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size

        self.patch_in = nn.Linear(self.token_dim, cfg.hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_register_tokens + self.n_patches, cfg.hidden_dim))
        nn.init.normal_(self.pos, std=0.02)

        self.register = nn.Parameter(torch.zeros(1, cfg.n_register_tokens, cfg.hidden_dim))
        nn.init.normal_(self.register, std=0.02)

        self.class_emb = nn.Embedding(cfg.num_classes, cfg.cond_dim)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

        self.style_emb = nn.Embedding(cfg.style_codebook, cfg.cond_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_dim, cfg.n_heads) for _ in range(cfg.depth)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim, elementwise_affine=False)
        self.patch_out = nn.Linear(cfg.hidden_dim, self.token_dim)

        # init output to near-zero to stabilize early training
        nn.init.zeros_(self.patch_out.weight)
        nn.init.zeros_(self.patch_out.bias)

    def forward(
        self,
        eps_img: torch.Tensor,          # [B, C, H, W]
        class_labels: torch.Tensor,     # [B]
        alpha: torch.Tensor,            # [B]
        style_ids: Optional[torch.Tensor] = None,  # [B, n_style_tokens] ints
    ) -> torch.Tensor:
        B, C, H, W = eps_img.shape
        assert H == self.cfg.image_size and W == self.cfg.image_size
        assert C == self.cfg.in_channels

        # patchify
        p = self.cfg.patch_size
        x = eps_img.reshape(B, C, self.grid, p, self.grid, p).permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_patches, -1)
        x = self.patch_in(x)

        # add register tokens
        reg = self.register.expand(B, -1, -1)
        x = torch.cat([reg, x], dim=1)  # [B, n_reg + n_patches, dim]
        x = x + self.pos

        # conditioning
        cemb = self.class_emb(class_labels)
        aemb = self.alpha_mlp(torch.log(alpha.clamp_min(1e-6)).unsqueeze(-1))

        if style_ids is None:
            style_ids = torch.randint(0, self.cfg.style_codebook, (B, self.cfg.n_style_tokens), device=eps_img.device)
        semb = self.style_emb(style_ids).sum(dim=1)

        cond = cemb + aemb + semb

        # transformer
        for blk in self.blocks:
            x = blk(x, cond)

        x = self.final_norm(x)
        x = self.patch_out(x[:, self.cfg.n_register_tokens:, :])  # discard register tokens

        # unpatchify
        x = x.reshape(B, self.grid, self.grid, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x
