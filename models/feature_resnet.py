from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class FeatureGroupsConfig:
    # which pooled grids to include as feature groups
    grids: Tuple[int, ...] = (1, 2, 4)   # 1 => global, 2 => 2x2, 4 => 4x4
    include_std: bool = True


def _imagenet_normalize(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


class ResNet50MultiStage(nn.Module):
    """Return feature maps from layer1..layer4 (no classifier head)."""
    def __init__(self, weights: str = "IMAGENET1K_V2"):
        super().__init__()
        w = getattr(torchvision.models, "ResNet50_Weights").__dict__.get(weights, None)
        if w is None:
            # fallback for older torchvision
            base = torchvision.models.resnet50(pretrained=True)
        else:
            base = torchvision.models.resnet50(weights=w)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Freeze weights; allow gradients to flow to inputs.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x_norm: torch.Tensor) -> List[torch.Tensor]:
        # x_norm is ImageNet-normalized.
        x = self.conv1(x_norm)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


def _pool_mean(x: torch.Tensor, grid: int) -> torch.Tensor:
    # x: [B,C,H,W] -> [B,C,grid,grid]
    if grid == 1:
        return x.mean(dim=(2,3), keepdim=True)
    return F.adaptive_avg_pool2d(x, (grid, grid))


def _pool_std(x: torch.Tensor, grid: int, eps: float = 1e-6) -> torch.Tensor:
    # std over patch regions using E[x^2] - (E[x])^2
    m = _pool_mean(x, grid)
    m2 = _pool_mean(x * x, grid)
    var = (m2 - m * m).clamp_min(0.0)
    return torch.sqrt(var + eps)


def make_feature_groups_from_maps(
    maps: List[torch.Tensor],
    cfg: FeatureGroupsConfig = FeatureGroupsConfig(),
) -> List[torch.Tensor]:
    """
    Convert multi-stage feature maps into drifting feature groups.

    Each returned tensor is [B, L, C], where L is number of locations in this group.
    We include:
      - pooled means at grids in cfg.grids (1,2,4)
      - pooled stds (optional) at the same grids
    """
    out: List[torch.Tensor] = []
    for fm in maps:
        B, C, H, W = fm.shape
        for g in cfg.grids:
            pm = _pool_mean(fm, g)   # [B,C,g,g] or [B,C,1,1]
            pm = pm.flatten(2).transpose(1,2)  # [B, L=g*g, C]
            out.append(pm)
            if cfg.include_std:
                ps = _pool_std(fm, g)
                ps = ps.flatten(2).transpose(1,2)
                out.append(ps)
    return out


class PixelFeatureEncoder(nn.Module):
    """
    Wrapper:
      images in [-1,1] -> map to [0,1] -> ImageNet normalize -> ResNet features -> groups
    """
    def __init__(self, weights: str = "IMAGENET1K_V2", group_cfg: FeatureGroupsConfig = FeatureGroupsConfig()):
        super().__init__()
        self.backbone = ResNet50MultiStage(weights=weights)
        self.group_cfg = group_cfg

    def forward(self, img_m11: torch.Tensor) -> List[torch.Tensor]:
        x01 = (img_m11 + 1.0) * 0.5
        xnorm = _imagenet_normalize(x01)
        maps = self.backbone(xnorm)             # list of [B,C,H,W]
        groups = make_feature_groups_from_maps(maps, self.group_cfg)
        # also include input x^2 channel-mean like Appendix A.5 (cheap)
        x2_mean = (xnorm * xnorm).mean(dim=(2,3), keepdim=False)  # [B,3]
        groups.append(x2_mean[:, None, :])  # [B,1,3]
        return groups
