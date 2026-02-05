import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn.functional as F

from .drifting_field import compute_drifting_field_V, feature_scale_S, drift_scale_lambda


@dataclass
class CFGAlphaSchedule:
    alpha_min: float = 1.0
    alpha_max: float = 4.0
    power_exponent: float = 5.0     # p(alpha) ∝ alpha^{-power_exponent}
    p_alpha_eq_1: float = 0.0       # optional point mass at alpha=1


def sample_powerlaw_alpha(n: int, sched: CFGAlphaSchedule, device: torch.device) -> torch.Tensor:
    """Power-law sampling used in Table 8 (Appendix A.7)."""
    a, b, k = float(sched.alpha_min), float(sched.alpha_max), float(sched.power_exponent)
    assert a >= 1.0 and b > a and k > 0

    u = torch.rand(n, device=device)
    if sched.p_alpha_eq_1 > 0:
        mask = u < sched.p_alpha_eq_1
        out = torch.empty(n, device=device)
        out[mask] = 1.0
        u2 = torch.rand(int((~mask).sum().item()), device=device)
        out[~mask] = _inv_cdf_powerlaw(u2, a, b, k)
        return out
    return _inv_cdf_powerlaw(u, a, b, k)


def _inv_cdf_powerlaw(u: torch.Tensor, a: float, b: float, k: float) -> torch.Tensor:
    # p(x) ∝ x^{-k} on [a,b]
    if abs(k - 1.0) < 1e-6:
        return a * torch.exp(u * math.log(b / a))
    one_minus_k = 1.0 - k
    A = a ** one_minus_k
    B = b ** one_minus_k
    return (u * (B - A) + A).pow(1.0 / one_minus_k)


def cfg_uncond_weight(alpha: float, Nneg: int, Nuncond: int) -> float:
    """
    Appendix A.7: given CFG strength α, compute unconditional-negative weight w:

        α = ((Nneg-1) + Nuncond*w) / (Nneg-1)
        => w = (α-1) * (Nneg-1) / Nuncond
    """
    if Nuncond <= 0:
        return 0.0
    return max(alpha - 1.0, 0.0) * max(Nneg - 1, 1) / float(Nuncond)


def drifting_loss_group(
    x_feat: torch.Tensor,       # [Nneg, L, D]  L = number of locations/features in this group
    pos_feat: torch.Tensor,     # [Npos, L, D]
    *,
    uncond_feat: Optional[torch.Tensor] = None,  # [Nuncond, L, D]
    uncond_w: float = 0.0,
    taus: Sequence[float] = (0.02, 0.05, 0.2),
) -> torch.Tensor:
    """
    Implements Eq. (26) + multi-τ aggregation in Appendix A.6, for one feature group.
    - Feature normalization S (Eq. 21)
    - Drift normalization λ (Eq. 25)
    - Multiple temperatures τ, aggregated drift sum_τ V/λ

    This group supports multiple spatial locations (L). We compute drift per-location,
    but compute S and λ across all locations jointly (Appendix A.6 last paragraph).

    Returns a scalar loss.
    """
    assert x_feat.dim() == 3 and pos_feat.dim() == 3
    Nneg, L, D = x_feat.shape
    Npos = pos_feat.shape[0]

    if uncond_feat is None:
        Nuncond = 0
        y_neg_feat = x_feat
    else:
        Nuncond = uncond_feat.shape[0]
        assert uncond_feat.shape[1:] == (L, D)
        y_neg_feat = torch.cat([x_feat, uncond_feat], dim=0)  # [Nneg+Nuncond, L, D]

    # ===== Feature normalization (Eq. 21) =====
    # flatten locations for scale estimation
    x_flat = x_feat.reshape(Nneg * L, D)
    y_pos_flat = pos_feat.reshape(Npos * L, D)
    y_neg_flat = y_neg_feat.reshape((Nneg + Nuncond) * L, D)
    y_all = torch.cat([y_pos_flat, y_neg_flat], dim=0)

    if Nuncond > 0 and uncond_w > 0:
        # weights: pos=1, gen neg=1, uncond neg=w
        w_pos = torch.ones(Npos * L, device=x_feat.device, dtype=x_feat.dtype)
        w_gen = torch.ones(Nneg * L, device=x_feat.device, dtype=x_feat.dtype)
        w_unc = torch.full((Nuncond * L,), float(uncond_w), device=x_feat.device, dtype=x_feat.dtype)
        y_w = torch.cat([w_pos, w_gen, w_unc], dim=0)
    else:
        y_w = None

    S = feature_scale_S(x_flat, y_all, y_weights=y_w)  # scalar
    x_n = x_feat / S
    pos_n = pos_feat / S
    yneg_n = y_neg_feat / S

    # prepare neg weights for Alg. 2 (CFG)
    if Nuncond > 0 and uncond_w > 0:
        logw = math.log(float(uncond_w))
        neg_logweights = torch.cat([
            torch.zeros(Nneg, device=x_feat.device, dtype=x_feat.dtype),
            torch.full((Nuncond,), logw, device=x_feat.device, dtype=x_feat.dtype),
        ], dim=0)  # [Nneg+Nuncond]
    else:
        neg_logweights = None

    # ===== Drift computation: per τ, per location =====
    V_sum = torch.zeros_like(x_n)  # [Nneg, L, D]

    for tau in taus:
        T = float(tau) * math.sqrt(D)  # τ̃ = τ * sqrt(Cj)

        V_loc = []
        for j in range(L):
            xj = x_n[:, j, :]                 # [Nneg, D]
            posj = pos_n[:, j, :]             # [Npos, D]
            negj = yneg_n[:, j, :]            # [Nneg+Nuncond, D]

            Vj = compute_drifting_field_V(
                xj, posj, negj, T,
                ignore_self_in_neg=True,
                neg_logweights=neg_logweights,
            )  # [Nneg, D]
            V_loc.append(Vj)

        V_tau = torch.stack(V_loc, dim=1)  # [Nneg, L, D]

        # Drift normalization over all locations (Eq. 25 + Appendix A.6)
        lam = drift_scale_lambda(V_tau.reshape(Nneg * L, D))
        V_tau = V_tau / lam
        V_sum = V_sum + V_tau

    target = (x_n + V_sum).detach()
    return F.mse_loss(x_n, target)


def drifting_loss_multi_group(
    x_groups: List[torch.Tensor],      # each [Nneg, L, D]
    pos_groups: List[torch.Tensor],    # each [Npos, L, D]
    *,
    uncond_groups: Optional[List[torch.Tensor]] = None,  # each [Nuncond, L, D]
    uncond_w: float = 0.0,
    taus: Sequence[float] = (0.02, 0.05, 0.2),
) -> torch.Tensor:
    """Sum of drifting losses across feature groups (Eq. 14, Appendix A.5/A.6)."""
    assert len(x_groups) == len(pos_groups)
    if uncond_groups is not None:
        assert len(uncond_groups) == len(x_groups)
    loss = 0.0
    for i in range(len(x_groups)):
        loss = loss + drifting_loss_group(
            x_groups[i],
            pos_groups[i],
            uncond_feat=None if uncond_groups is None else uncond_groups[i],
            uncond_w=uncond_w,
            taus=taus,
        )
    return loss
