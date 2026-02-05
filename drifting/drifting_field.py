import math
from typing import Optional

import torch


@torch.no_grad()
def compute_drifting_field_V(
    x: torch.Tensor,                # [N, D]
    y_pos: torch.Tensor,            # [N_pos, D]
    y_neg: torch.Tensor,            # [N_neg, D]
    T: float,
    *,
    ignore_self_in_neg: bool = False,
    neg_logweights: Optional[torch.Tensor] = None,  # [N_neg]
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Algorithm 2 from the paper (Appendix A.1).

    logit = -||x - y|| / T  (L2 distance)
    A_row = softmax_y(logit)
    A_col = softmax_x(logit)
    A = sqrt(A_row * A_col)
    then:
      W_pos = A_pos * sum(A_neg over y_neg)
      W_neg = A_neg * sum(A_pos over y_pos)
      V = (W_pos @ y_pos) - (W_neg @ y_neg)

    `neg_logweights` implements CFG unconditional-negative weighting:
      logit_neg += log(w) for the unconditional subset.
    """
    assert x.dim() == 2 and y_pos.dim() == 2 and y_neg.dim() == 2
    N, D = x.shape
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    assert y_pos.shape[1] == D and y_neg.shape[1] == D

    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]

    if ignore_self_in_neg:
        # Assume y_neg[:N] corresponds to x (same ordering).
        n_self = min(N, N_neg)
        big = 1e6
        dist_neg[:, :n_self] = dist_neg[:, :n_self] + torch.eye(N, n_self, device=x.device, dtype=x.dtype) * big

    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T

    if neg_logweights is not None:
        assert neg_logweights.shape == (N_neg,)
        logit_neg = logit_neg + neg_logweights.view(1, N_neg)

    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # normalize along both dimensions
    A_row = torch.softmax(logit, dim=1)  # over y
    A_col = torch.softmax(logit, dim=0)  # over x
    A = torch.sqrt(A_row * A_col + eps)

    A_pos = A[:, :N_pos]
    A_neg = A[:, N_pos:]

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    V = drift_pos - drift_neg
    return V


@torch.no_grad()
def feature_scale_S(
    x: torch.Tensor,      # [Nx, D]
    y: torch.Tensor,      # [Ny, D]
    *,
    y_weights: Optional[torch.Tensor] = None,  # [Ny]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Appendix A.6 feature normalization.

      S = (1/sqrt(D)) * E[ ||x - y|| ]

    We estimate E with batch averages. If `y_weights` is provided, we compute a
    weighted mean over y (needed for CFG unconditional negatives per A.7).
    """
    Nx, D = x.shape
    dist = torch.cdist(x, y)  # [Nx, Ny]
    if y_weights is None:
        mean_dist = dist.mean()
    else:
        assert y_weights.shape == (y.shape[0],)
        w = y_weights.clamp_min(0)
        denom = w.sum().clamp_min(eps)
        mean_dist = (dist * w.view(1, -1)).sum(dim=1).div(denom).mean()

    S = mean_dist / math.sqrt(D)
    return S.clamp_min(eps)


@torch.no_grad()
def drift_scale_lambda(V: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Appendix A.6 drift normalization:

      lambda = sqrt( E[ (1/D) * ||V||^2 ] )
    """
    _, D = V.shape
    lam = torch.sqrt((V.pow(2).sum(dim=1).mean() / D).clamp_min(eps))
    return lam
