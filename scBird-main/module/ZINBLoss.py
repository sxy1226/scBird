from __future__ import annotations

import torch
import torch.nn as nn


class ZINBLoss(nn.Module):
    """Numerically stable zero-inflated negative-binomial negative log likelihood."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        disp: torch.Tensor,
        pi: torch.Tensor,
        scale_factor: torch.Tensor | float = 1.0,
        ridge_lambda: float = 0.0,
    ) -> torch.Tensor:
        eps = self.eps
        if not torch.is_tensor(scale_factor):
            scale_factor = torch.tensor(scale_factor, device=mean.device, dtype=mean.dtype)
        if scale_factor.ndim == 1:
            scale_factor = scale_factor.unsqueeze(1)

        mean = torch.clamp(mean * scale_factor, min=eps, max=1e6)
        disp = torch.clamp(disp, min=eps, max=1e6)
        pi = torch.clamp(pi, min=eps, max=1.0 - eps)
        x = torch.clamp(x, min=0.0)

        log_nb = (
            torch.lgamma(x + disp)
            - torch.lgamma(disp)
            - torch.lgamma(x + 1.0)
            + disp * (torch.log(disp + eps) - torch.log(disp + mean + eps))
            + x * (torch.log(mean + eps) - torch.log(disp + mean + eps))
        )

        zero_nb = torch.exp(disp * (torch.log(disp + eps) - torch.log(disp + mean + eps)))
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        nonzero_case = -torch.log(1.0 - pi + eps) - log_nb
        result = torch.where(x <= eps, zero_case, nonzero_case)

        if ridge_lambda > 0:
            result = result + ridge_lambda * pi.square()
        return result.mean()
