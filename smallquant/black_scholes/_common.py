"""Shared tensor helpers for closed-form prices."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.distributions import (
    bivariate_dr78,
    bivariate_west,
    normal_cdf,
    normal_pdf,
)

__all__ = [
    "as_tensors",
    "bivariate_dr78",
    "bivariate_west",
    "black_price",
    "bachelier_price",
    "norm_cdf",
    "norm_pdf",
    "option_sign",
]

norm_cdf = normal_cdf
norm_pdf = normal_pdf


def as_tensors(*values: torch.Tensor, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, ...]:
    """Align tensor inputs on one dtype and device. Non-tensors raise."""
    return align(*values, dtype=dtype)


def option_sign(is_call: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.where(is_call, torch.ones_like(like), -torch.ones_like(like))


def black_price(
    forward: torch.Tensor,
    strike: torch.Tensor,
    std_dev: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Black-76 price. ``std_dev`` is volatility times sqrt(expiry)."""
    sign = option_sign(is_call, forward)
    intrinsic = torch.clamp(sign * (forward - strike), min=0.0) * discount
    safe = torch.clamp(std_dev, min=1e-12)
    d1 = torch.log(forward / strike) / safe + 0.5 * safe
    d2 = d1 - safe
    formula = discount * sign * (forward * norm_cdf(sign * d1) - strike * norm_cdf(sign * d2))
    return torch.where(std_dev > 0, formula, intrinsic)


def bachelier_price(
    forward: torch.Tensor,
    strike: torch.Tensor,
    std_dev: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Bachelier (normal) price. ``std_dev`` is normal volatility times sqrt(expiry)."""
    sign = option_sign(is_call, forward)
    gap = sign * (forward - strike)
    intrinsic = discount * torch.clamp(gap, min=0.0)
    safe = torch.clamp(std_dev, min=1e-12)
    h = gap / safe
    formula = discount * (safe * norm_pdf(h) + gap * norm_cdf(h))
    return torch.where(std_dev > 0, formula, intrinsic)
