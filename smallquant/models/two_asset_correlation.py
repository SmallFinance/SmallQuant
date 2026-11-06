"""Two-asset correlation option."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_dr78


def price(
    *,
    spot1: torch.Tensor,
    spot2: torch.Tensor,
    strike1: torch.Tensor,
    strike2: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    div1: torch.Tensor,
    div2: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Option on asset 2 paying when asset 1 finishes beyond ``strike1``.

    The payoff is ``max(phi * (spot2 - strike2), 0)`` if asset 1 is in the money
    at ``strike1``, and zero otherwise.
    """
    spot1, spot2, strike1, strike2, expiry, rate, vol1, vol2, div1, div2, correlation = (
        as_tensors(
            spot1, spot2, strike1, strike2, expiry, rate, vol1, vol2, div1, div2, correlation
        )
    )
    b1 = rate - div1
    b2 = rate - div2
    sqrt_t = torch.sqrt(expiry)
    y1 = (torch.log(spot1 / strike1) + (b1 - 0.5 * vol1 * vol1) * expiry) / (vol1 * sqrt_t)
    y2 = (torch.log(spot2 / strike2) + (b2 - 0.5 * vol2 * vol2) * expiry) / (vol2 * sqrt_t)
    call = spot2 * torch.exp((b2 - rate) * expiry) * bivariate_dr78(
        y2 + vol2 * sqrt_t, y1 + correlation * vol2 * sqrt_t, correlation
    ) - strike2 * torch.exp(-rate * expiry) * bivariate_dr78(y2, y1, correlation)
    put = strike2 * torch.exp(-rate * expiry) * bivariate_dr78(
        -y2, -y1, correlation
    ) - spot2 * torch.exp((b2 - rate) * expiry) * bivariate_dr78(
        -y2 - vol2 * sqrt_t, -y1 - correlation * vol2 * sqrt_t, correlation
    )
    return torch.where(is_call, call, put)
