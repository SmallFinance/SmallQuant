"""Stulz options on the minimum or maximum of two assets."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_west, norm_cdf


def _min_call(forward1, forward2, strike, discount, variance1, variance2, rho):
    std1 = torch.sqrt(variance1)
    std2 = torch.sqrt(variance2)
    variance = variance1 + variance2 - 2.0 * rho * std1 * std2
    std = torch.sqrt(variance)
    mod_rho1 = (rho * std2 - std1) / std
    mod_rho2 = (rho * std1 - std2) / std
    d1 = (torch.log(forward1 / forward2) + 0.5 * variance) / std
    d1_1 = (torch.log(forward1 / strike) + 0.5 * variance1) / std1
    d1_2 = (torch.log(forward2 / strike) + 0.5 * variance2) / std2
    alpha = bivariate_west(d1_1, -d1, mod_rho1)
    beta = bivariate_west(d1_2, d1 - std, mod_rho2)
    gamma = bivariate_west(d1_1 - std1, d1_2 - std2, rho)
    zero_strike = strike == 0
    alpha = torch.where(zero_strike, norm_cdf(-d1), alpha)
    beta = torch.where(zero_strike, norm_cdf(d1 - std), beta)
    gamma = torch.where(zero_strike, torch.ones_like(gamma), gamma)
    return discount * (forward1 * alpha + forward2 * beta - strike * gamma)


def price(
    *,
    spot1: torch.Tensor,
    spot2: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    div1: torch.Tensor,
    div2: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
    on_max: bool,
) -> torch.Tensor:
    """European call or put on the minimum or maximum of two assets."""
    spot1, spot2, strike, expiry, rate, vol1, vol2, div1, div2, correlation = as_tensors(
        spot1, spot2, strike, expiry, rate, vol1, vol2, div1, div2, correlation
    )
    discount = torch.exp(-rate * expiry)
    forward1 = spot1 * torch.exp((rate - div1) * expiry)
    forward2 = spot2 * torch.exp((rate - div2) * expiry)
    variance1 = vol1**2 * expiry
    variance2 = vol2**2 * expiry
    min_call = _min_call(forward1, forward2, strike, discount, variance1, variance2, correlation)
    black1 = discount * _black_undiscounted(forward1, strike, torch.sqrt(variance1), True)
    black2 = discount * _black_undiscounted(forward2, strike, torch.sqrt(variance2), True)
    max_call = black1 + black2 - min_call
    call = max_call if on_max else min_call
    zero = torch.zeros_like(strike)
    if on_max:
        max_zero = _min_call(forward1, forward2, zero, discount, variance1, variance2, correlation)
        # euroTwoAssetMaxBasketCall at strike 0 is black(F1,0)+black(F2,0)-min_call(0)
        max_at_zero = discount * forward1 + discount * forward2 - max_zero
        put = strike * discount - max_at_zero + max_call
    else:
        min_zero = _min_call(forward1, forward2, zero, discount, variance1, variance2, correlation)
        put = strike * discount - min_zero + min_call
    return torch.where(is_call, call, put)


def _black_undiscounted(forward, strike, std, is_call_flag: bool):
    from ._common import black_price

    flag = torch.ones_like(forward, dtype=torch.bool) if is_call_flag else torch.zeros_like(
        forward, dtype=torch.bool
    )
    return black_price(forward, torch.clamp(strike, min=1e-12), std, torch.ones_like(forward), flag)
