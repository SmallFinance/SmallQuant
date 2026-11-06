"""Geske compound option."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_dr78, black_price, norm_cdf


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    underlying_strike: torch.Tensor,
    expiry: torch.Tensor,
    underlying_expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    underlying_is_call: torch.Tensor,
) -> torch.Tensor:
    """Compound option.

    ``expiry`` is when the compound right is exercised for ``strike``. The
    underlying option expires at ``underlying_expiry`` with ``underlying_strike``.
    """
    spot, strike, underlying_strike, expiry, underlying_expiry, rate, vol, div = as_tensors(
        spot, strike, underlying_strike, expiry, underlying_expiry, rate, vol, div
    )
    tau = underlying_expiry - expiry
    std_tau = vol * torch.sqrt(tau)
    dd_tau = torch.exp(-div * tau)
    rd_tau = torch.exp(-rate * tau)
    critical = _critical_spot(
        underlying_strike, strike, std_tau, dd_tau, rd_tau, underlying_is_call
    )
    std_mother = vol * torch.sqrt(expiry)
    dd_mother = torch.exp(-div * expiry)
    rd_mother = torch.exp(-rate * expiry)
    scaled = rd_mother * critical / (spot * dd_mother)
    transformed = torch.log(scaled * torch.exp(0.5 * std_mother**2)) / std_mother

    std_daughter = vol * torch.sqrt(underlying_expiry)
    dd_daughter = torch.exp(-div * underlying_expiry)
    rd_daughter = torch.exp(-rate * underlying_expiry)
    forward = spot * dd_daughter / rd_daughter
    d_plus = torch.log(forward / underlying_strike) / std_daughter + 0.5 * std_daughter
    d_minus = d_plus - std_daughter
    phi = torch.where(underlying_is_call, torch.ones_like(spot), -torch.ones_like(spot))
    w = torch.where(is_call, torch.ones_like(spot), -torch.ones_like(spot))
    rho = torch.sqrt(expiry / underlying_expiry)
    mother_gap = transformed - std_mother
    n_gap = bivariate_dr78(-phi * w * mother_gap, phi * d_plus, w * rho)
    n_x = bivariate_dr78(-phi * w * transformed, phi * d_minus, w * rho)
    n_plain = norm_cdf(-phi * w * transformed)
    return (
        phi * w * spot * dd_daughter * n_gap
        - phi * w * underlying_strike * rd_daughter * n_x
        - w * strike * rd_mother * n_plain
    )


def _critical_spot(underlying_strike, compound_strike, std_dev, div_df, rate_df, underlying_is_call):
    guess = underlying_strike.clone()
    for _ in range(40):
        forward = guess * div_df / rate_df
        value = black_price(forward, underlying_strike, std_dev, rate_df, underlying_is_call)
        gap = value - compound_strike
        # dV/dS = div_df * sign * N(sign d1)
        d1 = torch.log(forward / underlying_strike) / std_dev + 0.5 * std_dev
        sign = torch.where(underlying_is_call, torch.ones_like(d1), -torch.ones_like(d1))
        slope = div_df * sign * norm_cdf(sign * d1)
        guess = torch.clamp(guess - gap / slope, min=1e-8)
        if torch.all(gap.abs() <= 1e-8):
            break
    return guess
