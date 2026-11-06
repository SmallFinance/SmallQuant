"""Holder-extensible and writer-extensible options."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_dr78, bivariate_west, black_price, norm_cdf


def writer_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    extended_strike: torch.Tensor,
    expiry: torch.Tensor,
    extended_expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Writer-extensible option."""
    spot, strike, extended_strike, expiry, extended_expiry, rate, vol, div = as_tensors(
        spot, strike, extended_strike, expiry, extended_expiry, rate, vol, div
    )
    carry = rate - div
    forward = spot * torch.exp(carry * expiry)
    discount = torch.exp(-rate * expiry)
    black = black_price(forward, strike, vol * torch.sqrt(expiry), discount, is_call)
    rho = -torch.sqrt(expiry / extended_expiry)
    z1 = (torch.log(spot / extended_strike) + (carry + 0.5 * vol * vol) * extended_expiry) / (
        vol * torch.sqrt(extended_expiry)
    )
    z2 = (torch.log(spot / strike) + (carry + 0.5 * vol * vol) * expiry) / (vol * torch.sqrt(expiry))
    call_extra = spot * torch.exp((carry - rate) * extended_expiry) * bivariate_west(
        z1, -z2, rho
    ) - extended_strike * torch.exp(-rate * extended_expiry) * bivariate_west(
        z1 - vol * torch.sqrt(extended_expiry), -z2 + vol * torch.sqrt(expiry), rho
    )
    put_extra = -spot * torch.exp((carry - rate) * extended_expiry) * bivariate_west(
        -z1, z2, rho
    ) + extended_strike * torch.exp(-rate * extended_expiry) * bivariate_west(
        -z1 + vol * torch.sqrt(extended_expiry), z2 - vol * torch.sqrt(expiry), rho
    )
    return black + torch.where(is_call, call_extra, put_extra)


def holder_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    extended_strike: torch.Tensor,
    premium: torch.Tensor,
    expiry: torch.Tensor,
    extended_expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Holder-extensible option. The holder pays ``premium`` at ``expiry`` to extend."""
    spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div = as_tensors(
        spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div
    )
    carry = rate - div
    rho = torch.sqrt(expiry / extended_expiry)
    i1_call, i2_call = _call_levels(
        spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div
    )
    i1_put, i2_put = _put_levels(
        spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div
    )

    def y(level):
        return (torch.log(spot / level) + (carry + 0.5 * vol * vol) * expiry) / (
            vol * torch.sqrt(expiry)
        )

    z1 = (torch.log(spot / extended_strike) + (carry + 0.5 * vol * vol) * extended_expiry) / (
        vol * torch.sqrt(extended_expiry)
    )
    z2 = (torch.log(spot / strike) + (carry + 0.5 * vol * vol) * expiry) / (vol * torch.sqrt(expiry))
    y1c, y2c = y(i2_call), y(i1_call)
    y1p, y2p = y(i2_put), y(i1_put)
    growth1 = torch.exp(-div * expiry)
    discount1 = torch.exp(-rate * expiry)
    call_bs = black_price(spot * torch.exp(carry * expiry), strike, vol * torch.sqrt(expiry), discount1, torch.ones_like(spot, dtype=torch.bool))
    put_bs = black_price(spot * torch.exp(carry * expiry), strike, vol * torch.sqrt(expiry), discount1, torch.zeros_like(spot, dtype=torch.bool))
    call = (
        call_bs
        + spot * torch.exp(-div * extended_expiry) * _rect(y1c, y2c, z1, rho)
        - extended_strike
        * torch.exp(-rate * extended_expiry)
        * _rect(y1c - vol * torch.sqrt(expiry), y2c - vol * torch.sqrt(expiry), z1 - vol * torch.sqrt(extended_expiry), rho)
        - spot * growth1 * _span(y1c, z2)
        + strike * discount1 * _span(y1c - vol * torch.sqrt(expiry), z2 - vol * torch.sqrt(expiry))
        - premium * discount1 * _span(y1c - vol * torch.sqrt(expiry), y2c - vol * torch.sqrt(expiry))
    )
    put = (
        put_bs
        - spot * torch.exp(-div * extended_expiry) * _rect(y1p, y2p, -z1, rho)
        + extended_strike
        * torch.exp(-rate * extended_expiry)
        * _rect(
            y1p - vol * torch.sqrt(expiry),
            y2p - vol * torch.sqrt(expiry),
            -z1 + vol * torch.sqrt(extended_expiry),
            rho,
        )
        + spot * growth1 * _span(z2, y2p)
        - strike * discount1 * _span(z2 - vol * torch.sqrt(expiry), y2p - vol * torch.sqrt(expiry))
        - premium * discount1 * _span(y1p - vol * torch.sqrt(expiry), y2p - vol * torch.sqrt(expiry))
    )
    return torch.where(is_call, call, put)


def _bivariate(left, right, rho):
    """Drezner bivariate CDF, with infinite limits reduced to a univariate CDF."""
    finite = bivariate_dr78(
        torch.where(torch.isfinite(left), left, torch.zeros_like(left)),
        torch.where(torch.isfinite(right), right, torch.zeros_like(right)),
        rho,
    )
    left_pos = torch.isposinf(left)
    right_pos = torch.isposinf(right)
    value = torch.where(left_pos, norm_cdf(right), finite)
    value = torch.where(right_pos & ~left_pos, norm_cdf(left), value)
    value = torch.where(left_pos & right_pos, torch.ones_like(value), value)
    negative = torch.isneginf(left) | torch.isneginf(right)
    return torch.where(negative, torch.zeros_like(value), value)


def _rect(lower_a, upper_a, upper_b, rho):
    """M(b, d) - M(a, d), the rectangle with lower limit -infinity."""
    return _bivariate(upper_a, upper_b, rho) - _bivariate(lower_a, upper_b, rho)


def _span(left, right):
    return norm_cdf(right) - norm_cdf(left)


def _bs(spot, strike, expiry, rate, vol, div, is_call: bool):
    flag = torch.ones_like(spot, dtype=torch.bool) if is_call else torch.zeros_like(spot, dtype=torch.bool)
    std = vol * torch.sqrt(expiry)
    discount = torch.exp(-rate * expiry)
    growth = torch.exp(-div * expiry)
    value = black_price(spot * growth / discount, strike, std, discount, flag)
    d1 = torch.log(spot / strike) / std + (rate - div + 0.5 * vol * vol) * expiry / std
    sign = 1.0 if is_call else -1.0
    delta = sign * growth * norm_cdf(sign * d1)
    return value, delta


def _call_levels(spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div):
    tau = extended_expiry - expiry
    i1 = spot.clone()
    if torch.all(premium == 0):
        i1 = torch.zeros_like(spot)
    else:
        for _ in range(40):
            value, delta = _bs(i1, extended_strike, tau, rate, vol, div, True)
            gap = value - premium
            i1 = i1 - gap / delta
            if torch.all(gap.abs() <= 0.001):
                break
    boundary = strike - extended_strike * torch.exp(-rate * tau)
    i2 = spot.clone()
    if torch.all(premium < boundary):
        i2 = torch.full_like(spot, float("inf"))
    else:
        for _ in range(40):
            value, delta = _bs(i2, extended_strike, tau, rate, vol, div, True)
            gap = value - premium - i2 + strike
            i2 = i2 - gap / (delta - 1.0)
            if torch.all(gap.abs() <= 0.001):
                break
    return i1, i2


def _put_levels(spot, strike, extended_strike, premium, expiry, extended_expiry, rate, vol, div):
    tau = extended_expiry - expiry
    i1 = spot.clone()
    for _ in range(40):
        value, delta = _bs(i1, extended_strike, tau, rate, vol, div, False)
        gap = value - premium + i1 - strike
        slope = delta - 1.0
        i1 = i1 - gap / slope
        if torch.all(gap.abs() <= 0.001):
            break
    i2 = spot.clone()
    if torch.all(premium == 0):
        i2 = torch.full_like(spot, float("inf"))
    else:
        for _ in range(40):
            value, delta = _bs(i2, extended_strike, tau, rate, vol, div, False)
            gap = value - premium
            i2 = i2 - gap / delta
            if torch.all(gap.abs() <= 0.001):
                break
    return i1, i2
