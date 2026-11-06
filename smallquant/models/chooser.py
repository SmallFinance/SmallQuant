"""Simple and complex chooser options."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_dr78, black_price, norm_cdf


def simple_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    choose: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
) -> torch.Tensor:
    """Simple chooser: at ``choose`` the holder picks a call or a put expiring at ``expiry``."""
    spot, strike, choose, expiry, rate, vol, div = as_tensors(
        spot, strike, choose, expiry, rate, vol, div
    )
    d = (torch.log(spot / strike) + (rate - div + 0.5 * vol * vol) * expiry) / (
        vol * torch.sqrt(expiry)
    )
    y = (
        torch.log(spot / strike) + (rate - div) * expiry + 0.5 * vol * vol * choose
    ) / (vol * torch.sqrt(choose))
    return (
        spot * torch.exp(-div * expiry) * norm_cdf(d)
        - strike * torch.exp(-rate * expiry) * norm_cdf(d - vol * torch.sqrt(expiry))
        - spot * torch.exp(-div * expiry) * norm_cdf(-y)
        + strike * torch.exp(-rate * expiry) * norm_cdf(-y + vol * torch.sqrt(choose))
    )


def complex_price(
    *,
    spot: torch.Tensor,
    strike_call: torch.Tensor,
    strike_put: torch.Tensor,
    choose: torch.Tensor,
    call_expiry: torch.Tensor,
    put_expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
) -> torch.Tensor:
    """Complex chooser with distinct call and put strikes and expiries."""
    spot, strike_call, strike_put, choose, call_expiry, put_expiry, rate, vol, div = as_tensors(
        spot, strike_call, strike_put, choose, call_expiry, put_expiry, rate, vol, div
    )
    critical = _critical_spot(spot, strike_call, strike_put, choose, call_expiry, put_expiry, rate, vol, div)
    tc = call_expiry - choose
    tp = put_expiry - choose
    sqrt_t = torch.sqrt(choose)
    d1 = (torch.log(spot / critical) + (rate - div + 0.5 * vol * vol) * choose) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    y1 = (torch.log(spot / strike_call) + (rate - div + 0.5 * vol * vol) * tc) / (vol * torch.sqrt(tc))
    y2 = (torch.log(spot / strike_put) + (rate - div + 0.5 * vol * vol) * tp) / (vol * torch.sqrt(tp))
    rho1 = torch.sqrt(choose / tc)
    rho2 = torch.sqrt(choose / tp)
    value = spot * torch.exp(-div * tc) * bivariate_dr78(d1, y1, rho1) - strike_call * torch.exp(
        -rate * tc
    ) * bivariate_dr78(d2, y1 - vol * torch.sqrt(tc), rho1)
    value = value - spot * torch.exp(-div * tp) * bivariate_dr78(-d1, -y2, rho2)
    value = value + strike_put * torch.exp(-rate * tp) * bivariate_dr78(
        -d2, -y2 + vol * torch.sqrt(tp), rho2
    )
    return value


def _critical_spot(spot, strike_call, strike_put, choose, call_expiry, put_expiry, rate, vol, div):
    """Newton search matching QuantLib's complex-chooser critical value."""
    tc = call_expiry - 2.0 * choose
    tp = put_expiry - 2.0 * choose
    guess = spot.clone()
    for _ in range(30):
        call, call_delta = _bs_and_delta(guess, strike_call, tc, rate, vol, div, True)
        put, put_delta = _bs_and_delta(guess, strike_put, tp, rate, vol, div, False)
        gap = call - put
        slope = call_delta - put_delta
        guess = guess - gap / slope
        if torch.all(gap.abs() <= 0.001):
            break
    return guess


def _bs_and_delta(spot, strike, expiry, rate, vol, div, is_call: bool):
    flag = torch.ones_like(spot, dtype=torch.bool) if is_call else torch.zeros_like(spot, dtype=torch.bool)
    std = vol * torch.sqrt(torch.clamp(expiry, min=1e-12))
    discount = torch.exp(-rate * expiry)
    growth = torch.exp(-div * expiry)
    forward = spot * growth / discount
    value = black_price(forward, strike, std, discount, flag)
    d1 = torch.log(forward / strike) / std + 0.5 * std
    sign = 1.0 if is_call else -1.0
    delta = sign * growth * norm_cdf(sign * d1)
    return value, delta
