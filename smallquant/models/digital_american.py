"""American cash-or-nothing and asset-or-nothing options."""

from __future__ import annotations

import torch

from ._common import as_tensors, norm_cdf


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    cash: torch.Tensor,
    is_call: torch.Tensor,
    payoff_at_expiry: bool,
    knock_in: bool = True,
    asset_or_nothing: bool = False,
) -> torch.Tensor:
    """American digital.

    ``payoff_at_expiry`` selects the at-expiry formula. Otherwise the cash is
    paid when the strike is hit. ``knock_in`` is used by the at-expiry formula;
    the knock-out engine is ``knock_in=False``.
    """
    spot, strike, expiry, rate, vol, div, cash = as_tensors(
        spot, strike, expiry, rate, vol, div, cash
    )
    if payoff_at_expiry:
        return _at_expiry(spot, strike, expiry, rate, vol, div, cash, is_call, knock_in, asset_or_nothing)
    return _at_hit(spot, strike, expiry, rate, vol, div, cash, is_call, asset_or_nothing)


def _at_expiry(spot, strike, expiry, rate, vol, div, cash, is_call, knock_in, asset_or_nothing):
    variance = vol**2 * expiry
    std = torch.sqrt(variance)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    forward = spot * div_df / rate_df
    mu = torch.log(div_df / rate_df) / variance - 0.5
    payoff_cash = forward if asset_or_nothing else cash
    if asset_or_nothing:
        mu = mu + 1.0
    log_h_s = torch.log(strike / spot)
    log_s_h = torch.log(spot / strike)

    def one(call: bool):
        if call:
            eta, phi = -1.0, (1.0 if knock_in else -1.0)
        else:
            eta, phi = 1.0, (-1.0 if knock_in else 1.0)
        d1 = phi * (log_s_h / std + mu * std)
        d2 = eta * (log_h_s / std + mu * std)
        cum1 = norm_cdf(d1)
        cum2 = norm_cdf(d2)
        if call:
            already = strike <= spot
        else:
            already = strike >= spot
        if knock_in:
            cum1 = torch.where(already, torch.full_like(cum1, 0.5), cum1)
            cum2 = torch.where(already, torch.full_like(cum2, 0.5), cum2)
        else:
            cum1 = torch.where(already, torch.zeros_like(cum1), cum1)
            cum2 = torch.where(already, torch.zeros_like(cum2), cum2)
        in_money = already
        y = torch.where(in_money, torch.ones_like(spot), torch.where(cum2 == 0, torch.zeros_like(spot), (strike / spot) ** (2.0 * mu)))
        if not knock_in:
            y = -y
        return rate_df * payoff_cash * (cum1 + y * cum2)

    call = one(True)
    put = one(False)
    return torch.where(is_call, call, put)


def _at_hit(spot, strike, expiry, rate, vol, div, cash, is_call, asset_or_nothing):
    variance = vol**2 * expiry
    std = torch.sqrt(variance)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    log_h_s = torch.log(strike / spot)
    mu = torch.log(div_df / rate_df) / variance - 0.5
    lam = torch.sqrt(mu * mu - 2.0 * torch.log(rate_df) / variance)
    d1 = log_h_s / std + lam * std
    d2 = d1 - 2.0 * lam * std

    def one(call: bool):
        alive = (strike > spot) if call else (strike < spot)
        if call:
            alpha = torch.where(alive, norm_cdf(-d1), torch.full_like(d1, 0.5))
            beta = torch.where(alive, norm_cdf(-d2), torch.full_like(d2, 0.5))
        else:
            alpha = torch.where(alive, norm_cdf(d1), torch.full_like(d1, 0.5))
            beta = torch.where(alive, norm_cdf(d2), torch.full_like(d2, 0.5))
        forward = torch.where(alive, (strike / spot) ** (mu + lam), torch.ones_like(spot))
        reflected = torch.where(alive, (strike / spot) ** (mu - lam), torch.ones_like(spot))
        if asset_or_nothing:
            amount = torch.where(alive, strike, spot)
        else:
            amount = cash
        return amount * (forward * alpha + reflected * beta)

    return torch.where(is_call, one(True), one(False))
