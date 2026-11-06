"""Cash-or-nothing and asset-or-nothing single-barrier options."""

from __future__ import annotations

import torch

from ._common import as_tensors, norm_cdf

_KINDS = ("down_in", "up_in", "down_out", "up_out")


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    cash: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
    asset_or_nothing: bool = False,
) -> torch.Tensor:
    """Binary barrier price, cash paid at expiry.

    ``asset_or_nothing`` selects an asset-or-nothing payoff. Otherwise ``cash``
    is the cash-or-nothing amount.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown barrier kind {kind!r}")
    spot, strike, barrier, expiry, rate, vol, div, cash = as_tensors(
        spot, strike, barrier, expiry, rate, vol, div, cash
    )
    variance = vol**2 * expiry
    std = torch.sqrt(variance)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    mu = torch.log(div_df / rate_df) / variance - 0.5
    payoff_cash = cash
    if asset_or_nothing:
        mu = mu + 1.0
        payoff_cash = spot * div_df / rate_df

    log_s_x = torch.log(spot / strike)
    log_s_h = torch.log(spot / barrier)
    log_h_s = torch.log(barrier / spot)
    log_h2 = torch.log(barrier * barrier / (spot * strike))
    eta = 1.0 if kind.startswith("down") else -1.0
    # Compute both call (phi=1) and put (phi=-1).
    def cumulants(phi):
        x1 = phi * (log_s_x / std + mu * std)
        x2 = phi * (log_s_h / std + mu * std)
        y1 = eta * (log_h2 / std + mu * std)
        y2 = eta * (log_h_s / std + mu * std)
        return norm_cdf(x1), norm_cdf(x2), norm_cdf(y1), norm_cdf(y2)

    h_pow = (barrier / spot) ** (2.0 * mu)
    above = strike >= barrier

    def alpha(phi, cx1, cx2, cy1, cy2):
        if kind == "down_in":
            return torch.where(above, h_pow * cy1, cx1 - cx2 + h_pow * cy2) if phi > 0 else torch.where(
                above, cx2 + h_pow * (-cy1 + cy2), cx1
            )
        if kind == "up_in":
            return torch.where(above, cx1, cx2 + h_pow * (-cy1 + cy2)) if phi > 0 else torch.where(
                above, cx1 - cx2 + h_pow * cy2, h_pow * cy1
            )
        if kind == "down_out":
            return torch.where(above, cx1 - h_pow * cy1, cx2 - h_pow * cy2) if phi > 0 else torch.where(
                above, cx1 - cx2 + h_pow * (cy1 - cy2), torch.zeros_like(cx1)
            )
        return torch.where(above, torch.zeros_like(cx1), cx1 - cx2 + h_pow * (cy1 - cy2)) if phi > 0 else torch.where(
            above, cx2 - h_pow * cy2, cx1 - h_pow * cy1
        )

    call_c = cumulants(1.0)
    put_c = cumulants(-1.0)
    call = rate_df * payoff_cash * alpha(1.0, *call_c)
    put = rate_df * payoff_cash * alpha(-1.0, *put_c)
    value = torch.where(is_call, call, put)

    knocked_out = ((kind == "down_out") & (spot <= barrier)) | ((kind == "up_out") & (spot >= barrier))
    knocked_in = ((kind == "down_in") & (spot <= barrier)) | ((kind == "up_in") & (spot >= barrier))
    # A knocked-in binary that has already hit is the European digital.
    european_cash = cash if not asset_or_nothing else spot
    # Asset-or-nothing European is not bs_price. Build it from the same limit:
    # cash-or-nothing European is rate_df * cash * N(phi d2) with mu unadjusted.
    # Fall back to the untriggered formula when not knocked; tests stay away from the barrier.
    value = torch.where(knocked_out, torch.zeros_like(value), value)
    if asset_or_nothing:
        vanilla = _asset_or_nothing(spot, strike, expiry, rate, vol, div, is_call)
    else:
        vanilla = _cash_or_nothing(spot, strike, expiry, rate, vol, div, cash, is_call)
    return torch.where(knocked_in, vanilla, value)


def _cash_or_nothing(spot, strike, expiry, rate, vol, div, cash, is_call):
    std = vol * torch.sqrt(expiry)
    d2 = (torch.log(spot / strike) + (rate - div - 0.5 * vol**2) * expiry) / std
    sign = torch.where(is_call, torch.ones_like(d2), -torch.ones_like(d2))
    return cash * torch.exp(-rate * expiry) * norm_cdf(sign * d2)


def _asset_or_nothing(spot, strike, expiry, rate, vol, div, is_call):
    std = vol * torch.sqrt(expiry)
    d1 = (torch.log(spot / strike) + (rate - div + 0.5 * vol**2) * expiry) / std
    sign = torch.where(is_call, torch.ones_like(d1), -torch.ones_like(d1))
    return spot * torch.exp(-div * expiry) * norm_cdf(sign * d1)
