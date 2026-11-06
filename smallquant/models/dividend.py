"""European option with a schedule of fixed cash dividends."""

from __future__ import annotations

import torch

from .black_scholes import price as bs_price
from ._common import as_tensors


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    dividend_times: torch.Tensor,
    dividend_amounts: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Escrowed-dividend Black-Scholes price.

    Dividends with times in ``[0, expiry]`` are removed from the spot at their
    prepaid value ``amount * exp(-(rate - div) * time)``.
    """
    spot, strike, expiry, rate, vol, div = as_tensors(spot, strike, expiry, rate, vol, div)
    times, amounts = as_tensors(dividend_times, dividend_amounts)
    inside = (times >= 0) & (times <= expiry)
    prepaid = amounts * torch.exp(-(rate - div) * times)
    riskless = torch.where(inside, prepaid, torch.zeros_like(prepaid)).sum()
    risky = spot - riskless
    # bs_price discounts the dividend yield on the risky spot, matching the
    # escrowed forward spot * exp(-div T) / exp(-rate T).
    return bs_price(
        spot=risky,
        strike=strike,
        expiry=expiry,
        rate=rate,
        vol=vol,
        div=div,
        is_call=is_call,
    )
