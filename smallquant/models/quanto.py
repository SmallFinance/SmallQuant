"""Quanto options under Black-Scholes."""

from __future__ import annotations

import torch

from .barrier import price as barrier_price
from .black_scholes import price as bs_price
from .forward_start import price as forward_price
from ._common import as_tensors


def _quanto_dividend(div, rate, foreign_rate, vol, fx_vol, correlation):
    return div + rate - foreign_rate + correlation * vol * fx_vol


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    foreign_rate: torch.Tensor,
    fx_vol: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """European quanto price.

    ``rate`` is the domestic risk-free rate. The quanto dividend yield is
    ``div + rate - foreign_rate + correlation * vol * fx_vol``.
    """
    spot, strike, expiry, rate, vol, div, foreign_rate, fx_vol, correlation = as_tensors(
        spot, strike, expiry, rate, vol, div, foreign_rate, fx_vol, correlation
    )
    quanto_div = _quanto_dividend(div, rate, foreign_rate, vol, fx_vol, correlation)
    return bs_price(
        spot=spot,
        strike=strike,
        expiry=expiry,
        rate=rate,
        vol=vol,
        div=quanto_div,
        is_call=is_call,
    )


def forward_start_price(
    *,
    spot: torch.Tensor,
    moneyness: torch.Tensor,
    reset: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    foreign_rate: torch.Tensor,
    fx_vol: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Quanto forward-starting European option."""
    spot, moneyness, reset, expiry, rate, vol, div, foreign_rate, fx_vol, correlation = as_tensors(
        spot, moneyness, reset, expiry, rate, vol, div, foreign_rate, fx_vol, correlation
    )
    return forward_price(
        spot=spot,
        moneyness=moneyness,
        reset=reset,
        expiry=expiry,
        rate=rate,
        vol=vol,
        div=_quanto_dividend(div, rate, foreign_rate, vol, fx_vol, correlation),
        is_call=is_call,
    )


def barrier_option_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    rebate: torch.Tensor,
    foreign_rate: torch.Tensor,
    fx_vol: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Quanto single-barrier European option."""
    spot, strike, barrier, expiry, rate, vol, div, rebate, foreign_rate, fx_vol, correlation = (
        as_tensors(
            spot, strike, barrier, expiry, rate, vol, div, rebate, foreign_rate, fx_vol, correlation
        )
    )
    return barrier_price(
        spot=spot,
        strike=strike,
        barrier=barrier,
        expiry=expiry,
        rate=rate,
        vol=vol,
        div=_quanto_dividend(div, rate, foreign_rate, vol, fx_vol, correlation),
        rebate=rebate,
        is_call=is_call,
        kind=kind,
    )
