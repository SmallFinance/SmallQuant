"""Forward-starting and forward-performance European options."""

from __future__ import annotations

import torch

from .black_scholes import price as bs_price
from ._common import as_tensors


def price(
    *,
    spot: torch.Tensor,
    moneyness: torch.Tensor,
    reset: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Forward-starting vanilla struck at ``moneyness`` times the spot at ``reset``."""
    spot, moneyness, reset, expiry, rate, vol, div = as_tensors(
        spot, moneyness, reset, expiry, rate, vol, div
    )
    tau = expiry - reset
    vanilla = bs_price(
        spot=spot,
        strike=moneyness * spot,
        expiry=tau,
        rate=rate,
        vol=vol,
        div=div,
        is_call=is_call,
    )
    return torch.exp(-div * reset) * vanilla


def performance_price(
    *,
    spot: torch.Tensor,
    moneyness: torch.Tensor,
    reset: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Forward-starting performance option, paid as a fraction of the future spot."""
    spot, moneyness, reset, expiry, rate, vol, div = as_tensors(
        spot, moneyness, reset, expiry, rate, vol, div
    )
    tau = expiry - reset
    vanilla = bs_price(
        spot=spot,
        strike=moneyness * spot,
        expiry=tau,
        rate=rate,
        vol=vol,
        div=div,
        is_call=is_call,
    )
    return torch.exp(-rate * reset) * vanilla / spot
