"""Bachelier price of a European option on a forward."""

from __future__ import annotations

import torch

from ._common import as_tensors, bachelier_price


def price(
    *,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    normal_vol: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Normal-model price. ``normal_vol`` is the absolute volatility of the forward."""
    forward, strike, expiry, normal_vol, discount = as_tensors(
        forward, strike, expiry, normal_vol, discount
    )
    return bachelier_price(
        forward, strike, normal_vol * torch.sqrt(expiry), discount, is_call
    )
