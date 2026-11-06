"""Black-76 price of a European option on a forward."""

from __future__ import annotations

import torch

from ._common import as_tensors, black_price


def price(
    *,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    vol: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Black-76 price.

    ``discount`` is the zero-bond price to expiry. ``vol`` is the lognormal volatility.
    """
    forward, strike, expiry, vol, discount = as_tensors(
        forward, strike, expiry, vol, discount
    )
    return black_price(forward, strike, vol * torch.sqrt(expiry), discount, is_call)
