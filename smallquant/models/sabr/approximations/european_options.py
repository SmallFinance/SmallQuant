"""European option prices under SABR: Black-76 at the Hagan implied volatility."""

from __future__ import annotations

import torch

from smallquant.black_scholes.vanilla_prices import option_price

from .implied_volatility import implied_vol

__all__ = ["european_option_price"]


def european_option_price(
    *,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    discount: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rho: torch.Tensor,
    nu: torch.Tensor,
    is_call: torch.Tensor | bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """SABR European price.

    Args:
        forward: forward price
        strike: strike
        expiry: year fraction
        discount: zero-bond price to expiry
        alpha: initial volatility level
        beta: backbone exponent in ``[0, 1]``
        rho: correlation
        nu: volatility of volatility
        is_call: boolean tensor, true for a call and false for a put
    """
    vol = implied_vol(
        forward=forward, strike=strike, expiry=expiry, alpha=alpha, beta=beta, rho=rho, nu=nu, dtype=dtype
    )
    return option_price(
        forward=forward, strike=strike, expiry=expiry, vol=vol, discount=discount, is_call=is_call, dtype=dtype
    )
