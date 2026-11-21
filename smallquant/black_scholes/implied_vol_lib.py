"""Implied volatility entry point with a choice of method."""

from __future__ import annotations

import enum

import torch

from . import implied_vol_approximation, implied_vol_newton_root
from .implied_vol_utils import UnderlyingDistribution

__all__ = ["ImpliedVolMethod", "implied_vol"]


class ImpliedVolMethod(enum.Enum):
    """How ``implied_vol`` solves.

    ``FAST_APPROX`` is a one-shot closed-form approximation with no iterations, for very
    large chains or as a starting point. ``NEWTON`` is the exact safeguarded Newton solve.
    """

    FAST_APPROX = 1
    NEWTON = 2


def implied_vol(
    price: torch.Tensor,
    *,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    is_call: torch.Tensor | bool = True,
    spot: torch.Tensor | None = None,
    forward: torch.Tensor | None = None,
    rate: torch.Tensor | None = None,
    div: torch.Tensor | None = None,
    discount: torch.Tensor | None = None,
    underlying_distribution: UnderlyingDistribution = UnderlyingDistribution.LOGNORMAL,
    method: ImpliedVolMethod = ImpliedVolMethod.NEWTON,
    tol: torch.Tensor | None = None,
    max_iter: int = 50,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Implied volatility of a chain of European prices.

    Pass ``spot`` with ``rate`` (and optionally ``div``), or ``forward`` with ``discount``
    or ``rate``, as in ``option_price``. ``method`` is ``ImpliedVolMethod.NEWTON``
    (default, exact) or ``ImpliedVolMethod.FAST_APPROX`` (lognormal only, no iterations).
    ``tol`` is a tensor broadcast with the chain, or ``None`` for ``1e-12`` (lognormal) or
    ``1e-10`` (normal). Each option stops once its price error is within its own ``tol``.
    ``max_iter`` is the shared step count. Both apply to Newton only.

    Example:
        >>> implied_vol(price, spot=spot, strike=strike, expiry=expiry, rate=rate,
        ...             method=ImpliedVolMethod.FAST_APPROX)
    """
    if method == ImpliedVolMethod.NEWTON:
        return implied_vol_newton_root.implied_vol(
            price, strike=strike, expiry=expiry, is_call=is_call, spot=spot, forward=forward,
            rate=rate, div=div, discount=discount, underlying_distribution=underlying_distribution,
            tol=tol, max_iter=max_iter, dtype=dtype,
        )
    if method == ImpliedVolMethod.FAST_APPROX:
        if underlying_distribution != UnderlyingDistribution.LOGNORMAL:
            raise ValueError("FAST_APPROX supports lognormal volatility only")
        return implied_vol_approximation.implied_vol(
            price, strike=strike, expiry=expiry, is_call=is_call, spot=spot, forward=forward,
            rate=rate, div=div, discount=discount, dtype=dtype,
        )
    raise ValueError(f"unknown method {method!r}")
