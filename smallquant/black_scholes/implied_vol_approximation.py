"""Fast one-shot Black implied volatility from the Stefanica-Radoicic explicit formula."""

from __future__ import annotations

import math

import torch

from smallquant._tensor import align, check

from .vanilla_prices import _flag

__all__ = ["implied_vol"]

# Polya's approximation N(t) ~ 1/2 + sign(t) / 2 * sqrt(1 - exp(-2 t^2 / pi)).
_POLYA = 2.0 / math.pi
_LOG2 = math.log(2.0)

# The same no-arbitrage tolerances as the Newton solver, on discounted prices.
_INTRINSIC_TOL = 1e-10
_UPPER_TOL = 1e-8


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
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Approximate Black implied volatility in closed form, with no iterations.

    Replacing the normal CDF in the Black formula by Polya's approximation
    ``N(t) ~ 1/2 + sign(t) / 2 * sqrt(1 - exp(-2 t^2 / pi))`` turns the price equation into
    a quadratic in ``exp(-2 gamma / pi)``, which Stefanica and Radoicic (2017) solve
    explicitly. Writing ``y = log(F / K)`` and ``alpha = C / (D K)`` for the normalized
    call price, the total volatility is

        ``sigma sqrt(T) = sqrt(gamma + y) + sqrt(gamma - y)``  if ``alpha > C0(y)``,
        ``sigma sqrt(T) = |sqrt(gamma + y) - sqrt(gamma - y)|`` otherwise,

    where ``C0`` is the Polya price at ``sigma sqrt(T) = sqrt(2 |y|)`` and ``gamma`` comes
    from the positive root of ``A beta^2 + B beta - C = 0`` (Table 3 of the paper).

    The Polya CDF is within 0.0032 of ``N``, so the price at the returned
    volatility is within ``0.0032 (F + K) D`` of the input price. Use it for very large
    chains where that is accurate enough, or as a starting point for an exact solver.

    Pass ``spot`` with ``rate`` (and optionally ``div``), or ``forward`` with ``discount``
    or ``rate``, as in ``option_price``. Puts are converted to calls by put-call parity.
    Prices outside the no-arbitrage bounds are NaN and a price at intrinsic value is zero
    volatility, as in the Newton solver; a price at the upper bound is infinite. The result
    is differentiable by plain autograd through the closed form.

    Returns lognormal (Black) volatility, not total volatility.

    References:
        Dan Stefanica and Rados Radoicic. An explicit implied volatility formula.
        International Journal of Theoretical and Applied Finance 20(7), 2017.
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2908494

    Example:
        >>> one = torch.tensor(1.0, dtype=torch.float64)
        >>> implied_vol(10.45 * one, spot=100 * one, strike=100 * one, expiry=one,
        ...             rate=0.05 * one)
    """
    forward, strike, expiry, discount, price = _market(
        price, strike, expiry, spot, forward, rate, div, discount, dtype
    )
    is_call = _flag(is_call, price)
    check("strike", strike, strike > 0)
    check("forward", forward, forward > 0)
    check("price", price, price >= 0)

    # Normalize by the discounted strike: y = log(F / K), alpha = price / (D K).
    scale = discount * strike
    y = torch.log(forward / strike)
    moneyness = torch.exp(y)
    alpha = price / scale

    # Put-call parity in normalized units: alpha_call - alpha_put = e^y - 1. Each option's
    # own price is used where it is the more accurate side, so OTM puts lose no precision.
    call = torch.where(is_call, alpha, alpha + (moneyness - 1.0))
    put = torch.where(is_call, alpha - (moneyness - 1.0), alpha)
    # Distance to the upper bound e^y for the call, 1 for the put.
    headroom = torch.where(is_call, moneyness - alpha, 1.0 - alpha)

    discounted_forward = discount * forward
    intrinsic = torch.where(is_call, discounted_forward - scale, scale - discounted_forward).clamp(min=0)
    upper = torch.where(is_call, discounted_forward, scale)
    below = price < intrinsic - _INTRINSIC_TOL
    above = price > upper + _UPPER_TOL
    at_intrinsic = (price - intrinsic).abs() <= _INTRINSIC_TOL
    at_upper = ~above & (headroom <= 0)
    bad_expiry = ~(expiry > 0)
    valid = ~(below | above | at_intrinsic | at_upper | bad_expiry) & (call > 0) & (put > 0)

    # Masked inputs: an invalid element gets the midpoint of its no-arbitrage range, so no
    # branch below takes the log or sqrt of a non-positive number or divides by zero.
    floor = (moneyness - 1.0).clamp(min=0)
    mid_call = 0.5 * (floor + moneyness)
    half_cap = 0.5 * torch.minimum(moneyness, torch.ones_like(moneyness))
    call = torch.where(valid, call, mid_call)
    put = torch.where(valid, put, 1.0 - half_cap)
    headroom = torch.where(valid, headroom, half_cap)
    safe_expiry = torch.where(valid, expiry, torch.ones_like(expiry))

    total = _total_vol(y, moneyness, call, put, headroom)
    vol = total / torch.sqrt(safe_expiry)

    vol = torch.where(at_upper & ~bad_expiry, torch.full_like(vol, float("inf")), vol)
    vol = torch.where(bad_expiry, torch.full_like(vol, float("nan")), vol)
    vol = torch.where(at_intrinsic, torch.zeros_like(vol), vol)
    return torch.where(below | above, torch.full_like(vol, float("nan")), vol)


def _market(
    price: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    spot: torch.Tensor | None,
    forward: torch.Tensor | None,
    rate: torch.Tensor | None,
    div: torch.Tensor | None,
    discount: torch.Tensor | None,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the spot-or-forward choice and return aligned forward and discount factor."""
    if (spot is None) == (forward is None):
        raise ValueError("pass exactly one of spot and forward")
    if rate is not None and discount is not None:
        raise ValueError("pass at most one of rate and discount")
    if spot is not None and rate is None:
        raise ValueError("spot needs rate")
    if forward is not None and div is not None:
        raise ValueError("div applies to spot only; a forward already includes carry")
    if forward is not None and rate is None and discount is None:
        raise ValueError("forward needs discount or rate")
    if spot is not None:
        if div is None:
            spot, strike, expiry, rate, price = align(spot, strike, expiry, rate, price, dtype=dtype)
            div = torch.zeros_like(rate)
        else:
            spot, strike, expiry, rate, div, price = align(spot, strike, expiry, rate, div, price, dtype=dtype)
        check("spot", spot, spot > 0)
        return spot * torch.exp((rate - div) * expiry), strike, expiry, torch.exp(-rate * expiry), price
    carry = rate if rate is not None else discount
    forward, strike, expiry, carry, price = align(forward, strike, expiry, carry, price, dtype=dtype)
    discount = torch.exp(-carry * expiry) if rate is not None else carry
    return forward, strike, expiry, discount, price


def _total_vol(
    y: torch.Tensor, moneyness: torch.Tensor, call: torch.Tensor, put: torch.Tensor, headroom: torch.Tensor
) -> torch.Tensor:
    """``sigma sqrt(T)`` from Stefanica-Radoicic, for in-range normalized prices.

    ``call`` and ``put`` are the normalized call and put prices and ``headroom`` is
    ``e^y - call``; all three must be positive.
    """
    k = _POLYA
    # R = 2 alpha_c - e^y + 1 and the coefficients of Table 3, written with cosh/sinh.
    r = call + put
    a = 4.0 * torch.sinh((1.0 - k) * y) ** 2
    b = 8.0 * torch.cosh(k * y) - 4.0 * torch.cosh((1.0 - k) * y) * (
        2.0 * torch.cosh(y) - r * r / moneyness
    )
    # C = e^{-2y} (R^2 - (e^y - 1)^2) ((e^y + 1)^2 - R^2) = 16 e^{-2y} c p (e^y - c)(1 + c),
    # kept in logs since it underflows for deep out-of-the-money prices.
    log_c = (
        math.log(16.0) + torch.log(call) + torch.log(put) + torch.log(headroom)
        + torch.log1p(call) - 2.0 * y
    )
    # beta is the positive root of A beta^2 + B beta - C = 0, as 2C / (B + root), which has
    # no cancellation because B > 0 for every in-range price: R^2 e^{-y} lies strictly
    # between 2 cosh(y) - 2 and 2 cosh(y) + 2, so B > 8 cosh(k y) - 8 cosh((1 - k) y) >= 0
    # since k = 2 / pi > 1 / 2.
    root = torch.sqrt(b * b + 4.0 * a * torch.exp(log_c))
    log_beta = _LOG2 + log_c - torch.log(b + root)
    gamma = -log_beta / k

    # gamma >= |y| in exact arithmetic; the floor only guards the sqrt against rounding.
    tiny = torch.finfo(y.dtype).tiny
    up = torch.sqrt((gamma + y).clamp(min=tiny))
    down = torch.sqrt((gamma - y).clamp(min=tiny))

    # C0: the Polya call price at sigma sqrt(T) = sqrt(2|y|), where d2 = 0 (y > 0) or
    # d1 = 0 (y < 0). Below it the smaller root applies.
    abs_y = y.abs()
    polya_tail = 0.5 * torch.sqrt(-torch.expm1(-2.0 * k * abs_y))
    c0 = torch.where(y >= 0, moneyness * (0.5 + polya_tail) - 0.5, moneyness * 0.5 - (0.5 - polya_tail))
    # |up - down| = 2|y| / (up + down), with no cancellation.
    small = 2.0 * abs_y / (up + down)
    return torch.where(call <= c0, small, up + down)
