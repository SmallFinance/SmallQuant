"""Black and Bachelier implied volatility by safeguarded Newton, with an implicit gradient."""

from __future__ import annotations

import math
from typing import Any

import torch

from smallquant._tensor import align, check
from smallquant.math.distributions import normal_cdf, normal_pdf

from ._common import bachelier_price
from .implied_vol_utils import UnderlyingDistribution
from .vanilla_prices import _flag, option_price

__all__ = ["implied_vol"]


def _tolerance(tol: torch.Tensor | None, like: torch.Tensor, default: float) -> torch.Tensor:
    """Per-element tolerance. ``None`` is ``default`` on ``like``."""
    if tol is None:
        return torch.full_like(like, default)
    return align(tol, like)[0]


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
    tol: torch.Tensor | None = None,
    max_iter: int = 50,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Implied volatility of a whole chain by Newton safeguarded with bisection.

    Pass ``spot`` with ``rate`` (and optionally ``div``), or ``forward`` with ``discount``
    or ``rate``, as in ``option_price``. ``underlying_distribution`` picks Black
    (lognormal) or Bachelier (normal) volatility.

    The lognormal initial guess is Brenner-Subrahmanyam. Prices outside the no-arbitrage
    bounds are NaN. A price equal to intrinsic value returns zero volatility. ``tol``
    broadcasts with the price; ``None`` is ``1e-12`` for Black and ``1e-10`` for Bachelier.
    Each element holds its volatility once the price error is within its own ``tol``.
    The backward pass is ``d vol / d price = 1 / vega`` and does not unroll the solver.
    """
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
    if underlying_distribution == UnderlyingDistribution.LOGNORMAL:
        return _lognormal(
            price, strike, expiry, is_call, spot, forward, rate, div, discount, tol, max_iter, dtype
        )
    if underlying_distribution == UnderlyingDistribution.NORMAL:
        return _normal(
            price, strike, expiry, is_call, spot, forward, rate, div, discount, tol, max_iter, dtype
        )
    raise ValueError(f"unknown underlying_distribution {underlying_distribution!r}")


def _lognormal(
    price: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    is_call: torch.Tensor | bool,
    spot: torch.Tensor | None,
    forward: torch.Tensor | None,
    rate: torch.Tensor | None,
    div: torch.Tensor | None,
    discount: torch.Tensor | None,
    tol: torch.Tensor | None,
    max_iter: int,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if spot is None:
        # A forward F with discount D is a spot F whose rate and carry both equal -log(D) / T.
        carry = rate if rate is not None else discount
        forward, strike, expiry, carry, price = align(forward, strike, expiry, carry, price, dtype=dtype)
        rate = carry if discount is None else -torch.log(carry) / expiry
        spot, div = forward, rate
    elif div is None:
        spot, strike, expiry, rate, price = align(spot, strike, expiry, rate, price, dtype=dtype)
        div = torch.zeros_like(rate)
    spot, strike, expiry, rate, div, price = align(spot, strike, expiry, rate, div, price, dtype=dtype)
    is_call = _flag(is_call, spot)
    check("spot", spot, spot > 0)
    check("strike", strike, strike > 0)
    check("expiry", expiry, expiry > 0)
    check("price", price, price >= 0)
    tol = _tolerance(tol, price, 1e-12)
    price, spot, strike, expiry, rate, div, tol, is_call = torch.broadcast_tensors(
        price, spot, strike, expiry, rate, div, tol, is_call
    )
    return _BlackImplied.apply(price, spot, strike, expiry, rate, div, is_call.to(dtype=price.dtype), tol, max_iter)


def _normal(
    price: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    is_call: torch.Tensor | bool,
    spot: torch.Tensor | None,
    forward: torch.Tensor | None,
    rate: torch.Tensor | None,
    div: torch.Tensor | None,
    discount: torch.Tensor | None,
    tol: torch.Tensor | None,
    max_iter: int,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if spot is not None:
        if div is None:
            spot, strike, expiry, rate, price = align(spot, strike, expiry, rate, price, dtype=dtype)
            div = torch.zeros_like(rate)
        else:
            spot, strike, expiry, rate, div, price = align(spot, strike, expiry, rate, div, price, dtype=dtype)
        forward = spot * torch.exp((rate - div) * expiry)
        discount = torch.exp(-rate * expiry)
    elif discount is None:
        forward, strike, expiry, rate, price = align(forward, strike, expiry, rate, price, dtype=dtype)
        discount = torch.exp(-rate * expiry)
    forward, strike, expiry, discount, price = align(forward, strike, expiry, discount, price, dtype=dtype)
    is_call = _flag(is_call, forward)
    tol = _tolerance(tol, price, 1e-10)
    price, forward, strike, expiry, discount, tol, is_call = torch.broadcast_tensors(
        price, forward, strike, expiry, discount, tol, is_call
    )
    return _NormalImplied.apply(
        price, forward, strike, expiry, discount, is_call.to(dtype=price.dtype), tol, max_iter
    )


class _BlackImplied(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        price: torch.Tensor,
        spot: torch.Tensor,
        strike: torch.Tensor,
        expiry: torch.Tensor,
        rate: torch.Tensor,
        div: torch.Tensor,
        is_call: torch.Tensor,
        tol: torch.Tensor,
        max_iter: int,
    ) -> torch.Tensor:
        flag = is_call > 0.5
        with torch.no_grad():
            vol, _active = _solve_black(price, spot, strike, expiry, rate, div, flag, tol, int(max_iter))
        ctx.save_for_backward(price, spot, strike, expiry, rate, div, is_call, vol)
        return vol

    @staticmethod
    def backward(
        ctx: Any, grad_vol: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        price, spot, strike, expiry, rate, div, is_call, vol = ctx.saved_tensors
        flag = is_call > 0.5
        with torch.enable_grad():
            leaves = tuple(tensor.detach().requires_grad_(True) for tensor in (spot, strike, expiry, rate, vol, div))
            model = option_price(
                spot=leaves[0],
                strike=leaves[1],
                expiry=leaves[2],
                rate=leaves[3],
                vol=leaves[4],
                div=leaves[5],
                is_call=flag,
            )
            grads = torch.autograd.grad(model.sum(), leaves)
        vega = grads[4]
        active = torch.isfinite(vol) & (vol > 0) & (vega.abs() > 0)
        unit = torch.where(active, grad_vol / vega, torch.zeros_like(grad_vol))
        # d vol / d theta = - (d price / d theta) / vega
        return (
            unit,
            -grads[0] * unit,
            -grads[1] * unit,
            -grads[2] * unit,
            -grads[3] * unit,
            -grads[5] * unit,
            None,
            None,
            None,
        )


class _NormalImplied(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        price: torch.Tensor,
        forward: torch.Tensor,
        strike: torch.Tensor,
        expiry: torch.Tensor,
        discount: torch.Tensor,
        is_call: torch.Tensor,
        tol: torch.Tensor,
        max_iter: int,
    ) -> torch.Tensor:
        flag = is_call > 0.5
        with torch.no_grad():
            vol = _solve_normal(price, forward, strike, expiry, discount, flag, tol, int(max_iter))
        ctx.save_for_backward(price, forward, strike, expiry, discount, is_call, vol)
        return vol

    @staticmethod
    def backward(
        ctx: Any, grad_vol: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        _, forward, strike, expiry, discount, is_call, vol = ctx.saved_tensors
        flag = is_call > 0.5
        with torch.enable_grad():
            leaves = tuple(
                tensor.detach().requires_grad_(True) for tensor in (forward, strike, expiry, vol, discount)
            )
            std = leaves[3] * torch.sqrt(leaves[2].clamp(min=1e-12))
            model = bachelier_price(leaves[0], leaves[1], std, leaves[4], flag)
            grads = torch.autograd.grad(model.sum(), leaves)
        vega = grads[3]
        unit = torch.where(torch.isfinite(vol) & (vol > 0), grad_vol / vega.clamp(min=1e-12), torch.zeros_like(grad_vol))
        return (unit, -grads[0] * unit, -grads[1] * unit, -grads[2] * unit, -grads[4] * unit, None, None, None)


def _solve_black(
    price: torch.Tensor,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    tol: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expiry_safe = expiry.clamp(min=1e-12)
    discount_spot = spot * torch.exp(-div * expiry_safe)
    discount_strike = strike * torch.exp(-rate * expiry_safe)
    intrinsic = torch.where(is_call, discount_spot - discount_strike, discount_strike - discount_spot).clamp(min=0)
    upper = torch.where(is_call, discount_spot, discount_strike)
    below = price < intrinsic - 1e-10
    above = price > upper + 1e-8
    at_intrinsic = (price - intrinsic).abs() <= 1e-10
    # Brenner-Subrahmanyam, using the discounted spot so dividend yield is in the guess.
    guess = (price / discount_spot.clamp(min=1e-8)) * math.sqrt(2.0 * math.pi) / torch.sqrt(expiry_safe)
    lo = torch.full_like(price, 1e-8)
    hi = torch.full_like(price, 5.0)
    vol = guess.clamp(lo, hi)
    settled = torch.zeros(price.shape, dtype=torch.bool, device=price.device)
    for _ in range(max_iter):
        model, vega = _black_and_vega(spot, strike, expiry_safe, rate, div, vol, is_call)
        settled = settled | ((model - price).abs() <= tol)
        too_high = model > price
        hi = torch.where(settled, hi, torch.where(too_high, vol, hi))
        lo = torch.where(settled, lo, torch.where(too_high, lo, vol))
        newton = vol - (model - price) / vega.clamp(min=1e-10)
        inside = (newton > lo) & (newton < hi) & torch.isfinite(newton)
        candidate = torch.where(inside, newton, 0.5 * (lo + hi))
        vol = torch.where(settled, vol, candidate)
    vol = torch.where(at_intrinsic, torch.zeros_like(vol), vol)
    vol = torch.where(below | above, torch.full_like(vol, float("nan")), vol)
    return vol, ~(below | above | at_intrinsic)


def _black_and_vega(
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    div: torch.Tensor,
    vol: torch.Tensor,
    is_call: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_t = torch.sqrt(expiry)
    d1 = (torch.log(spot / strike) + (rate - div + 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount_spot = spot * torch.exp(-div * expiry)
    discount_strike = strike * torch.exp(-rate * expiry)
    call = discount_spot * normal_cdf(d1) - discount_strike * normal_cdf(d2)
    put = discount_strike * normal_cdf(-d2) - discount_spot * normal_cdf(-d1)
    model = torch.where(is_call, call, put)
    vega = discount_spot * normal_pdf(d1) * sqrt_t
    return model, vega


def _solve_normal(
    price: torch.Tensor,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
    tol: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    std_scale = torch.sqrt(expiry.clamp(min=1e-12))
    intrinsic = discount * torch.where(is_call, forward - strike, strike - forward).clamp(min=0)
    undiscounted = price / discount.clamp(min=1e-12)
    # A normal-model guess: price ≈ discount * std / sqrt(2 pi) at the money.
    gap = (forward - strike).abs()
    guess = torch.where(
        gap < 1e-8,
        undiscounted * math.sqrt(2.0 * math.pi) / std_scale,
        gap / std_scale,
    )
    lo = torch.full_like(price, 1e-6)
    hi = torch.full_like(price, forward.abs() * 5.0 + 1.0)
    vol = guess.clamp(min=1e-4)
    settled = torch.zeros(price.shape, dtype=torch.bool, device=price.device)
    for _ in range(max_iter):
        model = bachelier_price(forward, strike, vol * std_scale, discount, is_call)
        settled = settled | ((model - price).abs() <= tol)
        # vega w.r.t. normal vol
        h = torch.where(is_call, forward - strike, strike - forward) / (vol * std_scale).clamp(min=1e-8)
        vega = discount * std_scale * normal_pdf(h)
        too_high = model > price
        hi = torch.where(settled, hi, torch.where(too_high, vol, hi))
        lo = torch.where(settled, lo, torch.where(too_high, lo, vol))
        newton = vol - (model - price) / vega.clamp(min=1e-10)
        inside = (newton > lo) & (newton < hi) & torch.isfinite(newton)
        candidate = torch.where(inside, newton, 0.5 * (lo + hi))
        vol = torch.where(settled, vol, candidate)
    at_intrinsic = (price - intrinsic).abs() <= 1e-8
    upper = discount * torch.where(is_call, torch.relu(forward) + strike.abs(), strike.abs() + torch.relu(-forward))
    # Normal prices are unbounded above as vol grows, so only the lower bound applies.
    del upper
    vol = torch.where(at_intrinsic, torch.zeros_like(vol), vol)
    vol = torch.where(price < intrinsic - 1e-8, torch.full_like(vol, float("nan")), vol)
    return vol
