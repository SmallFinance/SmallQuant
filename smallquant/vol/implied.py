"""Black and Bachelier implied volatility with an implicit gradient."""

from __future__ import annotations

import math

import torch

from smallquant._tensor import align, check
from smallquant.math.distributions import normal_cdf, normal_pdf
from smallquant.models.black_scholes import _flag, price as black_scholes_price
from smallquant.models._common import bachelier_price, black_price


def implied_vol(
    price,
    *,
    spot,
    strike,
    expiry,
    rate,
    div,
    is_call=True,
    tol=1e-12,
    max_iter=50,
    dtype=None,
):
    """Black-Scholes implied volatility of a whole chain.

    The initial guess is Brenner-Subrahmanyam, then Newton safeguarded by
    bisection. Prices outside the no-arbitrage bounds are NaN. A price equal
    to intrinsic value returns zero volatility. The backward pass is
    ``d vol / d price = 1 / vega`` and does not unroll the solver.

    Example:
        >>> spot = torch.tensor(100.0, dtype=torch.float64)
        >>> vol = torch.tensor(0.2, dtype=torch.float64)
    """
    spot, strike, expiry, rate, div, price = align(
        spot, strike, expiry, rate, div, price, dtype=dtype
    )
    is_call = _flag(is_call, spot)
    check("spot", spot, spot > 0)
    check("strike", strike, strike > 0)
    check("expiry", expiry, expiry > 0)
    check("price", price, price >= 0)
    return _BlackImplied.apply(price, spot, strike, expiry, rate, div, is_call.to(dtype=price.dtype), tol, max_iter)


def implied_normal_vol(
    price,
    *,
    forward,
    strike,
    expiry,
    discount,
    is_call=True,
    tol=1e-10,
    max_iter=50,
    dtype=None,
):
    """Bachelier implied normal volatility. The gradient is ``1 / vega``."""
    forward, strike, expiry, discount, price = align(
        forward, strike, expiry, discount, price, dtype=dtype
    )
    is_call = _flag(is_call, forward)
    return _NormalImplied.apply(
        price, forward, strike, expiry, discount, is_call.to(dtype=price.dtype), tol, max_iter
    )


class _BlackImplied(torch.autograd.Function):
    @staticmethod
    def forward(ctx, price, spot, strike, expiry, rate, div, is_call, tol, max_iter):
        flag = is_call > 0.5
        with torch.no_grad():
            vol, active = _solve_black(price, spot, strike, expiry, rate, div, flag, float(tol), int(max_iter))
        ctx.save_for_backward(price, spot, strike, expiry, rate, div, is_call, vol)
        ctx.tol = float(tol)
        return vol

    @staticmethod
    def backward(ctx, grad_vol):
        price, spot, strike, expiry, rate, div, is_call, vol = ctx.saved_tensors
        flag = is_call > 0.5
        with torch.enable_grad():
            leaves = tuple(tensor.detach().requires_grad_(True) for tensor in (spot, strike, expiry, rate, vol, div))
            model = black_scholes_price(
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
    def forward(ctx, price, forward, strike, expiry, discount, is_call, tol, max_iter):
        flag = is_call > 0.5
        with torch.no_grad():
            vol = _solve_normal(price, forward, strike, expiry, discount, flag, float(tol), int(max_iter))
        ctx.save_for_backward(price, forward, strike, expiry, discount, is_call, vol)
        return vol

    @staticmethod
    def backward(ctx, grad_vol):
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


def _solve_black(price, spot, strike, expiry, rate, div, is_call, tol, max_iter):
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
    for _ in range(max_iter):
        model, vega = _black_and_vega(spot, strike, expiry_safe, rate, div, vol, is_call)
        too_high = model > price
        hi = torch.where(too_high, vol, hi)
        lo = torch.where(too_high, lo, vol)
        newton = vol - (model - price) / vega.clamp(min=1e-10)
        inside = (newton > lo) & (newton < hi) & torch.isfinite(newton)
        vol = torch.where(inside, newton, 0.5 * (lo + hi))
    vol = torch.where(at_intrinsic, torch.zeros_like(vol), vol)
    vol = torch.where(below | above, torch.full_like(vol, float("nan")), vol)
    return vol, ~(below | above | at_intrinsic)


def _black_and_vega(spot, strike, expiry, rate, div, vol, is_call):
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


def _solve_normal(price, forward, strike, expiry, discount, is_call, tol, max_iter):
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
    for _ in range(max_iter):
        model = bachelier_price(forward, strike, vol * std_scale, discount, is_call)
        # vega w.r.t. normal vol
        h = torch.where(is_call, forward - strike, strike - forward) / (vol * std_scale).clamp(min=1e-8)
        vega = discount * std_scale * normal_pdf(h)
        too_high = model > price
        hi = torch.where(too_high, vol, hi)
        lo = torch.where(too_high, lo, vol)
        newton = vol - (model - price) / vega.clamp(min=1e-10)
        inside = (newton > lo) & (newton < hi) & torch.isfinite(newton)
        vol = torch.where(inside, newton, 0.5 * (lo + hi))
    at_intrinsic = (price - intrinsic).abs() <= 1e-8
    upper = discount * torch.where(is_call, torch.relu(forward) + strike.abs(), strike.abs() + torch.relu(-forward))
    # Normal prices are unbounded above as vol grows, so only the lower bound applies.
    del upper
    vol = torch.where(at_intrinsic, torch.zeros_like(vol), vol)
    vol = torch.where(price < intrinsic - 1e-8, torch.full_like(vol, float("nan")), vol)
    return vol
