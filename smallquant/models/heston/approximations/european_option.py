"""Heston European option price by a single Fourier integral (Attari 2004).

References:
    M. Attari, "Option Pricing Using Fourier Transforms: A Numerically Efficient
    Simplification" (2004), SSRN 520042.
    H. Albrecher, P. Mayer, W. Schoutens and J. Tistaert, "The Little Heston
    Trap" (2007), for the characteristic function branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from smallquant._tensor import align
from smallquant.black_scholes.vanilla_prices import _flag
from smallquant.math.quadrature import gauss_legendre
from smallquant.models.heston.heston_model import characteristic_function

__all__ = ["european_option_price"]

# Truncation: the integral stops where the tail bound falls below ``_TAIL``,
# searched on a log grid of ``_SCAN`` points on ``[_SCAN_LOW, _SCAN_HIGH]``.
_TAIL = 1e-15
_SCAN = 96
_SCAN_LOW = 1.0
_SCAN_HIGH = 1e5
# Quadrature: panels of ``_NODES`` Gauss-Legendre nodes. A quarter of the
# panels grade geometrically from ``_FIRST`` to ``_SPLIT``; the rest are equal.
_NODES = 32
_FIRST = 1e-3
_SPLIT = 16.0
_GEO_SHARE = 4


def european_option_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    div: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    is_call: torch.Tensor | bool,
    num_points: int = 1024,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """European call or put under Heston, by Attari's single Fourier integral.

    With ``F = spot * exp((rate - div) * expiry)``, ``l = log(strike / F)`` and
    ``phi`` the characteristic function of ``log(S_T / F)``, Attari (2004)
    writes the call as

        C = exp(-rate T) * (F - strike * (1/2 + I / pi)),
        I = int_0^inf [(Re phi + Im phi / w) cos(w l)
                       + (Im phi - Re phi / w) sin(w l)] / (1 + w^2) dw.

    The integrand decays like ``1 / w^2`` even before ``phi`` decays. Puts come
    from put-call parity.

    ``phi`` is :func:`smallquant.models.heston.characteristic_function` with
    unit spot and zero carry, which is the little-trap branch and stays
    continuous at long expiries.

    Quadrature. The integral is truncated per batch element at ``w_max``, the
    smallest point of a log grid on ``[1, 1e5]`` beyond which the tail bound
    ``|phi(w)| (1 + 1/w) / w`` stays below ``1e-15``. Short expiries, where
    ``phi`` decays slowly, get a wide interval and long expiries a narrow one.
    ``w_max`` is chosen without gradient; the boundary term it drops is below
    the tolerance. ``[0, w_max]`` is split into ``num_points // 32`` panels of
    32 Gauss-Legendre nodes (:func:`smallquant.math.quadrature.gauss_legendre`):
    ``[0, 1e-3]``, a quarter of the panels graded geometrically up to
    ``min(w_max, 16)``, and equal panels from there to ``w_max``. The graded
    panels resolve the integrand where ``phi`` has singularities close to the
    real axis near the origin (long expiries with a large ``sigma``); the equal
    panels resolve the ``cos(w l)`` oscillation in the tail. The panel and node
    axes are appended after the broadcast batch shape, so memory grows as
    ``batch * num_points``.

    With the default ``num_points=1024`` the prices agree with QuantLib's
    adaptive ``AnalyticHestonEngine`` within ``1e-11`` absolute on a spot of
    100, from ``expiry=0.05`` to ``10`` and for ``sigma`` up to 1.5. Inputs
    whose ``phi`` decays slowly over a wide range (a tiny ``v0 * expiry`` with a
    very large ``sigma``) need a larger ``num_points``, and lose accuracy once
    ``phi`` has not decayed by ``w = 1e5``.

    All numeric arguments are tensors that broadcast together; ``is_call`` is a
    bool or bool tensor. The result has the broadcast shape and is
    differentiable in every floating input. ``expiry <= 0`` returns the
    intrinsic value.
    """
    if num_points < 4 * _NODES:
        raise ValueError(f"num_points must be at least {4 * _NODES}")
    spot, strike, expiry, rate, div, v0, kappa, theta, sigma, rho = align(
        spot, strike, expiry, rate, div, v0, kappa, theta, sigma, rho, dtype=dtype
    )
    is_call = _flag(is_call, spot)
    shape = torch.broadcast_shapes(
        spot.shape, strike.shape, expiry.shape, rate.shape, div.shape,
        v0.shape, kappa.shape, theta.shape, sigma.shape, rho.shape, is_call.shape,
    )

    live = expiry > 0
    safe_expiry = torch.where(live, expiry, torch.ones_like(expiry))
    forward = spot * torch.exp((rate - div) * safe_expiry)
    discount = torch.exp(-rate * safe_expiry)
    log_moneyness = torch.log(strike / forward)

    # Heston parameters with two trailing axes for the panel and node grid.
    params = [
        value.expand(shape).reshape(*shape, 1, 1)
        for value in (safe_expiry, v0, kappa, theta, sigma, rho)
    ]

    def phi(w: torch.Tensor) -> torch.Tensor:
        # Characteristic function of log(S_T / F): unit spot, zero carry.
        expiry_, v0_, kappa_, theta_, sigma_, rho_ = params
        return characteristic_function(
            w,
            spot=torch.ones_like(expiry_),
            rate=torch.zeros_like(expiry_),
            div=torch.zeros_like(expiry_),
            expiry=expiry_,
            v0=v0_,
            kappa=kappa_,
            theta=theta_,
            sigma=sigma_,
            rho=rho_,
        )

    lower, upper = _panels(_cutoff(phi, spot), num_points // _NODES)
    moneyness = log_moneyness.expand(shape).reshape(*shape, 1, 1)

    def integrand(w: torch.Tensor) -> torch.Tensor:
        value = phi(w)
        re = value.real.to(spot.dtype)
        im = value.imag.to(spot.dtype)
        angle = w * moneyness
        return (
            (re + im / w) * torch.cos(angle) + (im - re / w) * torch.sin(angle)
        ) / (1.0 + w * w)

    integral = gauss_legendre(integrand, lower, upper, _NODES).sum(dim=-1)
    call = discount * (forward - strike * (0.5 + integral / math.pi))
    put = call - discount * (forward - strike)
    price = torch.where(is_call, call, put)

    intrinsic = torch.where(is_call, spot - strike, strike - spot).clamp(min=0.0)
    return torch.where(live, price, intrinsic)


def _cutoff(phi: Callable[[torch.Tensor], torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    """Truncation point of the Attari integral, as a constant per batch element."""
    with torch.no_grad():
        grid = torch.logspace(
            math.log10(_SCAN_LOW), math.log10(_SCAN_HIGH), _SCAN, dtype=like.dtype, device=like.device
        )
        # |integrand| * (1 + w^2) <= |phi| * (1 + 1/w) * sqrt(2), so the tail
        # beyond w is at most about sqrt(2) |phi(w)| (1 + 1/w) / w.
        modulus = phi(grid.unsqueeze(0)).abs().to(like.dtype)[..., 0, :]
        bound = modulus * (1.0 + 1.0 / grid) / grid
        above = bound >= _TAIL
        index = torch.arange(_SCAN, device=like.device)
        last = torch.where(above, index, torch.full_like(index, -1)).amax(dim=-1)
        pick = (last + 1).clamp(max=_SCAN - 1)
        return grid[pick]


def _panels(cutoff: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper panel edges on ``[0, cutoff]``, panels on a new last axis.

    ``[0, _FIRST]``, then geometric panels up to ``min(cutoff, _SPLIT)``, which
    resolve singularities of the integrand close to the origin, then equal
    panels up to ``cutoff`` for the oscillating tail.
    """
    near = max(count // _GEO_SHARE, 2)
    far = count - near
    split = cutoff.clamp(max=_SPLIT).unsqueeze(-1)
    end = cutoff.unsqueeze(-1)
    steps = torch.arange(near, dtype=cutoff.dtype, device=cutoff.device) / (near - 1)
    geometric = _FIRST * (split / _FIRST) ** steps
    steps = torch.arange(1, far + 1, dtype=cutoff.dtype, device=cutoff.device) / far
    uniform = split + (end - split) * steps
    edges = torch.cat((torch.zeros_like(split), geometric, uniform), dim=-1)
    return edges[..., :-1], edges[..., 1:]
