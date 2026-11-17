"""Characteristic-function pricing: COS and Carr-Madan.

A characteristic function here is a callable ``characteristic(u)`` returning
``E[exp(i u log S_T)]``. ``u`` is a complex tensor whose last dimension is the
integration grid; the function's own parameters must broadcast against it, which
``black_characteristic`` and ``heston_characteristic`` arrange by giving every
parameter a trailing singleton dimension.
"""

from __future__ import annotations

import math

import torch

from smallquant._tensor import align
from smallquant.models.heston import characteristic_function as heston_cf


def price(
    *,
    spot,
    strike,
    expiry,
    rate,
    div,
    is_call,
    characteristic,
    method="cos",
    num_terms=128,
    width=12.0,
    dtype=None,
):
    """European price from a characteristic function of ``log(spot)``.

    ``method`` is ``"cos"`` (Fang and Oosterlee 2008) or ``"carr_madan"`` (Carr and
    Madan 1999). Every input broadcasts, so one call prices a batch of strikes,
    expiries or model parameters.

    COS prices the put, whose payoff is bounded, and gets the call by parity. The put
    sees the left tail of ``log S_T``, so the truncation range reaches
    ``c1 - width * sqrt(c2 + sqrt(c4))`` on the left (Fang and Oosterlee's fat-tail
    rule) and ``c1 + width * sqrt(c2)`` on the right, from the cumulants of
    ``log S_T``. Wide distributions (high vol of vol, long expiry) need more terms:
    ``num_terms=1024`` covers vol of vol up to about 1.2 at five years.

    Carr-Madan integrates in forward moneyness ``log(K / F)``, so the integrand barely
    oscillates, with damping ``alpha = 0.75`` and a trapezoid rule of step 0.1 over
    ``num_terms`` points, so the upper limit is ``0.1 * num_terms``. Short expiries and
    high vol of vol decay slowly in frequency and need ``num_terms`` in the thousands.
    """
    if method not in ("cos", "carr_madan"):
        raise ValueError("method must be 'cos' or 'carr_madan'")
    spot, strike, expiry, rate, div = align(spot, strike, expiry, rate, div, dtype=dtype)
    spot, strike, expiry, rate, div = torch.broadcast_tensors(spot, strike, expiry, rate, div)
    forward_gap = spot * torch.exp(-div * expiry) - strike * torch.exp(-rate * expiry)
    if method == "cos":
        put = _cos_put(strike, expiry, rate, characteristic, num_terms, width)
        call = put + forward_gap
    else:
        forward = spot * torch.exp((rate - div) * expiry)
        call = _carr_madan_call(forward, strike, expiry, rate, characteristic, num_terms)
        put = call - forward_gap
    is_call = torch.as_tensor(is_call, device=call.device)
    return torch.where(is_call, call, put)


def _trailing(*values):
    """Give each parameter a trailing singleton so it broadcasts against the grid."""
    return tuple(value.unsqueeze(-1) for value in values)


def black_characteristic(*, spot, rate, div, vol, expiry):
    """Characteristic function of ``log(S_T)`` in Black-Scholes."""

    def characteristic(u):
        u = u.to(torch.complex128)
        s, r, q, v, t = _trailing(*align(spot, rate, div, vol, expiry))
        mu = (torch.log(s) + (r - q - 0.5 * v * v) * t).to(torch.complex128)
        var = (v * v * t).to(torch.complex128)
        return torch.exp(1j * u * mu - 0.5 * var * u * u)

    return characteristic


def heston_characteristic(*, spot, rate, div, expiry, v0, kappa, theta, sigma, rho):
    """Characteristic function of ``log(S_T)`` in Heston, broadcasting over a trailing grid."""

    def characteristic(u):
        s, r, q, t, v, k, th, sg, rh = _trailing(
            *align(spot, rate, div, expiry, v0, kappa, theta, sigma, rho)
        )
        return heston_cf(u, spot=s, rate=r, div=q, expiry=t, v0=v, kappa=k, theta=th, sigma=sg, rho=rh)

    return characteristic


def _cumulants(characteristic, like):
    """First, second and fourth cumulants of ``log S_T``, for the COS truncation range.

    Central differences of the cumulant generating function ``K(t) = log E[exp(t X)]``,
    evaluated as ``log characteristic(-i t)``. The range is a numerical choice, so it is
    computed without gradient.
    """
    step = 1e-2
    t = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64, device=like.device) * step
    with torch.no_grad():
        values = characteristic(-1j * t.to(torch.complex128))
        cgf = torch.log(values).real
        cgf = torch.broadcast_to(cgf, like.shape + (5,))
        m2, m1, zero, p1, p2 = cgf.unbind(-1)
        c1 = (p1 - m1) / (2.0 * step)
        c2 = (p1 - 2.0 * zero + m1) / step**2
        c4 = (p2 - 4.0 * p1 + 6.0 * zero - 4.0 * m1 + m2) / step**4
    return c1.to(like.dtype), c2.clamp(min=1e-8).to(like.dtype), c4.abs().to(like.dtype)


def _cos_put(strike, expiry, rate, characteristic, num_terms, width):
    c1, c2, c4 = _cumulants(characteristic, strike)
    left = width * torch.sqrt(c2 + torch.sqrt(c4))
    right = width * torch.sqrt(c2)
    a, b = (c1 - left).unsqueeze(-1), (c1 + right).unsqueeze(-1)
    k = torch.arange(num_terms, dtype=strike.dtype, device=strike.device)
    frequency = k * math.pi / (b - a)
    phi = characteristic(frequency.to(torch.complex128))
    shift = torch.exp(-1j * frequency.to(torch.complex128) * a.to(torch.complex128))
    strike_ = strike.unsqueeze(-1)
    upper = torch.clamp(torch.log(strike_), min=a, max=b)
    payoff = 2.0 / (b - a) * (strike_ * _psi(k, a, b, a, upper) - _chi(k, a, b, a, upper))
    weights = torch.where(k == 0, 0.5, 1.0)
    terms = (shift * phi).real.to(strike.dtype) * payoff * weights
    return torch.exp(-rate * expiry) * terms.sum(dim=-1)


def _chi(k, a, b, c, d):
    """Cosine coefficients of ``exp(y)`` on ``[c, d]`` within ``[a, b]``."""
    scale = k * math.pi / (b - a)
    numerator = (
        torch.cos(scale * (d - a)) * torch.exp(d)
        - torch.cos(scale * (c - a)) * torch.exp(c)
        + scale * (torch.sin(scale * (d - a)) * torch.exp(d) - torch.sin(scale * (c - a)) * torch.exp(c))
    )
    return numerator / (1.0 + scale**2)


def _psi(k, a, b, c, d):
    """Cosine coefficients of ``1`` on ``[c, d]`` within ``[a, b]``."""
    scale = k * math.pi / (b - a)
    safe = torch.where(k == 0, torch.ones_like(scale), scale)
    value = (torch.sin(safe * (d - a)) - torch.sin(safe * (c - a))) / safe
    return torch.where(k == 0, d - c, value)


def _carr_madan_call(forward, strike, expiry, rate, characteristic, num_terms):
    alpha = 0.75
    eta = 0.1
    grid = torch.arange(num_terms, dtype=strike.dtype, device=strike.device)
    v = eta * grid
    u = (v - 1j * (alpha + 1.0)).to(torch.complex128)
    log_forward = torch.log(forward).unsqueeze(-1).to(torch.complex128)
    # Characteristic function of log(S_T / F): removes the large log-level phase.
    phi = characteristic(u) * torch.exp(-1j * u * log_forward)
    denominator = (alpha * alpha + alpha - v * v + 1j * v * (2.0 * alpha + 1.0)).to(torch.complex128)
    moneyness = torch.log(strike / forward).unsqueeze(-1)
    # Trapezoid rule: the integrand's nearest poles sit alpha = 0.75 from the real axis,
    # so its error falls like exp(-2 pi alpha / eta), far below double precision here.
    trapezoid = torch.where(grid == 0, 0.5, 1.0)
    integrand = torch.exp(-1j * v.to(torch.complex128) * moneyness.to(torch.complex128)) * phi / denominator
    integral = (integrand.real.to(strike.dtype) * trapezoid).sum(dim=-1) * eta
    undiscounted = forward * torch.exp(-alpha * moneyness.squeeze(-1)) / math.pi * integral
    return torch.exp(-rate * expiry) * undiscounted
