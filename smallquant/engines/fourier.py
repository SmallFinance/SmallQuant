"""Characteristic-function pricing: COS and Carr-Madan."""

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

    ``characteristic(u)`` returns ``E[exp(i u log S_T)]``. ``method`` is
    ``"cos"`` (Fang and Oosterlee 2008) or ``"carr_madan"``.
    """
    if method not in ("cos", "carr_madan"):
        raise ValueError("method must be 'cos' or 'carr_madan'")
    spot, strike, expiry, rate, div = align(spot, strike, expiry, rate, div, dtype=dtype)
    if method == "cos":
        call = _cos(spot, strike, expiry, rate, div, characteristic, num_terms, width)
    else:
        call = _carr_madan(spot, strike, expiry, rate, characteristic, num_terms)
    forward_gap = spot * torch.exp(-div * expiry) - strike * torch.exp(-rate * expiry)
    put = call - forward_gap
    return torch.where(is_call, call, put)


def black_characteristic(*, spot, rate, div, vol, expiry):
    """Characteristic function of ``log(S_T)`` in Black-Scholes."""

    def characteristic(u):
        if not torch.is_complex(u):
            u = u.to(torch.complex128)
        mu = torch.log(spot) + (rate - div - 0.5 * vol * vol) * expiry
        var = vol * vol * expiry
        mu = mu.to(torch.complex128)
        var = var.to(torch.complex128)
        return torch.exp(1j * u * mu - 0.5 * var * u * u)

    return characteristic


def heston_characteristic(*, spot, rate, div, expiry, v0, kappa, theta, sigma, rho):
    def characteristic(u):
        return heston_cf(
            u, spot=spot, rate=rate, div=div, expiry=expiry, v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho
        )

    return characteristic


def _cos(spot, strike, expiry, rate, div, characteristic, num_terms, width):
    # Truncate log(S) around the Black-style mean when the caller did not pass cumulants.
    mean, variance = _cumulants(characteristic)
    half = width * torch.sqrt(variance.clamp(min=1e-8))
    a = mean - half
    b = mean + half
    k = torch.arange(num_terms, dtype=spot.dtype, device=spot.device)
    interval = b - a
    frequency = k * math.pi / interval
    phi = characteristic(frequency.to(torch.complex128))
    shift = torch.exp(-1j * frequency.to(torch.complex128) * a.to(torch.complex128))
    log_strike = torch.log(strike)
    lower = torch.maximum(log_strike, a)
    vk = 2.0 / interval * (_chi(k, a, b, lower, b) - strike * _psi(k, a, b, lower, b))
    weights = torch.ones_like(k)
    weights[0] = 0.5
    terms = (shift * phi).real * vk * weights
    return torch.exp(-rate * expiry) * terms.sum(dim=-1)


def _cumulants(characteristic):
    """First two cumulants of log-spot from a centred difference of the characteristic function."""
    step = 1e-4
    plus = characteristic(torch.tensor(step, dtype=torch.float64))
    minus = characteristic(torch.tensor(-step, dtype=torch.float64))
    origin = characteristic(torch.tensor(0.0, dtype=torch.float64))
    first = (plus - minus) / (2.0 * step)
    mean = (first / 1j).real
    second = (plus - 2.0 * origin + minus) / step**2
    variance = (-second).real - mean**2
    return mean.to(dtype=torch.float64), variance.to(dtype=torch.float64).clamp(min=1e-8)


def _chi(k, a, b, c, d):
    scale = k * math.pi / (b - a)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    one = torch.ones_like(scale)
    numerator = (
        torch.cos(safe * (d - a)) * torch.exp(d)
        - torch.cos(safe * (c - a)) * torch.exp(c)
        + safe * (torch.sin(safe * (d - a)) * torch.exp(d) - torch.sin(safe * (c - a)) * torch.exp(c))
    )
    value = numerator / (1.0 + safe**2)
    zero = torch.exp(d) - torch.exp(c)
    return torch.where(k == 0, zero, value)


def _psi(k, a, b, c, d):
    scale = k * math.pi / (b - a)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    value = (torch.sin(safe * (d - a)) - torch.sin(safe * (c - a))) / safe
    return torch.where(k == 0, d - c, value)


def _carr_madan(spot, strike, expiry, rate, characteristic, num_terms):
    alpha = 0.75
    eta = 0.25
    grid = torch.arange(num_terms, dtype=spot.dtype, device=spot.device)
    v = eta * grid
    damp = alpha + 1.0
    u = (v - 1j * damp).to(torch.complex128)
    phi = characteristic(u)
    denom = alpha * alpha + alpha - v**2 + 1j * v * (2 * alpha + 1)
    psi = torch.exp(torch.tensor(-rate * expiry, dtype=torch.complex128, device=spot.device)) * phi / denom.to(
        torch.complex128
    )
    log_strike = torch.log(strike)
    simpson = 3.0 + torch.remainder(grid, 2.0)
    simpson[0] = 1.0
    integrand = torch.exp(-1j * v.to(torch.complex128) * log_strike.to(torch.complex128)) * psi * simpson
    integral = integrand.real.sum() * eta / 3.0
    return torch.exp(-alpha * log_strike) / math.pi * integral
