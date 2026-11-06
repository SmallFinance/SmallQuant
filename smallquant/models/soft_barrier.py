"""Soft-barrier European options."""

from __future__ import annotations

import torch

from .barrier import price as hard_barrier
from .black_scholes import price as bs_price
from ._common import as_tensors, norm_cdf

_KINDS = ("down_in", "up_in", "down_out", "up_out")


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier_lo: torch.Tensor,
    barrier_hi: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Soft barrier between ``barrier_lo`` and ``barrier_hi``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown barrier kind {kind!r}")
    spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div = as_tensors(
        spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div
    )
    rate = torch.where((rate - div).abs() < 1e-10, div + 1e-6, rate)
    eta = 1.0 if is_call.dtype == torch.bool else 1.0
    call_knock = _knock_in(spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div, 1.0)
    put_knock = _knock_in(spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div, -1.0)
    knock = torch.where(is_call, call_knock, put_knock)
    vanilla = bs_price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=is_call
    )
    soft = knock if kind in ("down_in", "up_in") else vanilla - knock
    single = torch.isclose(barrier_hi, barrier_lo, atol=1e-4, rtol=0.0)
    hard = hard_barrier(
        spot=spot,
        strike=strike,
        barrier=barrier_hi,
        expiry=expiry,
        rate=rate,
        vol=vol,
        div=div,
        rebate=torch.zeros_like(spot),
        is_call=is_call,
        kind=kind,
    )
    knocked_in = ((kind == "down_in") & (spot <= barrier_lo)) | (
        (kind == "up_in") & (spot >= barrier_hi)
    )
    knocked_out = ((kind == "down_out") & (spot <= barrier_lo)) | (
        (kind == "up_out") & (spot >= barrier_hi)
    )
    value = torch.where(single, hard, soft)
    value = torch.where(knocked_in, vanilla, value)
    return torch.where(knocked_out, torch.zeros_like(value), value)


def _knock_in(spot, strike, lower, upper, expiry, rate, vol, div, eta: float):
    carry = rate - div
    mu = (carry + 0.5 * vol * vol) / vol**2
    sqrt_t = torch.sqrt(expiry)
    sx = spot * strike
    log_u = torch.log(upper * upper / sx)
    log_l = torch.log(lower * lower / sx)
    lam1 = torch.exp(-0.5 * vol * vol * expiry * (mu + 0.5) * (mu - 0.5))
    lam2 = torch.exp(-0.5 * vol * vol * expiry * (mu - 0.5) * (mu - 1.5))
    d1 = log_u / (vol * sqrt_t) + mu * vol * sqrt_t
    d2 = d1 - (mu + 0.5) * vol * sqrt_t
    d3 = log_u / (vol * sqrt_t) + (mu - 1.0) * vol * sqrt_t
    d4 = d3 - (mu - 0.5) * vol * sqrt_t
    e1 = log_l / (vol * sqrt_t) + mu * vol * sqrt_t
    e2 = e1 - (mu + 0.5) * vol * sqrt_t
    e3 = log_l / (vol * sqrt_t) + (mu - 1.0) * vol * sqrt_t
    e4 = e3 - (mu - 0.5) * vol * sqrt_t
    term1 = (
        eta
        * spot
        * torch.exp((carry - rate) * expiry)
        * spot ** (-2.0 * mu)
        * sx ** (mu + 0.5)
        / (2.0 * (mu + 0.5))
    )
    term1 = term1 * (
        (upper * upper / sx) ** (mu + 0.5) * norm_cdf(eta * d1)
        - lam1 * norm_cdf(eta * d2)
        - (lower * lower / sx) ** (mu + 0.5) * norm_cdf(eta * e1)
        + lam1 * norm_cdf(eta * e2)
    )
    term2 = (
        eta
        * strike
        * torch.exp(-rate * expiry)
        * spot ** (-2.0 * (mu - 1.0))
        * sx ** (mu - 0.5)
        / (2.0 * (mu - 0.5))
    )
    term2 = term2 * (
        (upper * upper / sx) ** (mu - 0.5) * norm_cdf(eta * d3)
        - lam2 * norm_cdf(eta * d4)
        - (lower * lower / sx) ** (mu - 0.5) * norm_cdf(eta * e3)
        + lam2 * norm_cdf(eta * e4)
    )
    return (term1 - term2) / (upper - lower)
