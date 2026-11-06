"""Double-barrier cash-or-nothing options."""

from __future__ import annotations

import math

import torch

from ._common import as_tensors

_KINDS = ("knock_in", "knock_out", "kiko", "koki")


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
    cash: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Double-barrier cash-or-nothing price.

    ``kind`` is ``knock_in``, ``knock_out``, ``kiko`` or ``koki``. The strike is
    only used to read the variance anchor, matching QuantLib.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown double-barrier kind {kind!r}")
    spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div, cash = as_tensors(
        spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div, cash
    )
    variance = vol**2 * expiry
    if kind in ("knock_in", "knock_out"):
        return _series(spot, barrier_lo, barrier_hi, expiry, rate, div, cash, variance, kind, 99)
    return _kiko(spot, barrier_lo, barrier_hi, expiry, rate, div, cash, variance, kind, 999)


def _series(spot, lower, upper, expiry, rate, div, cash, variance, kind, terms):
    sigma2 = variance / expiry
    carry = rate - div
    alpha = -0.5 * (2.0 * carry / sigma2 - 1.0)
    beta = -0.25 * (2.0 * carry / sigma2 - 1.0) ** 2 - 2.0 * rate / sigma2
    width = torch.log(upper / lower)
    factor = (2.0 * math.pi * cash) / width**2
    total = torch.zeros_like(spot)
    for i in range(1, terms + 1):
        term1 = ((spot / lower) ** alpha - ((-1.0) ** i) * (spot / upper) ** alpha) / (
            alpha**2 + (i * math.pi / width) ** 2
        )
        term2 = torch.sin(i * math.pi / width * torch.log(spot / lower))
        term3 = torch.exp(-0.5 * ((i * math.pi / width) ** 2 - beta) * variance)
        total = total + factor * i * term1 * term2 * term3
    if kind == "knock_out":
        return torch.clamp(total, min=0.0)
    return torch.clamp(cash * torch.exp(-rate * expiry) - total, min=0.0)


def _kiko(spot, lower, upper, expiry, rate, div, cash, variance, kind, terms):
    if kind == "koki":
        lower, upper = upper, lower
    sigma2 = variance / expiry
    carry = rate - div
    alpha = -0.5 * (2.0 * carry / sigma2 - 1.0)
    beta = -0.25 * (2.0 * carry / sigma2 - 1.0) ** 2 - 2.0 * rate / sigma2
    width = torch.log(upper / lower)
    log_s = torch.log(spot / lower)
    total = torch.zeros_like(spot)
    for i in range(1, terms + 1):
        factor = (i * math.pi / width) ** 2 - beta
        term1 = (beta - (i * math.pi / width) ** 2 * torch.exp(-0.5 * factor * variance)) / factor
        term2 = torch.sin(i * math.pi / width * log_s)
        total = total + (2.0 / (i * math.pi)) * term1 * term2
    total = total + 1.0 - log_s / width
    return torch.clamp(total * cash * (spot / lower) ** alpha, min=0.0)
