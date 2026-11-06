"""Ikeda-Kunitomo double-barrier European options."""

from __future__ import annotations

import torch

from .black_scholes import price as bs_price
from ._common import as_tensors, norm_cdf

_KINDS = ("knock_in", "knock_out")


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
    series: int = 5,
) -> torch.Tensor:
    """Double-barrier price. ``kind`` is ``knock_in`` or ``knock_out``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown double-barrier kind {kind!r}")
    spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div = as_tensors(
        spot, strike, barrier_lo, barrier_hi, expiry, rate, vol, div
    )
    std = vol * torch.sqrt(expiry)
    mu1 = 2.0 * (rate - div) / vol**2 + 1.0
    bsigma = (rate - div + 0.5 * vol**2) * expiry / std
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)

    def knock_out(call: bool) -> torch.Tensor:
        acc_spot = torch.zeros_like(spot)
        acc_strike = torch.zeros_like(spot)
        for n in range(-series, series + 1):
            l2n = barrier_lo ** (2 * n)
            u2n = barrier_hi ** (2 * n)
            if call:
                d1 = torch.log(spot * u2n / (strike * l2n)) / std + bsigma
                d2 = torch.log(spot * u2n / (barrier_hi * l2n)) / std + bsigma
                d3 = torch.log(barrier_lo ** (2 * n + 2) / (strike * spot * u2n)) / std + bsigma
                d4 = torch.log(barrier_lo ** (2 * n + 2) / (barrier_hi * spot * u2n)) / std + bsigma
                w_spot = (barrier_hi**n / barrier_lo**n) ** mu1
                w_refl = (barrier_lo ** (n + 1) / (barrier_hi**n * spot)) ** mu1
                acc_spot = acc_spot + w_spot * (norm_cdf(d1) - norm_cdf(d2)) - w_refl * (
                    norm_cdf(d3) - norm_cdf(d4)
                )
                w_spot_k = (barrier_hi**n / barrier_lo**n) ** (mu1 - 2.0)
                w_refl_k = (barrier_lo ** (n + 1) / (barrier_hi**n * spot)) ** (mu1 - 2.0)
                acc_strike = acc_strike + w_spot_k * (
                    norm_cdf(d1 - std) - norm_cdf(d2 - std)
                ) - w_refl_k * (norm_cdf(d3 - std) - norm_cdf(d4 - std))
            else:
                y1 = torch.log(spot * u2n / (barrier_lo ** (2 * n + 1))) / std + bsigma
                y2 = torch.log(spot * u2n / (strike * l2n)) / std + bsigma
                y3 = torch.log(barrier_lo ** (2 * n + 2) / (barrier_lo * spot * u2n)) / std + bsigma
                y4 = torch.log(barrier_lo ** (2 * n + 2) / (strike * spot * u2n)) / std + bsigma
                w_k = (barrier_hi**n / barrier_lo**n) ** (mu1 - 2.0)
                w_refl_k = (barrier_lo ** (n + 1) / (barrier_hi**n * spot)) ** (mu1 - 2.0)
                acc_strike = acc_strike + w_k * (
                    norm_cdf(y1 - std) - norm_cdf(y2 - std)
                ) - w_refl_k * (norm_cdf(y3 - std) - norm_cdf(y4 - std))
                w_s = (barrier_hi**n / barrier_lo**n) ** mu1
                w_refl = (barrier_lo ** (n + 1) / (barrier_hi**n * spot)) ** mu1
                acc_spot = acc_spot + w_s * (norm_cdf(y1) - norm_cdf(y2)) - w_refl * (
                    norm_cdf(y3) - norm_cdf(y4)
                )
        if call:
            return torch.clamp(spot * div_df * acc_spot - strike * rate_df * acc_strike, min=0.0)
        return torch.clamp(strike * rate_df * acc_strike - spot * div_df * acc_spot, min=0.0)

    call_ko = knock_out(True)
    put_ko = knock_out(False)
    vanilla = bs_price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=is_call
    )
    ko = torch.where(is_call, call_ko, put_ko)
    if kind == "knock_out":
        return ko
    return torch.clamp(vanilla - ko, min=0.0)
