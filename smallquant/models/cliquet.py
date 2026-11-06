"""Analytic cliquet and performance options on a reset schedule."""

from __future__ import annotations

import torch

from ._common import as_tensors, black_price


def price(
    *,
    spot: torch.Tensor,
    moneyness: torch.Tensor,
    reset_times: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Sum of forward-starting vanillas on ``reset_times``, ending at ``expiry``.

    The option has not started: there is no accrued coupon and no caps or floors.
    """
    spot, moneyness, expiry, rate, vol, div = as_tensors(spot, moneyness, expiry, rate, vol, div)
    resets = torch.as_tensor(reset_times, dtype=spot.dtype, device=spot.device)
    times = torch.cat([resets.reshape(-1), expiry.reshape(-1)])
    total = torch.zeros_like(spot)
    strike = spot * moneyness
    for i in range(1, times.numel()):
        start = times[i - 1]
        end = times[i]
        weight = torch.exp(-div * start)
        tau = end - start
        discount = torch.exp(-rate * tau)
        forward = spot * torch.exp(-div * tau) / discount
        total = total + weight * black_price(forward, strike, vol * torch.sqrt(tau), discount, is_call)
    return total


def performance_price(
    *,
    spot: torch.Tensor,
    moneyness: torch.Tensor,
    reset_times: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Performance option: each reset pays ``moneyness`` times a unit-strike forward option."""
    spot, moneyness, expiry, rate, vol, div = as_tensors(spot, moneyness, expiry, rate, vol, div)
    resets = torch.as_tensor(reset_times, dtype=spot.dtype, device=spot.device)
    times = torch.cat([resets.reshape(-1), expiry.reshape(-1)])
    total = torch.zeros_like(spot)
    unit = torch.ones_like(spot)
    for i in range(1, times.numel()):
        start = times[i - 1]
        end = times[i]
        tau = end - start
        period_df = torch.exp(-rate * tau)
        forward = (1.0 / moneyness) * torch.exp(-div * tau) / period_df
        black = black_price(forward, unit, vol * torch.sqrt(tau), period_df, is_call)
        total = total + torch.exp(-rate * start) * moneyness * black
    return total
