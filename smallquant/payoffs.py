"""Undiscounted path payoffs. The Monte Carlo engine applies the discount."""

from __future__ import annotations

import torch


def european(paths: torch.Tensor, times: torch.Tensor, *, strike: torch.Tensor, is_call: torch.Tensor) -> torch.Tensor:
    """Vanilla payoff on the spot at the last time."""
    del times
    return _vanilla(paths[..., -1, 0], strike, is_call)


def arithmetic_average_price(
    paths: torch.Tensor, times: torch.Tensor, *, strike: torch.Tensor, is_call: torch.Tensor
) -> torch.Tensor:
    """Arithmetic average of the fixings after time 0."""
    del times
    return _vanilla(paths[..., 1:, 0].mean(dim=-1), strike, is_call)


def geometric_average_price(
    paths: torch.Tensor, times: torch.Tensor, *, strike: torch.Tensor, is_call: torch.Tensor
) -> torch.Tensor:
    """Geometric average of the fixings after time 0."""
    del times
    fixings = paths[..., 1:, 0]
    average = torch.exp(torch.log(fixings).mean(dim=-1))
    return _vanilla(average, strike, is_call)


def cash_or_nothing(
    paths: torch.Tensor,
    times: torch.Tensor,
    *,
    strike: torch.Tensor,
    is_call: torch.Tensor,
    cash: torch.Tensor,
    smoothing: str | None = None,
    eps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cash-or-nothing on the terminal spot.

    ``smoothing="sigmoid"`` replaces the indicator with a logistic of width
    ``eps``, so a pathwise delta is no longer identically zero.
    """
    del times
    terminal = paths[..., -1, 0]
    strike = torch.as_tensor(strike, dtype=terminal.dtype, device=terminal.device)
    cash = torch.as_tensor(cash, dtype=terminal.dtype, device=terminal.device)
    moneyness = torch.where(is_call, terminal - strike, strike - terminal)
    if smoothing is None:
        indicator = (moneyness > 0).to(dtype=terminal.dtype)
    elif smoothing == "sigmoid":
        if eps is None:
            raise ValueError("sigmoid smoothing requires eps")
        width = torch.as_tensor(eps, dtype=terminal.dtype, device=terminal.device)
        indicator = torch.sigmoid(moneyness / width)
    else:
        raise ValueError(f"unknown smoothing {smoothing!r}")
    return cash * indicator


def _vanilla(level: torch.Tensor, strike: torch.Tensor, is_call: torch.Tensor) -> torch.Tensor:
    strike = torch.as_tensor(strike, dtype=level.dtype, device=level.device)
    call = torch.relu(level - strike)
    put = torch.relu(strike - level)
    return torch.where(is_call, call, put)
