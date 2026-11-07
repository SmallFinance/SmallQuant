"""Undiscounted path payoffs. The Monte Carlo engine applies the discount."""

from __future__ import annotations

import warnings

import torch

from smallquant._tensor import validation_enabled


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
    if smoothing is None and validation_enabled():
        warnings.warn(
            "pathwise delta of a cash-or-nothing is zero; select smoothing or a likelihood-ratio weight",
            UserWarning,
            stacklevel=2,
        )
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


def barrier_survival(
    paths: torch.Tensor,
    times: torch.Tensor,
    *,
    strike: torch.Tensor,
    is_call: torch.Tensor,
    barrier: torch.Tensor,
    vol: torch.Tensor,
    kind: str = "down_and_out",
) -> torch.Tensor:
    """One-step survival payoff for a single barrier (Glasserman 2004, §7.2).

    The knock indicator on each interval is replaced by the Brownian-bridge
    hitting probability, which is differentiable in the path. ``kind`` is
    ``down_and_out`` or ``up_and_out``. A discrete-monitoring shift is not
    applied; pass a shifted barrier if you want the Broadie-Glasserman-Kou
    correction ``barrier * exp(±0.5826 * vol * sqrt(dt))``.
    """
    if kind not in ("down_and_out", "up_and_out"):
        raise ValueError("survival payoff supports down_and_out and up_and_out")
    spot = paths[..., 0]
    left = spot[..., :-1]
    right = spot[..., 1:]
    dt = times[1:] - times[:-1]
    variance = (vol * vol) * dt
    if kind == "down_and_out":
        inside = (left > barrier) & (right > barrier)
        gap = torch.log(left / barrier) * torch.log(right / barrier)
    else:
        inside = (left < barrier) & (right < barrier)
        gap = torch.log(barrier / left) * torch.log(barrier / right)
    hit = torch.exp(-2.0 * gap / variance.clamp(min=1e-18))
    survive = torch.where(inside, (1.0 - hit).clamp(0.0, 1.0), torch.zeros_like(hit))
    vanilla = _vanilla(spot[..., -1], strike, is_call)
    return vanilla * survive.prod(dim=-1)


def _vanilla(level: torch.Tensor, strike: torch.Tensor, is_call: torch.Tensor) -> torch.Tensor:
    strike = torch.as_tensor(strike, dtype=level.dtype, device=level.device)
    call = torch.relu(level - strike)
    put = torch.relu(strike - level)
    return torch.where(is_call, call, put)
