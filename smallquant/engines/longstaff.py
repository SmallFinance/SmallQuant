"""Longstaff-Schwartz early exercise with a frozen regression boundary."""

from __future__ import annotations

import torch


def exercise_index(paths: torch.Tensor, times: torch.Tensor, intrinsic, rate: torch.Tensor, degree: int = 2):
    """Exercise time index per path. The regression is under ``no_grad``.

    The basis is monomials ``1, x, ..., x^degree`` in the spot, fitted on the
    same in-the-money paths. The boundary is a boolean policy, so Greeks of a
    later discounted exercise value do not move the boundary (envelope theorem).
    """
    spot = paths[..., 0]
    num_paths, num_times = spot.shape
    with torch.no_grad():
        cashflow = intrinsic(spot[:, -1])
        when = torch.full((num_paths,), num_times - 1, dtype=torch.long, device=spot.device)
        for index in range(num_times - 2, 0, -1):
            exercise_value = intrinsic(spot[:, index])
            in_the_money = exercise_value > 0
            if not bool(in_the_money.any()):
                continue
            dt = times[index + 1] - times[index]
            continuation_target = cashflow[in_the_money] * torch.exp(-rate * dt)
            basis = _basis(spot[in_the_money, index], degree)
            coefficient = torch.linalg.lstsq(basis, continuation_target.unsqueeze(-1)).solution.squeeze(-1)
            continuation = basis @ coefficient
            take = exercise_value[in_the_money] > continuation
            chosen = in_the_money.clone()
            chosen[in_the_money] = take
            cashflow = torch.where(chosen, exercise_value, cashflow * torch.exp(-rate * dt))
            when = torch.where(chosen, torch.full_like(when, index), when)
        # Paths that are never exercised early keep the terminal index.
        del cashflow
    return when


def discounted_exercise(paths, times, intrinsic, rate, when):
    """Discounted intrinsic value at the frozen exercise index."""
    spot = paths[..., 0]
    num_paths = spot.shape[0]
    gathered = spot[torch.arange(num_paths, device=spot.device), when]
    value = intrinsic(gathered)
    exercise_time = times[when]
    return value * torch.exp(-rate * exercise_time)


def _basis(spot: torch.Tensor, degree: int) -> torch.Tensor:
    columns = [torch.ones_like(spot)]
    power = spot
    for _ in range(degree):
        columns.append(power)
        power = power * spot
    return torch.stack(columns, dim=-1)
