"""Batched tridiagonal solves."""

from __future__ import annotations

import torch


def solve(lower: torch.Tensor, diag: torch.Tensor, upper: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``lower_i x_{i-1} + diag_i x_i + upper_i x_{i+1} = rhs_i``.

    The arguments share a trailing length ``n``. ``lower[..., 0]`` and
    ``upper[..., -1]`` are ignored. The Thomas algorithm is stable for the
    diagonally dominant matrices produced by implicit Black-Scholes steps.
    """
    if lower.shape != diag.shape or upper.shape != diag.shape or rhs.shape != diag.shape:
        raise ValueError("tridiagonal arguments must share a shape")
    size = diag.shape[-1]
    if size < 1:
        raise ValueError("tridiagonal system must be non-empty")
    codiagonal: list[torch.Tensor] = []
    updated = []
    pivot = diag[..., 0]
    codiagonal.append(upper[..., 0] / pivot)
    updated.append(rhs[..., 0] / pivot)
    for index in range(1, size):
        pivot = diag[..., index] - lower[..., index] * codiagonal[index - 1]
        updated.append((rhs[..., index] - lower[..., index] * updated[index - 1]) / pivot)
        if index < size - 1:
            codiagonal.append(upper[..., index] / pivot)
    solution = [updated[-1]]
    for index in range(size - 2, -1, -1):
        solution.append(updated[index] - codiagonal[index] * solution[-1])
    solution.reverse()
    return torch.stack(solution, dim=-1)
