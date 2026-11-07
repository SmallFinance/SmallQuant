"""Linear-algebra helpers with no finance in them."""

from __future__ import annotations

import torch


def cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Lower-triangular Cholesky factor of a symmetric positive-definite matrix."""
    return torch.linalg.cholesky(matrix)


def apply(factor: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    """Apply a lower-triangular factor to standard normal draws.

    ``draws`` ends in the dimension being correlated. The result has the same
    shape.
    """
    return draws @ factor.transpose(-1, -2)
