"""Shared choices for the implied volatility solvers."""

from __future__ import annotations

import enum

__all__ = ["UnderlyingDistribution"]


class UnderlyingDistribution(enum.Enum):
    """Which model's volatility to imply.

    ``LOGNORMAL`` is Black-Scholes / Black-76 volatility. ``NORMAL`` is Bachelier
    (absolute) volatility of the forward.
    """

    LOGNORMAL = 1
    NORMAL = 2
