"""Discount and zero curves."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.interp import linear, log_linear


class DiscountCurve:
    """Discount factors on a time grid.

    ``times`` is strictly increasing and includes 0. ``interpolation`` is
    ``log_linear`` or ``linear``. Rates are continuously compounded.
    """

    def __init__(self, times: torch.Tensor, discount_factors: torch.Tensor, interpolation: str = "log_linear"):
        if interpolation not in ("log_linear", "linear"):
            raise ValueError("interpolation must be 'log_linear' or 'linear'")
        self.times, self.discount_factors = align(times, discount_factors)
        self.interpolation = interpolation

    def discount(self, t: torch.Tensor) -> torch.Tensor:
        t = align(t, self.times)[0]
        if self.interpolation == "log_linear":
            return log_linear(t, self.times, self.discount_factors.clamp(min=1e-18))
        return linear(t, self.times, self.discount_factors)

    def zero_rate(self, t: torch.Tensor) -> torch.Tensor:
        """Continuously compounded zero rate."""
        t = align(t, self.times)[0]
        df = self.discount(t)
        safe = torch.where(t <= 0, torch.ones_like(t), t)
        rate = -torch.log(df.clamp(min=1e-18)) / safe
        instant = self.forward_rate(torch.zeros_like(t), torch.full_like(t, 1e-4))
        return torch.where(t <= 0, instant, rate)

    def forward_rate(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Simply the continuous forward, ``log(P(t1)/P(t2)) / (t2-t1)``."""
        t1, t2 = align(t1, t2, self.times)[:2]
        return torch.log(self.discount(t1) / self.discount(t2).clamp(min=1e-18)) / (t2 - t1).clamp(min=1e-12)


class ZeroCurve(DiscountCurve):
    """Build a discount curve from continuously compounded zero rates."""

    def __init__(self, times: torch.Tensor, zero_rates: torch.Tensor, interpolation: str = "log_linear"):
        times, zero_rates = align(times, zero_rates)
        super().__init__(times, torch.exp(-zero_rates * times), interpolation)
