"""Market data passed to an engine. The core still prices in year fractions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Market:
    """Spot, a flat continuous rate, a dividend yield, or a discount curve.

    ``curve`` supplies ``discount(t)`` and ``zero_rate(t)`` when the flat
    ``rate`` is omitted. ``div`` is a continuous dividend yield.
    """

    spot: torch.Tensor
    rate: torch.Tensor | None = None
    div: torch.Tensor | None = None
    discount: torch.Tensor | None = None
    curve: object | None = None

    def rate_at(self, expiry: torch.Tensor) -> torch.Tensor:
        if self.curve is not None:
            return self.curve.zero_rate(expiry)
        if self.rate is None:
            raise ValueError("market needs a rate or a curve")
        return self.rate

    def div_at(self, expiry: torch.Tensor) -> torch.Tensor:
        if self.div is None:
            return torch.zeros((), dtype=expiry.dtype, device=expiry.device)
        return self.div

    def discount_at(self, expiry: torch.Tensor) -> torch.Tensor:
        if self.discount is not None:
            return self.discount
        if self.curve is not None:
            return self.curve.discount(expiry)
        return torch.exp(-self.rate_at(expiry) * expiry)
