"""Margrabe option to exchange one asset for another."""

from __future__ import annotations

import torch

from ._common import as_tensors, norm_cdf, norm_pdf


def price(
    *,
    spot1: torch.Tensor,
    spot2: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    div1: torch.Tensor,
    div2: torch.Tensor,
    correlation: torch.Tensor,
    quantity1: torch.Tensor | None = None,
    quantity2: torch.Tensor | None = None,
) -> torch.Tensor:
    """European option to exchange ``quantity2`` of asset 2 for ``quantity1`` of asset 1.

    Omitted quantities are one. An explicit quantity must be a tensor.
    """
    spot1, spot2, expiry, rate, vol1, vol2, div1, div2, correlation = as_tensors(
        spot1, spot2, expiry, rate, vol1, vol2, div1, div2, correlation
    )
    quantity1 = (
        torch.ones_like(spot1) if quantity1 is None else as_tensors(quantity1, dtype=spot1.dtype)[0]
    )
    quantity2 = (
        torch.ones_like(spot2) if quantity2 is None else as_tensors(quantity2, dtype=spot2.dtype)[0]
    )
    df = torch.exp(-rate * expiry)
    forward1 = spot1 * torch.exp((rate - div1) * expiry)
    forward2 = spot2 * torch.exp((rate - div2) * expiry)
    std1 = vol1 * torch.sqrt(expiry)
    std2 = vol2 * torch.sqrt(expiry)
    variance = std1**2 + std2**2 - 2.0 * correlation * std1 * std2
    std = torch.sqrt(variance)
    d1 = (torch.log((quantity1 * forward1) / (quantity2 * forward2)) + 0.5 * variance) / std
    d2 = d1 - std
    return df * (quantity1 * forward1 * norm_cdf(d1) - quantity2 * forward2 * norm_cdf(d2))
