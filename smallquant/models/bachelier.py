"""Bachelier price of a European option on a forward."""

from __future__ import annotations

import torch

from ._common import as_tensors, bachelier_price


def price(
    *,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    normal_vol: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Normal-model price. ``normal_vol`` is the absolute volatility of the forward."""
    forward, strike, expiry, normal_vol, discount = as_tensors(
        forward, strike, expiry, normal_vol, discount
    )
    return bachelier_price(
        forward, strike, normal_vol * torch.sqrt(expiry), discount, is_call
    )


def step(
    *,
    forward: torch.Tensor,
    times: torch.Tensor,
    normals: torch.Tensor,
    normal_vol: torch.Tensor,
) -> torch.Tensor:
    """Exact normal paths of the forward, shape ``[num_paths, num_steps + 1, dim]``.

    ``normals`` are standard normal Wiener increments. The forward at each
    grid time is ``forward + normal_vol * W_t``.
    """
    forward, times, normals, normal_vol = as_tensors(forward, times, normals, normal_vol)
    num_paths, num_steps, dim = normals.shape
    interval = (times[1:] - times[:-1]).reshape(1, num_steps, 1)
    wiener = torch.cumsum(normals * torch.sqrt(interval), dim=1)
    future = forward + normal_vol * wiener
    initial = torch.broadcast_to(forward, (num_paths, 1, dim)).clone()
    return torch.cat((initial, future), dim=1)
