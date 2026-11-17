"""Geometric Brownian motion paths and the Black-Scholes likelihood-ratio weight."""

from __future__ import annotations

import torch

from smallquant._tensor import align


def step(
    *,
    spot: torch.Tensor,
    times: torch.Tensor,
    normals: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
) -> torch.Tensor:
    """Exact lognormal paths, shape ``[num_paths, num_steps + 1, dim]``.

    ``normals`` are standard normal Wiener increments on each step of
    ``times`` (the grid includes 0). Flat rates and volatility make this the
    exact transition, which is what QuantLib's Black-Scholes process applies
    on each grid interval.
    """
    spot, times, normals, rate, vol, div = align(spot, times, normals, rate, vol, div)
    num_paths, num_steps, dim = normals.shape
    interval = (times[1:] - times[:-1]).reshape(1, num_steps, 1)
    wiener = torch.cumsum(normals * torch.sqrt(interval), dim=1)
    elapsed = times[1:].reshape(1, num_steps, 1)
    exponent = (rate - div - 0.5 * vol * vol) * elapsed + vol * wiener
    future = spot * torch.exp(exponent)
    initial = torch.broadcast_to(spot, (num_paths, 1, dim)).clone()
    return torch.cat((initial, future), dim=1)


def likelihood_weight(
    *,
    normals: torch.Tensor,
    times: torch.Tensor,
    spot: torch.Tensor,
    vol: torch.Tensor,
) -> torch.Tensor:
    """Likelihood-ratio weight whose derivative is the Black-Scholes score.

    The weight equals one at the current spot and volatility, so a Monte Carlo
    price is unchanged. Autograd through the weight is the likelihood-ratio
    delta and vega (Glasserman 2004). ``normals`` are Wiener increments of
    shape ``[num_paths, num_steps, dim]``.
    """
    spot, vol, times = align(spot, vol, times)
    interval = times[1:] - times[:-1]
    elapsed = times[-1] - times[0]
    terminal = (normals[..., 0] * torch.sqrt(interval)).sum(dim=-1)
    z = terminal / torch.sqrt(elapsed)
    score_spot = z / (spot * vol * torch.sqrt(elapsed))
    score_vol = (z * z - 1.0) / vol - z * torch.sqrt(elapsed)
    return 1.0 + score_spot * (spot - spot.detach()) + score_vol * (vol - vol.detach())
