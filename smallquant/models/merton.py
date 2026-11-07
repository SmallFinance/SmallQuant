"""Merton (1976) jump-diffusion price and an exact Poisson step."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.models.black_scholes import price as black_price


def price(
    *,
    spot,
    strike,
    expiry,
    rate,
    vol,
    div,
    is_call,
    jump_intensity,
    jump_mean,
    jump_vol,
    num_terms: int = 64,
    dtype=None,
):
    """Poisson mixture of Black-Scholes prices.

    ``jump_mean`` and ``jump_vol`` are the normal parameters of the log jump.
    ``num_terms`` is the number of Poisson terms.
    """
    spot, strike, expiry, rate, vol, div, jump_intensity, jump_mean, jump_vol = align(
        spot, strike, expiry, rate, vol, div, jump_intensity, jump_mean, jump_vol, dtype=dtype
    )
    compensator = torch.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    intensity = jump_intensity * (1.0 + compensator)
    mean_count = intensity * expiry
    total = torch.zeros_like(spot)
    weight = torch.exp(-mean_count)
    for n in range(num_terms):
        sigma_n = torch.sqrt(vol**2 + n * jump_vol**2 / expiry.clamp(min=1e-12))
        rate_n = rate - jump_intensity * compensator + n * torch.log((1.0 + compensator).clamp(min=1e-18)) / expiry.clamp(
            min=1e-12
        )
        total = total + weight * black_price(
            spot=spot,
            strike=strike,
            expiry=expiry,
            rate=rate_n,
            vol=sigma_n,
            div=div,
            is_call=is_call,
        )
        weight = weight * mean_count / (n + 1)
    return total


def step(*, spot, times, normals, rate, div, vol, jump_intensity, jump_mean, jump_vol, jump_normals, counts=None):
    """Lognormal diffusion plus a compound Poisson jump on each interval.

    ``jump_normals`` is the normal deviate of the jump sum and has the same
    shape as ``normals``. The Poisson count is drawn with ``torch.poisson``
    and held fixed, so gradients flow through the jump mean and volatility but
    not through the discrete count. Pass ``counts`` to reuse a draw.
    """
    spot, times, normals, rate, div, vol, jump_intensity, jump_mean, jump_vol, jump_normals = align(
        spot, times, normals, rate, div, vol, jump_intensity, jump_mean, jump_vol, jump_normals
    )
    num_paths, num_steps, dim = normals.shape
    interval = (times[1:] - times[:-1]).reshape(1, num_steps, 1)
    if counts is None:
        counts = torch.poisson((jump_intensity * interval).expand(num_paths, num_steps, dim))
    compensator = torch.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    drift = (rate - div - 0.5 * vol * vol - jump_intensity * compensator) * interval
    diffusion = vol * normals * torch.sqrt(interval)
    jumps = jump_mean * counts + jump_vol * torch.sqrt(counts.clamp(min=0.0)) * jump_normals
    jumps = torch.where(counts > 0, jumps, torch.zeros_like(jumps))
    exponent = torch.cumsum(drift + diffusion + jumps, dim=1)
    future = spot * torch.exp(exponent)
    initial = torch.broadcast_to(spot, (num_paths, 1, dim)).clone()
    return torch.cat((initial, future), dim=1)
