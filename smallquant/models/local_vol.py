"""Log-Euler local-volatility paths and grid interpolation."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.interp import linear


def step(*, spot, times, normals, rate, div, strikes, expiries, local_vols):
    """Log-Euler paths. ``local_vols`` is shaped ``[expiries, strikes]``."""
    spot, times, normals, rate, div, strikes, expiries, local_vols = align(
        spot, times, normals, rate, div, strikes, expiries, local_vols
    )
    num_paths, num_steps, dim = normals.shape
    state = spot.reshape(-1)[0].expand(num_paths, dim).clone()
    paths = [state[:, 0] if dim == 1 else state]
    stored = [state]
    for index in range(num_steps):
        dt = times[index + 1] - times[index]
        elapsed = times[index]
        sigma = _local(state[..., 0], elapsed, strikes, expiries, local_vols).unsqueeze(-1)
        state = state * torch.exp(
            (rate - div - 0.5 * sigma * sigma) * dt + sigma * torch.sqrt(dt) * normals[:, index, :]
        )
        stored.append(state)
    return torch.stack(stored, dim=1)


def _local(spot, time, strikes, expiries, local_vols):
    # Interpolate in strike at the two surrounding expiries, then in time.
    expiry_index = torch.searchsorted(expiries.detach(), time.detach().reshape(-1)).clamp(1, expiries.numel() - 1)
    left = expiry_index[0] - 1
    right = expiry_index[0]
    span = (expiries[right] - expiries[left]).clamp(min=1e-12)
    weight = ((time - expiries[left]) / span).clamp(0.0, 1.0)
    left_vol = linear(spot, strikes, local_vols[left])
    right_vol = linear(spot, strikes, local_vols[right])
    return left_vol + weight * (right_vol - left_vol)
