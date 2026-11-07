"""CRR binomial and trinomial lattices for European and American options."""

from __future__ import annotations

import torch

from smallquant._tensor import align, check


def price(
    *,
    spot,
    strike,
    expiry,
    rate,
    vol,
    div,
    is_call,
    exercise="european",
    num_steps=200,
    kind="crr",
    dtype=None,
):
    """Lattice price.

    ``kind`` is ``"crr"`` or ``"trinomial"``. ``exercise`` is ``"european"``
    or ``"american"``.
    """
    if kind not in ("crr", "trinomial"):
        raise ValueError("kind must be 'crr' or 'trinomial'")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    spot, strike, expiry, rate, vol, div = align(spot, strike, expiry, rate, vol, div, dtype=dtype)
    check("vol", vol, vol > 0)
    check("expiry", expiry, expiry > 0)
    if kind == "crr":
        return _crr(spot, strike, expiry, rate, vol, div, is_call, exercise, num_steps)
    return _trinomial(spot, strike, expiry, rate, vol, div, is_call, exercise, num_steps)


def _intrinsic(spot, strike, is_call):
    return torch.where(is_call, torch.relu(spot - strike), torch.relu(strike - spot))


def _crr(spot, strike, expiry, rate, vol, div, is_call, exercise, num_steps):
    dt = expiry / num_steps
    growth = torch.exp((rate - div) * dt)
    discount = torch.exp(-rate * dt)
    up = torch.exp(vol * torch.sqrt(dt))
    down = 1.0 / up
    probability = ((growth - down) / (up - down)).clamp(0.0, 1.0)
    # Terminal spots: S * u^{n-j} * d^{j}
    steps = torch.arange(num_steps + 1, dtype=spot.dtype, device=spot.device)
    values = _intrinsic(spot * up ** (num_steps - steps) * down**steps, strike, is_call)
    for step in range(num_steps - 1, -1, -1):
        values = discount * (probability * values[:-1] + (1.0 - probability) * values[1:])
        if exercise == "american":
            spots = spot * up ** (step - steps[: step + 1]) * down ** steps[: step + 1]
            values = torch.maximum(values, _intrinsic(spots, strike, is_call))
    return values[0]


def _trinomial(spot, strike, expiry, rate, vol, div, is_call, exercise, num_steps):
    dt = expiry / num_steps
    dx = vol * torch.sqrt(3.0 * dt)
    discount = torch.exp(-rate * dt)
    nu = rate - div - 0.5 * vol * vol
    up_prob = 0.5 * ((vol**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    down_prob = 0.5 * ((vol**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    mid_prob = 1.0 - up_prob - down_prob
    up_prob = up_prob.clamp(0.0, 1.0)
    down_prob = down_prob.clamp(0.0, 1.0)
    mid_prob = mid_prob.clamp(0.0, 1.0)
    # Renormalise after the clamp.
    total = (up_prob + mid_prob + down_prob).clamp(min=1e-12)
    up_prob, mid_prob, down_prob = up_prob / total, mid_prob / total, down_prob / total
    nodes = torch.arange(-num_steps, num_steps + 1, dtype=spot.dtype, device=spot.device)
    values = _intrinsic(spot * torch.exp(nodes * dx), strike, is_call)
    for step in range(num_steps - 1, -1, -1):
        values = discount * (up_prob * values[:-2] + mid_prob * values[1:-1] + down_prob * values[2:])
        if exercise == "american":
            live = torch.arange(-step, step + 1, dtype=spot.dtype, device=spot.device)
            values = torch.maximum(values, _intrinsic(spot * torch.exp(live * dx), strike, is_call))
    return values[0]
