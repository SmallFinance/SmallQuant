"""Differentiable deposit and par-swap bootstrap.

Callers pass year fractions. The pricing core does not import ``smallquant.time``.
"""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.rates.curve import DiscountCurve


def bootstrap(*, pay_times, rates, kinds, deltas):
    """Bootstrap discount factors from deposits and spot-starting par swaps.

    ``pay_times``, ``rates`` and ``deltas`` are aligned tensors of pillar
    length. ``kinds`` is a sequence of ``"deposit"`` or ``"swap"``, one per
    pillar. A deposit at pillar ``i`` has continuous rate ``rates[i]``. A swap
    ending at pillar ``i`` pays fixed ``rates[i]`` on every pillar ``0..i``
    with accrual ``deltas``. The float leg is worth ``1 - DF``.

    The map from quotes to discount factors is closed form, so bucketed deltas
    come from autograd.

    Returns:
        DiscountCurve including time 0.
    """
    pay_times, rates, deltas = align(pay_times, rates, deltas)
    if len(kinds) != pay_times.shape[0]:
        raise ValueError("kinds must have one entry per pillar")
    times = [pay_times.new_zeros(())]
    factors = [pay_times.new_ones(())]
    known = {}
    for index, kind in enumerate(kinds):
        pillar = pay_times[index]
        if kind == "deposit":
            factor = torch.exp(-rates[index] * pillar)
        elif kind == "swap":
            level = pillar.new_zeros(())
            for earlier in range(index):
                level = level + deltas[earlier] * known[earlier]
            factor = (1.0 - rates[index] * level) / (1.0 + rates[index] * deltas[index])
        else:
            raise ValueError(f"unknown instrument kind {kind!r}")
        known[index] = factor
        times.append(pillar)
        factors.append(factor)
    return DiscountCurve(torch.stack(times), torch.stack(factors))


def par_swap_rate(curve: DiscountCurve, pay_times: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Par rate of a spot-starting swap paying on ``pay_times``."""
    pay_times, deltas = align(pay_times, deltas, curve.times)[:2]
    factors = curve.discount(pay_times)
    level = (deltas * factors).sum()
    return (1.0 - factors[-1]) / level
