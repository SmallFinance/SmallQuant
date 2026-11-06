"""Batched Monte Carlo on the tensor device."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..math.random import brownian_bridge as apply_brownian_bridge
from ..math.random import standard_normal

Payoff = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Step = Callable[[torch.Tensor], torch.Tensor]


def price(
    payoff: Payoff,
    step: Step,
    *,
    times: torch.Tensor,
    discount: torch.Tensor,
    num_paths: int,
    seed: int,
    dim: int = 1,
    antithetic: bool = True,
    brownian_bridge: bool = False,
    chunk_size: int | None = None,
    return_stderr: bool = False,
    control: Payoff | None = None,
    control_price: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Discounted Monte Carlo price.

    ``step(normals)`` maps ``[num_paths, num_steps, dim]`` standard normals to
    paths ``[num_paths, num_steps + 1, dim]``. ``payoff(paths, times)`` returns
    the undiscounted cash flow per path. ``discount`` scales that cash flow.

    ``brownian_bridge`` reorders each path's normals with QuantLib's bridge
    before ``step``. Antithetic variates stack each draw with its negation.
    ``chunk_size`` evaluates the step and the payoff in path blocks after one
    generator draw, so the price matches the unchunked call.

    A control variate subtracts ``beta * (control - control_price)``. ``beta``
    is the least-squares coefficient on these paths and is held fixed for
    differentiation. ``control_price`` is the known discounted control price.
    """
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be a 1-d grid with at least one step")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if (control is None) != (control_price is None):
        raise ValueError("control and control_price must be supplied together")
    steps = times.numel() - 1
    normals = standard_normal(
        num_paths=num_paths,
        num_steps=steps,
        dim=dim,
        seed=seed,
        dtype=times.dtype,
        device=times.device,
        antithetic=antithetic,
    )
    if brownian_bridge:
        normals = apply_brownian_bridge(normals, times)
    discount = torch.as_tensor(discount, dtype=times.dtype, device=times.device)

    values: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    block = num_paths if chunk_size is None else chunk_size
    for start in range(0, num_paths, block):
        paths = step(normals[start : start + block])
        if paths.ndim != 3 or paths.shape[0] != min(block, num_paths - start) or paths.shape[1] != steps + 1:
            raise ValueError("step must return [num_paths, num_steps + 1, dim]")
        values.append(payoff(paths, times))
        if control is not None:
            controls.append(control(paths, times))
    sampled = torch.cat(values, dim=0) * discount
    if sampled.shape != (num_paths,):
        raise ValueError("payoff must return one value per path")

    if control is None:
        estimate = sampled.mean()
        residuals = sampled
    else:
        control_sample = torch.cat(controls, dim=0) * discount
        known = torch.as_tensor(control_price, dtype=times.dtype, device=times.device)
        centered_control = control_sample - control_sample.mean()
        variance = (centered_control * centered_control).sum()
        covariance = ((sampled - sampled.mean()) * centered_control).sum()
        beta = torch.where(variance > 0, covariance / variance, torch.zeros_like(variance)).detach()
        residuals = sampled - beta * (control_sample - known)
        estimate = residuals.mean()

    if not return_stderr:
        return estimate
    stderr = residuals.std(unbiased=True) / math.sqrt(num_paths)
    return estimate, stderr
