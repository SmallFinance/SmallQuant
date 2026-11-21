"""One-dimensional Black-Scholes finite differences."""

from __future__ import annotations

from collections.abc import Callable

import torch

from smallquant._tensor import align, check

from ..math.tridiagonal import solve as solve_tridiagonal


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    exercise: str = "european",
    num_space: int = 400,
    num_time: int = 400,
    theta: torch.Tensor | None = None,
    rannacher_steps: int = 2,
    grid_scale: torch.Tensor | None = None,
    barrier: torch.Tensor | None = None,
    barrier_kind: str | None = None,
    rebate: torch.Tensor | None = None,
    local_vol: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Crank-Nicolson price on a log-spot grid concentrated at the strike.

    ``theta`` is a tensor broadcast with the batch, or ``None`` for ``0.5``
    (Crank-Nicolson); ``1`` is fully implicit. ``grid_scale`` likewise, or
    ``None`` for ``6``. The first
    ``rannacher_steps`` are forced to fully implicit Euler (Rannacher 1984) so a
    kink in the expiry payoff does not pollute the convergence. The price at
    ``spot`` is read from a natural cubic spline, so delta and gamma from
    autograd do not jump from one grid node to the next.

    ``exercise="american"`` projects the grid onto the intrinsic value after
    every time step.
    """
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    spot, strike, expiry, rate, vol, div = align(spot, strike, expiry, rate, vol, div)
    theta = torch.full_like(spot, 0.5) if theta is None else align(theta, spot)[0]
    grid_scale = torch.full_like(spot, 6.0) if grid_scale is None else align(grid_scale, spot)[0]
    check("theta", theta, (theta >= 0) & (theta <= 1))
    if barrier is not None or local_vol is not None:
        return _with_barrier_or_local_vol(
            spot=spot,
            strike=strike,
            expiry=expiry,
            rate=rate,
            vol=vol,
            div=div,
            is_call=is_call,
            exercise=exercise,
            num_space=num_space,
            num_time=num_time,
            theta=theta,
            rannacher_steps=rannacher_steps,
            grid_scale=grid_scale,
            barrier=barrier,
            barrier_kind=barrier_kind,
            rebate=rebate,
            local_vol=local_vol,
        )
    if num_space < 8 or num_time < 1:
        raise ValueError("num_space must be at least 8 and num_time at least 1")
    batch_shape = torch.broadcast_shapes(
        spot.shape, strike.shape, expiry.shape, rate.shape, vol.shape, div.shape, is_call.shape,
        theta.shape, grid_scale.shape,
    )
    spot, strike, expiry, rate, vol, div, is_call, theta, grid_scale = (
        tensor.expand(batch_shape).reshape(-1)
        for tensor in (spot, strike, expiry, rate, vol, div, is_call, theta, grid_scale)
    )
    intrinsic = torch.where(is_call, torch.relu(spot - strike), torch.relu(strike - spot))
    safe_expiry = torch.where(expiry <= 0, torch.ones_like(expiry), expiry)
    log_spot = _log_grid(strike, spot.detach(), safe_expiry, vol, num_space, grid_scale)
    spots = torch.exp(log_spot)
    payoff = torch.where(
        is_call[:, None],
        torch.relu(spots - strike[:, None]),
        torch.relu(strike[:, None] - spots),
    )
    lower, diag, upper = _spatial(log_spot, rate, vol, div)
    values = payoff
    step = safe_expiry[:, None] / num_time
    for index in range(num_time):
        remaining = step * (index + 1)
        step_theta = torch.ones_like(theta) if index < rannacher_steps else theta
        boundary = _boundary(log_spot, strike, rate, div, remaining, is_call, exercise)
        values = _theta_step(values, lower, diag, upper, step, step_theta[:, None], boundary)
        if exercise == "american":
            values = torch.maximum(values, payoff)
    solved = _spline(log_spot, values, torch.log(spot))
    return torch.where(expiry <= 0, intrinsic, solved).reshape(batch_shape)


def _log_grid(
    strike: torch.Tensor,
    spot: torch.Tensor,
    expiry: torch.Tensor,
    vol: torch.Tensor,
    num_space: int,
    grid_scale: torch.Tensor,
) -> torch.Tensor:
    center = torch.log(strike)[:, None]
    width = grid_scale[:, None] * vol[:, None] * torch.sqrt(expiry[:, None])
    reach = (torch.log(spot) - torch.log(strike)).abs()[:, None] + 0.5
    half = torch.maximum(width, reach)
    density = torch.clamp(half / 8.0, min=1e-3)
    nodes = torch.linspace(0, 1, num_space, dtype=strike.dtype, device=strike.device)
    alpha = torch.asinh(-half / density)
    beta = torch.asinh(half / density)
    coordinate = alpha * (1.0 - nodes) + beta * nodes
    return center + density * torch.sinh(coordinate)


def _spatial(log_spot: torch.Tensor, rate: torch.Tensor, vol: torch.Tensor, div: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    drift = (rate - div - 0.5 * vol * vol)[:, None]
    diffusion = (0.5 * vol * vol)[:, None]
    spacing = log_spot[:, 1:] - log_spot[:, :-1]
    left = spacing[:, :-1]
    right = spacing[:, 1:]
    denom = left * right * (left + right)
    first_left = -(right * right) / denom
    first_mid = (right * right - left * left) / denom
    first_right = (left * left) / denom
    second_left = 2.0 * right / denom
    second_mid = -2.0 * (left + right) / denom
    second_right = 2.0 * left / denom
    lower = drift * first_left + diffusion * second_left
    diag = drift * first_mid + diffusion * second_mid - rate[:, None]
    upper = drift * first_right + diffusion * second_right
    return lower, diag, upper


def _boundary(log_spot: torch.Tensor, strike: torch.Tensor, rate: torch.Tensor, div: torch.Tensor, remaining: torch.Tensor, is_call: torch.Tensor, exercise: str) -> torch.Tensor:
    low = torch.exp(log_spot[:, :1])
    high = torch.exp(log_spot[:, -1:])
    strike = strike[:, None]
    carry_low = low * torch.exp(-div[:, None] * remaining)
    carry_high = high * torch.exp(-div[:, None] * remaining)
    cash = strike * torch.exp(-rate[:, None] * remaining)
    call = is_call[:, None]
    if exercise == "american":
        lower = torch.where(call, torch.zeros_like(low), torch.relu(strike - low))
        upper = torch.where(call, torch.relu(high - strike), torch.zeros_like(high))
    else:
        lower = torch.where(call, torch.zeros_like(low), torch.relu(cash - carry_low))
        upper = torch.where(call, torch.relu(carry_high - cash), torch.zeros_like(high))
    return torch.cat((lower, upper), dim=-1)


def _theta_step(
    values: torch.Tensor,
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    step: torch.Tensor,
    theta: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    explicit = (
        (1.0 - theta) * step * lower * values[:, :-2]
        + (1.0 + (1.0 - theta) * step * diag) * values[:, 1:-1]
        + (1.0 - theta) * step * upper * values[:, 2:]
    )
    rhs_left = explicit[:, :1] + theta * step * lower[:, :1] * boundary[:, :1]
    rhs_right = explicit[:, -1:] + theta * step * upper[:, -1:] * boundary[:, 1:]
    rhs = torch.cat((rhs_left, explicit[:, 1:-1], rhs_right), dim=-1)
    lhs_lower = -theta * step * lower
    lhs_upper = -theta * step * upper
    lhs_diag = 1.0 - theta * step * diag
    lhs_lower = torch.cat((torch.zeros_like(lhs_lower[:, :1]), lhs_lower[:, 1:]), dim=-1)
    lhs_upper = torch.cat((lhs_upper[:, :-1], torch.zeros_like(lhs_upper[:, :1])), dim=-1)
    interior = solve_tridiagonal(lhs_lower, lhs_diag, lhs_upper, rhs)
    return torch.cat((boundary[:, :1], interior, boundary[:, 1:]), dim=-1)


def _spline(nodes: torch.Tensor, values: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    spacing = nodes[:, 1:] - nodes[:, :-1]
    slope = (values[:, 1:] - values[:, :-1]) / spacing
    left = spacing[:, :-1]
    right = spacing[:, 1:]
    system_lower = left / 6.0
    system_diag = (left + right) / 3.0
    system_upper = right / 6.0
    system_lower = torch.cat((torch.zeros_like(system_lower[:, :1]), system_lower[:, 1:]), dim=-1)
    system_upper = torch.cat((system_upper[:, :-1], torch.zeros_like(system_upper[:, :1])), dim=-1)
    interior_moment = solve_tridiagonal(system_lower, system_diag, system_upper, slope[:, 1:] - slope[:, :-1])
    moment = torch.cat(
        (torch.zeros_like(values[:, :1]), interior_moment, torch.zeros_like(values[:, :1])),
        dim=-1,
    )
    index = (query[:, None] >= nodes).sum(dim=-1) - 1
    index = index.clamp(0, nodes.shape[-1] - 2)

    def take(grid: torch.Tensor) -> torch.Tensor:
        return grid.gather(-1, index[:, None]).squeeze(-1)

    next_index = (index + 1)[:, None]
    left_node = take(nodes)
    right_node = nodes.gather(-1, next_index).squeeze(-1)
    left_value = take(values)
    right_value = values.gather(-1, next_index).squeeze(-1)
    left_moment = take(moment)
    right_moment = moment.gather(-1, next_index).squeeze(-1)
    span = right_node - left_node
    weight_left = (right_node - query) / span
    weight_right = (query - left_node) / span
    return (
        weight_left * left_value
        + weight_right * right_value
        + ((weight_left**3 - weight_left) * left_moment + (weight_right**3 - weight_right) * right_moment)
        * span**2
        / 6.0
    )


def _with_barrier_or_local_vol(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    exercise: str,
    num_space: int,
    num_time: int,
    theta: torch.Tensor,
    rannacher_steps: int,
    grid_scale: torch.Tensor,
    barrier: torch.Tensor | None,
    barrier_kind: str | None,
    rebate: torch.Tensor | None,
    local_vol: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Crank-Nicolson with a Dirichlet barrier and an optional local volatility."""
    spot, strike, expiry, rate, vol, div = align(spot, strike, expiry, rate, vol, div)
    kind = None if barrier_kind is None else barrier_kind.replace("_and_", "_")
    knock_in = kind in ("down_in", "up_in")
    if knock_in:
        vanilla = price(
            spot=spot,
            strike=strike,
            expiry=expiry,
            rate=rate,
            vol=vol,
            div=div,
            is_call=is_call,
            exercise=exercise,
            num_space=num_space,
            num_time=num_time,
            theta=theta,
            rannacher_steps=rannacher_steps,
            grid_scale=grid_scale,
            local_vol=local_vol,
        )
        knocked_out = _with_barrier_or_local_vol(
            spot=spot,
            strike=strike,
            expiry=expiry,
            rate=rate,
            vol=vol,
            div=div,
            is_call=is_call,
            exercise=exercise,
            num_space=num_space,
            num_time=num_time,
            theta=theta,
            rannacher_steps=rannacher_steps,
            grid_scale=grid_scale,
            barrier=barrier,
            barrier_kind="down_out" if kind == "down_in" else "up_out",
            rebate=rebate,
            local_vol=local_vol,
        )
        return vanilla - knocked_out
    batch_shape = torch.broadcast_shapes(
        spot.shape, strike.shape, expiry.shape, rate.shape, vol.shape, div.shape, theta.shape, grid_scale.shape
    )
    spot, strike, expiry, rate, vol, div, theta, grid_scale = (
        tensor.expand(batch_shape).reshape(-1)
        for tensor in (spot, strike, expiry, rate, vol, div, theta, grid_scale)
    )
    is_call = is_call.expand(batch_shape).reshape(-1)
    safe_expiry = torch.where(expiry <= 0, torch.ones_like(expiry), expiry)
    log_spot = _log_grid(strike, spot.detach(), safe_expiry, vol, num_space, grid_scale)
    spots = torch.exp(log_spot)
    payoff = torch.where(is_call[:, None], torch.relu(spots - strike[:, None]), torch.relu(strike[:, None] - spots))
    cash = torch.zeros_like(spot) if rebate is None else align(rebate)[0].expand_as(spot)
    values = payoff
    step = safe_expiry[:, None] / num_time
    for index in range(num_time):
        remaining = step * (index + 1)
        if local_vol is None:
            lower, diag, upper = _spatial(log_spot, rate, vol, div)
        else:
            elapsed = safe_expiry - remaining[:, 0]
            node_vol = local_vol(spots, elapsed)
            lower, diag, upper = _spatial_nodes(log_spot, rate, node_vol, div)
        step_theta = torch.ones_like(theta) if index < rannacher_steps else theta
        boundary = _boundary(log_spot, strike, rate, div, remaining, is_call, exercise)
        values = _theta_step(values, lower, diag, upper, step, step_theta[:, None], boundary)
        if barrier is not None:
            level = align(barrier, spot)[0]
            if kind == "down_out":
                values = torch.where(spots < level, cash[:, None], values)
            elif kind == "up_out":
                values = torch.where(spots > level, cash[:, None], values)
        if exercise == "american":
            values = torch.maximum(values, payoff)
    solved = _spline(log_spot, values, torch.log(spot))
    intrinsic = torch.where(is_call, torch.relu(spot - strike), torch.relu(strike - spot))
    return torch.where(expiry <= 0, intrinsic, solved).reshape(batch_shape)


def _spatial_nodes(log_spot: torch.Tensor, rate: torch.Tensor, vol_nodes: torch.Tensor, div: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vol = vol_nodes[:, 1:-1]
    drift = rate[:, None] - div[:, None] - 0.5 * vol * vol
    diffusion = 0.5 * vol * vol
    spacing = log_spot[:, 1:] - log_spot[:, :-1]
    left = spacing[:, :-1]
    right = spacing[:, 1:]
    denom = left * right * (left + right)
    first_left = -(right * right) / denom
    first_mid = (right * right - left * left) / denom
    first_right = (left * left) / denom
    second_left = 2.0 * right / denom
    second_mid = -2.0 * (left + right) / denom
    second_right = 2.0 * left / denom
    lower = drift * first_left + diffusion * second_left
    diag = drift * first_mid + diffusion * second_mid - rate[:, None]
    upper = drift * first_right + diffusion * second_right
    return lower, diag, upper
