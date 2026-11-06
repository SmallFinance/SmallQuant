"""One-dimensional Black-Scholes finite differences."""

from __future__ import annotations

import torch

from ..math.tridiagonal import solve as solve_tridiagonal
from ..models._common import as_tensors


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
    theta: float = 0.5,
    rannacher_steps: int = 2,
    grid_scale: float = 6.0,
) -> torch.Tensor:
    """Crank-Nicolson price on a log-spot grid concentrated at the strike.

    ``theta=0.5`` is Crank-Nicolson and ``theta=1`` is fully implicit. The first
    ``rannacher_steps`` are forced to fully implicit Euler (Rannacher 1984) so a
    kink in the expiry payoff does not pollute the convergence. The price at
    ``spot`` is read from a natural cubic spline, so delta and gamma from
    autograd do not jump from one grid node to the next.

    ``exercise="american"`` projects the grid onto the intrinsic value after
    every time step.
    """
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    if num_space < 8 or num_time < 1:
        raise ValueError("num_space must be at least 8 and num_time at least 1")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    spot, strike, expiry, rate, vol, div = as_tensors(spot, strike, expiry, rate, vol, div)
    batch_shape = torch.broadcast_shapes(
        spot.shape, strike.shape, expiry.shape, rate.shape, vol.shape, div.shape, is_call.shape
    )
    spot, strike, expiry, rate, vol, div, is_call = (
        tensor.expand(batch_shape).reshape(-1)
        for tensor in (spot, strike, expiry, rate, vol, div, is_call)
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
        step_theta = 1.0 if index < rannacher_steps else theta
        boundary = _boundary(log_spot, strike, rate, div, remaining, is_call, exercise)
        values = _theta_step(values, lower, diag, upper, step, step_theta, boundary)
        if exercise == "american":
            values = torch.maximum(values, payoff)
    solved = _spline(log_spot, values, torch.log(spot))
    return torch.where(expiry <= 0, intrinsic, solved).reshape(batch_shape)


def _log_grid(strike, spot, expiry, vol, num_space: int, grid_scale: float) -> torch.Tensor:
    center = torch.log(strike)[:, None]
    width = grid_scale * vol[:, None] * torch.sqrt(expiry[:, None])
    reach = (torch.log(spot) - torch.log(strike)).abs()[:, None] + 0.5
    half = torch.maximum(width, reach)
    density = torch.clamp(half / 8.0, min=1e-3)
    nodes = torch.linspace(0, 1, num_space, dtype=strike.dtype, device=strike.device)
    alpha = torch.asinh(-half / density)
    beta = torch.asinh(half / density)
    coordinate = alpha * (1.0 - nodes) + beta * nodes
    return center + density * torch.sinh(coordinate)


def _spatial(log_spot, rate, vol, div):
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


def _boundary(log_spot, strike, rate, div, remaining, is_call, exercise: str) -> torch.Tensor:
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


def _theta_step(values, lower, diag, upper, step, theta: float, boundary):
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


def _spline(nodes, values, query):
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

    def take(grid):
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
