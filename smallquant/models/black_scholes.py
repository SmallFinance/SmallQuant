"""Black-Scholes-Merton closed form for European options."""

from __future__ import annotations

import torch

from smallquant._tensor import align, check
from smallquant.math.distributions import normal_cdf


def _flag(is_call: torch.Tensor | bool, like: torch.Tensor) -> torch.Tensor:
    if isinstance(is_call, bool):
        return torch.tensor(is_call, device=like.device)
    return is_call.to(device=like.device)


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor | bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """European Black-Scholes-Merton price.

    Args:
        spot: underlying spot
        strike: strike
        expiry: time to expiry in years
        rate: continuously compounded risk-free rate
        vol: volatility
        div: continuous dividend / carry yield
        is_call: boolean tensor, true for a call and false for a put

    Returns:
        Option price tensor, broadcast from the inputs.

    At ``expiry <= 0`` or ``vol <= 0`` the price is its limit, the discounted forward
    intrinsic value ``max(+/-(spot * e^{-div T} - strike * e^{-rate T}), 0)``, with finite
    gradients.
    """
    spot, strike, expiry, rate, vol, div = align(
        spot, strike, expiry, rate, vol, div, dtype=dtype
    )
    is_call = _flag(is_call, spot)
    check("spot", spot, spot > 0)
    check("strike", strike, strike > 0)
    check("vol", vol, vol >= 0)
    check("expiry", expiry, expiry >= 0)

    # Mask degenerate inputs before sqrt/division so the unused branch has no NaN gradient.
    degenerate = (expiry <= 0) | (vol <= 0)
    safe_expiry = torch.where(degenerate, torch.ones_like(expiry), expiry)
    safe_vol = torch.where(degenerate, torch.ones_like(vol), vol)

    sqrt_t = torch.sqrt(safe_expiry)
    d1 = (torch.log(spot / strike) + (rate - div + 0.5 * safe_vol**2) * safe_expiry) / (
        safe_vol * sqrt_t
    )
    d2 = d1 - safe_vol * sqrt_t

    nd1 = normal_cdf(d1)
    nd2 = normal_cdf(d2)

    expiry_or_zero = torch.clamp(expiry, min=0.0)
    discounted_spot = spot * torch.exp(-div * expiry_or_zero)
    discounted_strike = strike * torch.exp(-rate * expiry_or_zero)
    call = discounted_spot * nd1 - discounted_strike * nd2
    put = discounted_strike * (1.0 - nd2) - discounted_spot * (1.0 - nd1)

    call_limit = torch.relu(discounted_spot - discounted_strike)
    put_limit = torch.relu(discounted_strike - discounted_spot)
    call = torch.where(degenerate, call_limit, call)
    put = torch.where(degenerate, put_limit, put)

    return torch.where(is_call, call, put)


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
