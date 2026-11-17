"""Geometric-average Asian options with closed forms."""

from __future__ import annotations

import torch

from ._common import as_tensors, black_price, norm_cdf


def continuous_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Continuous geometric-average price Asian."""
    spot, strike, expiry, rate, vol, div = as_tensors(spot, strike, expiry, rate, vol, div)
    adjusted_div = 0.5 * (rate + div + vol * vol / 6.0)
    forward = spot * torch.exp(-adjusted_div * expiry) / torch.exp(-rate * expiry)
    discount = torch.exp(-rate * expiry)
    return black_price(forward, strike, vol * torch.sqrt(expiry / 3.0), discount, is_call)


def discrete_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    fixing_times: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Discrete geometric-average price Asian with no past fixings.

    ``fixing_times`` are year fractions from today, all non-negative.
    """
    spot, strike, expiry, rate, vol, div = as_tensors(spot, strike, expiry, rate, vol, div)
    times = torch.as_tensor(fixing_times, dtype=spot.dtype, device=spot.device)
    times = times[times >= 0]
    n = times.numel()
    time_sum = times.sum()
    index = torch.arange(1, n, device=spot.device)
    # i runs over pastFixings+1 .. N-1, pastFixings = 0, so i = 1..n-1
    # temp += fixingTimes[i-1] * (N - i)
    if n > 1:
        temp = (times[:-1] * (n - index).to(times.dtype)).sum()
    else:
        temp = torch.zeros((), dtype=spot.dtype, device=spot.device)
    variance = vol * vol / n / n * (time_sum + 2.0 * temp)
    nu = rate - div - 0.5 * vol * vol
    mu_g = torch.log(spot) + nu * time_sum / n
    forward = torch.exp(mu_g + 0.5 * variance)
    return black_price(forward, strike, torch.sqrt(variance), torch.exp(-rate * expiry), is_call)


def discrete_strike_price(
    *,
    spot: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    fixing_times: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Discrete geometric-average strike Asian with no past fixings.

    The last fixing time is the start of the residual life. ``expiry`` is the
    option expiry and must be at or after the last fixing.
    """
    spot, expiry, rate, vol, div = as_tensors(spot, expiry, rate, vol, div)
    times = torch.as_tensor(fixing_times, dtype=spot.dtype, device=spot.device)
    # QuantLib measures fixing times from the first fixing date.
    offset = times[0]
    relative = times - offset
    n = relative.numel()
    time_sum = relative.sum()
    index = torch.arange(1, n, device=spot.device)
    if n > 1:
        temp = (relative[:-1] * (n - index).to(relative.dtype)).sum()
    else:
        temp = torch.zeros((), dtype=spot.dtype, device=spot.device)
    # QuantLib measures the residual life from the first fixing when there is no past fixing.
    residual = expiry - times[0]
    variance = vol * vol / n / n * (time_sum + 2.0 * temp)
    covariance = vol * vol / n * time_sum
    sigma_sum_2 = variance + vol * vol * residual - 2.0 * covariance
    nu = rate - div - 0.5 * vol * vol
    mu_g = torch.log(spot) + nu * time_sum / n
    sqrt_var = torch.sqrt(sigma_sum_2)
    y1 = (
        torch.log(spot) + (rate - div) * residual - mu_g - 0.5 * variance + 0.5 * sigma_sum_2
    ) / sqrt_var
    y2 = y1 - sqrt_var
    call = spot * torch.exp(-div * residual) * norm_cdf(y1) - torch.exp(
        mu_g + 0.5 * variance - rate * residual
    ) * norm_cdf(y2)
    put = -spot * torch.exp(-div * residual) * norm_cdf(-y1) + torch.exp(
        mu_g + 0.5 * variance - rate * residual
    ) * norm_cdf(-y2)
    return torch.where(is_call, call, put)
