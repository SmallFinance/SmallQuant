"""SVI, SSVI and Dupire local volatility."""

from __future__ import annotations

import torch

from smallquant._tensor import align


def svi_total_variance(log_moneyness, a, b, rho, m, sigma):
    """Raw SVI total variance. Gatheral (2004)."""
    log_moneyness, a, b, rho, m, sigma = align(log_moneyness, a, b, rho, m, sigma)
    gap = log_moneyness - m
    return a + b * (rho * gap + torch.sqrt(gap * gap + sigma * sigma))


def svi_implied_vol(log_moneyness, expiry, a, b, rho, m, sigma):
    """Black implied volatility of a raw SVI slice."""
    total = svi_total_variance(log_moneyness, a, b, rho, m, sigma)
    return torch.sqrt((total / expiry).clamp(min=0.0))


def svi_butterfly_ok(log_moneyness, a, b, rho, m, sigma):
    """Mask of log-moneyness points that pass Gatheral's butterfly condition."""
    log_moneyness, a, b, rho, m, sigma = align(log_moneyness, a, b, rho, m, sigma)
    gap = log_moneyness - m
    root = torch.sqrt(gap * gap + sigma * sigma)
    total = a + b * (rho * gap + root)
    first = b * (rho + gap / root)
    second = b * sigma * sigma / root**3
    term = 1.0 - log_moneyness * first / (2.0 * total)
    density = term**2 - first**2 / 4.0 * (1.0 / total + 0.25) + second / 2.0
    return (density >= 0) & (total > 0) & (b >= 0) & (sigma > 0) & (rho.abs() < 1)


def ssvi_total_variance(log_moneyness, theta, rho, phi):
    """Surface SVI total variance. Gatheral and Jacquier (2014)."""
    log_moneyness, theta, rho, phi = align(log_moneyness, theta, rho, phi)
    return 0.5 * theta * (
        1.0 + rho * phi * log_moneyness + torch.sqrt((phi * log_moneyness + rho) ** 2 + (1.0 - rho**2))
    )


def ssvi_implied_vol(log_moneyness, expiry, theta, rho, phi):
    total = ssvi_total_variance(log_moneyness, theta, rho, phi)
    return torch.sqrt((total / expiry).clamp(min=0.0))


def ssvi_calendar_ok(theta):
    """Calendar no-arbitrage mask: total ATM variance is increasing in expiry."""
    return theta[..., 1:] >= theta[..., :-1]


def dupire_local_vol(price_fn, *, spot, times, strikes, rate, div):
    """Dupire local volatility from a differentiable call price function.

    ``price_fn(strike, expiry)`` returns undiscounted-currency call prices.
    The result is the local volatility at each ``(strikes, times)`` pair.
    """
    spot, times, strikes, rate, div = align(spot, times, strikes, rate, div)
    strike = strikes.detach().requires_grad_(True)
    expiry = times.detach().requires_grad_(True)
    strike_b, expiry_b = torch.broadcast_tensors(strike, expiry)
    value = price_fn(strike_b, expiry_b)
    d_strike, d_expiry = torch.autograd.grad(value.sum(), (strike_b, expiry_b), create_graph=True)
    (d_strike_2,) = torch.autograd.grad(d_strike.sum(), strike_b, retain_graph=True)
    numerator = d_expiry + (rate - div) * strike_b * d_strike + div * value
    denominator = 0.5 * strike_b**2 * d_strike_2
    local_variance = numerator / denominator.clamp(min=1e-12)
    return torch.sqrt(local_variance.clamp(min=0.0))
