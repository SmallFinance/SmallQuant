"""Hagan, Kumar, Lesniewski and Woodward (2002) SABR implied volatility."""

from __future__ import annotations

import torch

from smallquant._tensor import align


def implied_vol(*, forward, strike, expiry, alpha, beta, rho, nu, dtype=None):
    """Lognormal SABR volatility.

    The at-the-money point uses the series limit of ``z / x(z)``, so the value
    and its gradient stay finite when ``strike == forward``.

    Args:
        forward: forward price
        strike: strike
        expiry: year fraction
        alpha: initial volatility level
        beta: backbone exponent in ``[0, 1]``
        rho: correlation
        nu: volatility of volatility
    """
    forward, strike, expiry, alpha, beta, rho, nu = align(
        forward, strike, expiry, alpha, beta, rho, nu, dtype=dtype
    )
    rho = rho.clamp(-0.999999, 0.999999)
    one_minus_beta = 1.0 - beta
    log_moneyness = torch.log(forward / strike)
    fk_power = (forward * strike) ** (one_minus_beta / 2.0)
    z = nu / alpha * fk_power * log_moneyness
    radical = torch.sqrt((1.0 - 2.0 * rho * z + z * z).clamp(min=0.0))
    xz = torch.log((radical + z - rho) / (1.0 - rho))
    small = z.abs() < 1e-7
    z_over_x = torch.where(small, torch.ones_like(z), z / torch.where(small, torch.ones_like(xz), xz))
    log2 = log_moneyness**2
    denominator = fk_power * (
        1.0 + one_minus_beta**2 / 24.0 * log2 + one_minus_beta**4 / 1920.0 * log2**2
    )
    prefactor = alpha / denominator
    correction = (
        one_minus_beta**2 / 24.0 * alpha**2 / (forward * strike).clamp(min=1e-18) ** one_minus_beta
        + 0.25 * rho * beta * nu * alpha / fk_power
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    wing = prefactor * z_over_x * (1.0 + correction * expiry)
    forward_power = forward ** one_minus_beta
    atm_correction = (
        one_minus_beta**2 / 24.0 * alpha**2 / forward_power.clamp(min=1e-18) ** 2
        + 0.25 * rho * beta * nu * alpha / forward_power
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    atm = alpha / forward_power * (1.0 + atm_correction * expiry)
    return torch.where(log_moneyness.abs() < 1e-8, atm, wing)
