"""Heston characteristic function and Andersen's QE step."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.distributions import normal_cdf


def characteristic_function(u, *, spot, rate, div, expiry, v0, kappa, theta, sigma, rho):
    """Heston characteristic function of ``log(spot)``, little-trap branch.

    Albrecher, Mayer, Schoutens and Tistaert (2007). ``u`` may be complex.
    The remaining arguments are real tensors and broadcast against ``u``.
    """
    spot, rate, div, expiry, v0, kappa, theta, sigma, rho = align(
        spot, rate, div, expiry, v0, kappa, theta, sigma, rho
    )
    if not torch.is_complex(u):
        u = u.to(torch.complex128)
    else:
        u = u.to(torch.complex128)
    real = spot.dtype
    def c(value):
        return value.to(real).to(torch.complex128)

    iu = 1j * u
    sigma_c = c(sigma)
    kappa_c = c(kappa)
    rho_c = c(rho)
    theta_c = c(theta)
    v0_c = c(v0)
    expiry_c = c(expiry)
    discriminant = torch.sqrt(
        (rho_c * sigma_c * iu - kappa_c) ** 2 + sigma_c**2 * (iu + u * u)
    )
    half = kappa_c - rho_c * sigma_c * iu
    g = (half - discriminant) / (half + discriminant)
    exp_dt = torch.exp(-discriminant * expiry_c)
    log_term = torch.log((1.0 - g * exp_dt) / (1.0 - g))
    capital_c = (c(rate) - c(div)) * iu * expiry_c + kappa_c * theta_c / sigma_c**2 * (
        (half - discriminant) * expiry_c - 2.0 * log_term
    )
    capital_d = (half - discriminant) / sigma_c**2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return torch.exp(capital_c + capital_d * v0_c + iu * torch.log(c(spot)))


def step(*, spot, v0, times, normals, rate, div, kappa, theta, sigma, rho):
    """Andersen (2008) QE paths. ``normals`` has a trailing dimension of 2.

    Dimension 0 drives the variance and dimension 1 is the idiosyncratic spot
    shock. The result has shape ``[num_paths, num_steps + 1, 2]`` with the
    spot in the last dimension's first column and the variance in the second.
    """
    spot, v0, times, normals, rate, div, kappa, theta, sigma, rho = align(
        spot, v0, times, normals, rate, div, kappa, theta, sigma, rho
    )
    if normals.shape[-1] != 2:
        raise ValueError("Heston QE requires normals with dim 2")
    num_paths, num_steps, _ = normals.shape
    dt = times[1:] - times[:-1]
    variance = v0.reshape(-1)[0].expand(num_paths).clone()
    log_spot = torch.log(spot).reshape(-1)[0].expand(num_paths).clone()
    spots = [spot.reshape(-1)[0].expand(num_paths)]
    variances = [variance]
    gamma1 = 0.5
    gamma2 = 0.5
    for index in range(num_steps):
        step_dt = dt[index]
        z_v = normals[:, index, 0]
        z_s = normals[:, index, 1]
        variance = _qe_next(variance, kappa, theta, sigma, step_dt, z_v)
        k0 = -(rho * kappa * theta) / sigma * step_dt
        k1 = gamma1 * step_dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = gamma2 * step_dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = gamma1 * step_dt * (1.0 - rho * rho)
        k4 = gamma2 * step_dt * (1.0 - rho * rho)
        log_spot = (
            log_spot
            + (rate - div) * step_dt
            + k0
            + k1 * variances[-1]
            + k2 * variance
            + torch.sqrt((k3 * variances[-1] + k4 * variance).clamp(min=0.0)) * z_s
        )
        spots.append(torch.exp(log_spot))
        variances.append(variance)
    spot_path = torch.stack(spots, dim=1)
    var_path = torch.stack(variances, dim=1)
    return torch.stack((spot_path, var_path), dim=-1)


def _qe_next(variance, kappa, theta, sigma, dt, z):
    decay = torch.exp(-kappa * dt)
    mean = theta + (variance - theta) * decay
    square = (
        variance * sigma**2 * decay / kappa * (1.0 - decay)
        + theta * sigma**2 / (2.0 * kappa) * (1.0 - decay) ** 2
    )
    psi = square / mean.clamp(min=1e-12) ** 2
    psi_c = 1.5
    two_over = (2.0 / psi.clamp(min=1e-8)).clamp(min=1.0)
    b2 = two_over - 1.0 + torch.sqrt(two_over * (two_over - 1.0))
    a = mean / (1.0 + b2)
    quadratic = a * (torch.sqrt(b2) + z) ** 2
    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / mean.clamp(min=1e-12)
    uniform = normal_cdf(z).clamp(1e-12, 1.0 - 1e-12)
    exponential = torch.where(
        uniform <= p,
        torch.zeros_like(uniform),
        torch.log((1.0 - p) / (1.0 - uniform)) / beta,
    )
    return torch.where(psi <= psi_c, quadratic, exponential).clamp(min=0.0)
