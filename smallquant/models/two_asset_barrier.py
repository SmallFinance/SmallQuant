"""Two-asset barrier option. Asset 2 knocks the option on asset 1."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_dr78, norm_cdf

_KINDS = ("down_in", "up_in", "down_out", "up_out")


def price(
    *,
    spot1: torch.Tensor,
    spot2: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol1: torch.Tensor,
    vol2: torch.Tensor,
    div1: torch.Tensor,
    div2: torch.Tensor,
    correlation: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Barrier on ``spot2`` applied to a vanilla option on ``spot1``."""
    if kind not in _KINDS:
        raise ValueError(f"unknown barrier kind {kind!r}")
    (
        spot1,
        spot2,
        strike,
        barrier,
        expiry,
        rate,
        vol1,
        vol2,
        div1,
        div2,
        correlation,
    ) = as_tensors(
        spot1, spot2, strike, barrier, expiry, rate, vol1, vol2, div1, div2, correlation
    )
    b1 = rate - div1
    b2 = rate - div2
    sqrt_t = torch.sqrt(expiry)
    mu1 = b1 - 0.5 * vol1 * vol1
    mu2 = b2 - 0.5 * vol2 * vol2
    d1 = (torch.log(spot1 / strike) + (mu1 + vol1 * vol1) * expiry) / (vol1 * sqrt_t)
    d2 = d1 - vol1 * sqrt_t
    shift = 2.0 * correlation * torch.log(barrier / spot2) / (vol2 * sqrt_t)
    d3 = d1 + shift
    d4 = d2 + shift
    e1 = (torch.log(barrier / spot2) - (mu2 + correlation * vol1 * vol2) * expiry) / (vol2 * sqrt_t)
    e2 = e1 + correlation * vol1 * sqrt_t
    e3 = e1 - 2.0 * torch.log(barrier / spot2) / (vol2 * sqrt_t)
    e4 = e2 - 2.0 * torch.log(barrier / spot2) / (vol2 * sqrt_t)

    def bivariate(left, right, rho):
        return bivariate_dr78(left, right, rho)

    def piece(eta, phi):
        mirror_mu = torch.exp(
            (2.0 * (mu2 + correlation * vol1 * vol2) * torch.log(barrier / spot2)) / vol2**2
        )
        mirror = torch.exp((2.0 * mu2 * torch.log(barrier / spot2)) / vol2**2)
        rho = -eta * phi * correlation
        return eta * spot1 * torch.exp((b1 - rate) * expiry) * (
            bivariate(eta * d1, phi * e1, rho) - mirror_mu * bivariate(eta * d3, phi * e3, rho)
        ) - eta * strike * torch.exp(-rate * expiry) * (
            bivariate(eta * d2, phi * e2, rho) - mirror * bivariate(eta * d4, phi * e4, rho)
        )

    call_vanilla = spot1 * norm_cdf(d1) - strike * torch.exp(-rate * expiry) * norm_cdf(d2)
    put_vanilla = strike * torch.exp(-rate * expiry) * norm_cdf(-d2) - spot1 * norm_cdf(-d1)
    out = {
        "down_out": (piece(1.0, -1.0), piece(-1.0, -1.0)),
        "up_out": (piece(1.0, 1.0), piece(-1.0, 1.0)),
        "down_in": (call_vanilla - piece(1.0, -1.0), put_vanilla - piece(-1.0, -1.0)),
        "up_in": (call_vanilla - piece(1.0, 1.0), put_vanilla - piece(-1.0, 1.0)),
    }[kind]
    return torch.where(is_call, out[0], out[1])
