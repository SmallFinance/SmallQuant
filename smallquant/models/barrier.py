"""Single-barrier European options, Merton and Reiner-Rubinstein."""

from __future__ import annotations

import torch

from ._common import as_tensors, norm_cdf

_KINDS = ("down_in", "up_in", "down_out", "up_out")


def _n(x: torch.Tensor) -> torch.Tensor:
    return norm_cdf(x)


def _guard(power: torch.Tensor, nval: torch.Tensor) -> torch.Tensor:
    return torch.where(nval == 0, torch.zeros_like(power), power * nval)


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    rebate: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """European single-barrier price.

    ``kind`` is ``down_in``, ``up_in``, ``down_out`` or ``up_out``. ``rebate`` is
    paid at expiry for knock-in and at hit for knock-out.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown barrier kind {kind!r}")
    spot, strike, barrier, expiry, rate, vol, div, rebate = as_tensors(
        spot, strike, barrier, expiry, rate, vol, div, rebate
    )
    std = vol * torch.sqrt(expiry)
    mu = (rate - div) / vol**2 - 0.5
    mu_sigma = (1.0 + mu) * std
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    hs = barrier / spot
    pow0 = hs ** (2.0 * mu)
    pow1 = pow0 * hs * hs

    def A(phi):
        x1 = torch.log(spot / strike) / std + mu_sigma
        return phi * (spot * div_df * _n(phi * x1) - strike * rate_df * _n(phi * (x1 - std)))

    def B(phi):
        x2 = torch.log(spot / barrier) / std + mu_sigma
        return phi * (spot * div_df * _n(phi * x2) - strike * rate_df * _n(phi * (x2 - std)))

    def C(eta, phi):
        y1 = torch.log(barrier * hs / strike) / std + mu_sigma
        return phi * (
            spot * div_df * _guard(pow1, _n(eta * y1))
            - strike * rate_df * _guard(pow0, _n(eta * (y1 - std)))
        )

    def D(eta, phi):
        y2 = torch.log(barrier / spot) / std + mu_sigma
        return phi * (
            spot * div_df * _guard(pow1, _n(eta * y2))
            - strike * rate_df * _guard(pow0, _n(eta * (y2 - std)))
        )

    def E(eta):
        x2 = torch.log(spot / barrier) / std + mu_sigma
        y2 = torch.log(barrier / spot) / std + mu_sigma
        return rebate * rate_df * (
            _n(eta * (x2 - std)) - _guard(pow0, _n(eta * (y2 - std)))
        )

    def F(eta):
        lam = torch.sqrt(mu**2 + 2.0 * rate / vol**2)
        z = torch.log(barrier / spot) / std + lam * std
        return rebate * (
            _guard(hs ** (mu + lam), _n(eta * z))
            + _guard(hs ** (mu - lam), _n(eta * (z - 2.0 * lam * std)))
        )

    above = strike >= barrier
    call = {
        "down_in": torch.where(above, C(1.0, 1.0) + E(1.0), A(1.0) - B(1.0) + D(1.0, 1.0) + E(1.0)),
        "up_in": torch.where(above, A(1.0) + E(-1.0), B(1.0) - C(-1.0, 1.0) + D(-1.0, 1.0) + E(-1.0)),
        "down_out": torch.where(above, A(1.0) - C(1.0, 1.0) + F(1.0), B(1.0) - D(1.0, 1.0) + F(1.0)),
        "up_out": torch.where(above, F(-1.0), A(1.0) - B(1.0) + C(-1.0, 1.0) - D(-1.0, 1.0) + F(-1.0)),
    }[kind]
    put = {
        "down_in": torch.where(above, B(-1.0) - C(1.0, -1.0) + D(1.0, -1.0) + E(1.0), A(-1.0) + E(1.0)),
        "up_in": torch.where(above, A(-1.0) - B(-1.0) + D(-1.0, -1.0) + E(-1.0), C(-1.0, -1.0) + E(-1.0)),
        "down_out": torch.where(above, A(-1.0) - B(-1.0) + C(1.0, -1.0) - D(1.0, -1.0) + F(1.0), F(1.0)),
        "up_out": torch.where(above, B(-1.0) - D(-1.0, -1.0) + F(-1.0), A(-1.0) - C(-1.0, -1.0) + F(-1.0)),
    }[kind]
    return torch.where(is_call, call, put)
