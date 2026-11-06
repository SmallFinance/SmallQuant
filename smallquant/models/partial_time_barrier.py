"""Partial-time barrier options."""

from __future__ import annotations

import torch

from .black_scholes import price as bs_price
from ._common import as_tensors, bivariate_dr78

_KINDS = ("down_out", "down_in", "up_out", "up_in")
_RANGES = ("start", "end_b1", "end_b2")
_FLIP = {"down_out": "up_out", "up_out": "down_out", "down_in": "up_in", "up_in": "down_in"}


def price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    cover: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
    kind: str,
    barrier_range: str,
) -> torch.Tensor:
    """Partial-time barrier.

    ``cover`` is the year fraction to the end of the barrier window.
    ``barrier_range`` is ``start``, ``end_b1`` or ``end_b2``. Down-and-in and
    up-and-in are available for ``start`` only, as in QuantLib.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown barrier kind {kind!r}")
    if barrier_range not in _RANGES:
        raise ValueError(f"unknown barrier range {barrier_range!r}")
    spot, strike, barrier, expiry, cover, rate, vol, div = as_tensors(
        spot, strike, barrier, expiry, cover, rate, vol, div
    )
    call = _call(spot, strike, barrier, expiry, cover, rate, vol, div, kind, barrier_range)
    put_strike = spot * spot / strike
    put_barrier = spot * spot / barrier
    mirrored = _call(
        spot, put_strike, put_barrier, expiry, cover, div, vol, rate, _FLIP[kind], barrier_range
    )
    put = strike / spot * mirrored
    return torch.where(is_call, call, put)


def _call(spot, strike, barrier, expiry, cover, rate, vol, div, kind, barrier_range):
    if barrier_range == "start":
        eta = 1.0 if kind.startswith("down") else -1.0
        out = _ca(spot, strike, barrier, expiry, cover, rate, vol, div, eta)
        if kind in ("down_out", "up_out"):
            return out
        vanilla = bs_price(
            spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=torch.ones_like(spot, dtype=torch.bool)
        )
        return vanilla - out
    if kind in ("down_in", "up_in"):
        raise ValueError(f"{kind} partial-time end barrier is not implemented")
    if barrier_range == "end_b1":
        return _cob1(spot, strike, barrier, expiry, cover, rate, vol, div)
    return _cob2(spot, strike, barrier, expiry, cover, rate, vol, div, kind)


def _terms(spot, strike, barrier, expiry, cover, rate, vol, div):
    carry = rate - div
    rho = torch.sqrt(cover / expiry)
    vol_cover = vol
    mu = (carry - 0.5 * vol_cover**2) / vol_cover**2
    sqrt_t = torch.sqrt(expiry)
    sqrt_c = torch.sqrt(cover)

    def d1():
        return (torch.log(spot / strike) + (carry + 0.5 * vol**2) * expiry) / (sqrt_t * vol)

    def d2():
        return d1() - vol * sqrt_t

    def e1():
        return (torch.log(spot / barrier) + (carry + 0.5 * vol**2) * cover) / (sqrt_c * vol)

    def e2():
        return e1() - vol * sqrt_c

    def e3():
        return e1() + 2.0 * torch.log(barrier / spot) / (vol * sqrt_c)

    def e4():
        return e3() - vol * sqrt_c

    def f1():
        return (
            torch.log(spot / strike) + 2.0 * torch.log(barrier / spot) + (carry + 0.5 * vol**2) * expiry
        ) / (vol * sqrt_t)

    def f2():
        return f1() - vol * sqrt_t

    def g1():
        return (torch.log(spot / barrier) + (carry + 0.5 * vol**2) * expiry) / (sqrt_t * vol)

    def g2():
        return g1() - vol * sqrt_t

    def g3():
        return g1() + 2.0 * torch.log(barrier / spot) / (vol * sqrt_t)

    def g4():
        return g3() - vol * sqrt_t

    return {
        "carry": carry,
        "rho": rho,
        "mu": mu,
        "d1": d1(),
        "d2": d2(),
        "e1": e1(),
        "e2": e2(),
        "e3": e3(),
        "e4": e4(),
        "f1": f1(),
        "f2": f2(),
        "g1": g1(),
        "g2": g2(),
        "g3": g3(),
        "g4": g4(),
        "hs_mu": (barrier / spot) ** (2.0 * mu),
        "hs_mu1": (barrier / spot) ** (2.0 * (mu + 1.0)),
    }


def _m(a, b, rho):
    return bivariate_dr78(a, b, rho)


def _ca(spot, strike, barrier, expiry, cover, rate, vol, div, eta: float):
    t = _terms(spot, strike, barrier, expiry, cover, rate, vol, div)
    carry = t["carry"]
    result = spot * torch.exp((carry - rate) * expiry)
    result = result * (
        _m(t["d1"], eta * t["e1"], eta * t["rho"])
        - t["hs_mu1"] * _m(t["f1"], eta * t["e3"], eta * t["rho"])
    )
    result = result - strike * torch.exp(-rate * expiry) * (
        _m(t["d2"], eta * t["e2"], eta * t["rho"])
        - t["hs_mu"] * _m(t["f2"], eta * t["e4"], eta * t["rho"])
    )
    return result


def _cob1(spot, strike, barrier, expiry, cover, rate, vol, div):
    t = _terms(spot, strike, barrier, expiry, cover, rate, vol, div)
    carry = t["carry"]
    growth = spot * torch.exp((carry - rate) * expiry)
    strike_df = strike * torch.exp(-rate * expiry)
    above = strike > barrier
    high = growth * (
        _m(t["d1"], t["e1"], t["rho"]) - t["hs_mu1"] * _m(t["f1"], -t["e3"], -t["rho"])
    ) - strike_df * (
        _m(t["d2"], t["e2"], t["rho"]) - t["hs_mu"] * _m(t["f2"], -t["e4"], -t["rho"])
    )
    low = growth * (_m(-t["g1"], -t["e1"], t["rho"]) - t["hs_mu1"] * _m(-t["g3"], t["e3"], -t["rho"]))
    low = low - strike_df * (_m(-t["g2"], -t["e2"], t["rho"]) - t["hs_mu"] * _m(-t["g4"], t["e4"], -t["rho"]))
    low = low - growth * (
        _m(-t["d1"], -t["e1"], t["rho"]) - t["hs_mu1"] * _m(-t["f1"], t["e3"], -t["rho"])
    )
    low = low + strike_df * (
        _m(-t["d2"], -t["e2"], t["rho"]) - t["hs_mu"] * _m(-t["f2"], t["e4"], -t["rho"])
    )
    low = low + growth * (_m(t["g1"], t["e1"], t["rho"]) - t["hs_mu1"] * _m(t["g3"], -t["e3"], -t["rho"]))
    low = low - strike_df * (_m(t["g2"], t["e2"], t["rho"]) - t["hs_mu"] * _m(t["g4"], -t["e4"], -t["rho"]))
    return torch.where(above, high, low)


def _cob2(spot, strike, barrier, expiry, cover, rate, vol, div, kind):
    if not torch.all(strike < barrier):
        raise ValueError("strike must be below the barrier for an end-B2 partial barrier")
    t = _terms(spot, strike, barrier, expiry, cover, rate, vol, div)
    carry = t["carry"]
    growth = spot * torch.exp((carry - rate) * expiry)
    strike_df = strike * torch.exp(-rate * expiry)
    if kind == "down_out":
        result = growth * (_m(t["g1"], t["e1"], t["rho"]) - t["hs_mu1"] * _m(t["g3"], -t["e3"], -t["rho"]))
        return result - strike_df * (
            _m(t["g2"], t["e2"], t["rho"]) - t["hs_mu"] * _m(t["g4"], -t["e4"], -t["rho"])
        )
    result = growth * (_m(-t["g1"], -t["e1"], t["rho"]) - t["hs_mu1"] * _m(-t["g3"], t["e3"], -t["rho"]))
    result = result - strike_df * (
        _m(-t["g2"], -t["e2"], t["rho"]) - t["hs_mu"] * _m(-t["g4"], t["e4"], -t["rho"])
    )
    result = result - growth * (
        _m(-t["d1"], -t["e1"], t["rho"]) - t["hs_mu1"] * _m(t["e3"], -t["f1"], -t["rho"])
    )
    result = result + strike_df * (
        _m(-t["d2"], -t["e2"], t["rho"]) - t["hs_mu"] * _m(t["e4"], -t["f2"], -t["rho"])
    )
    return result
