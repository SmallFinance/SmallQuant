"""European vanilla, binary, asset-or-nothing and single-barrier prices.

``option_price`` covers three models in one call, as TF Quant Finance does:
Black-Scholes when ``spot`` is passed, Black-76 when ``forward`` is passed, and
Bachelier when ``normal_vol`` is passed instead of ``vol``.
"""

from __future__ import annotations

import torch

from smallquant._tensor import align, check
from smallquant.math.distributions import normal_cdf

from ._common import as_tensors, bachelier_price, black_price, norm_cdf

__all__ = ["asset_or_nothing_price", "barrier_price", "binary_price", "option_price"]

_KINDS = ("down_in", "up_in", "down_out", "up_out")


def _flag(is_call: torch.Tensor | bool, like: torch.Tensor) -> torch.Tensor:
    if isinstance(is_call, bool):
        return torch.tensor(is_call, device=like.device)
    return is_call.to(device=like.device)


def _market(
    *,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    volatility: torch.Tensor,
    spot: torch.Tensor | None,
    forward: torch.Tensor | None,
    rate: torch.Tensor | None,
    div: torch.Tensor | None,
    discount: torch.Tensor | None,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align inputs and resolve the spot-or-forward choice into a forward and a discount factor.

    ``spot`` needs ``rate`` (``div`` defaults to zero). ``forward`` needs ``discount`` or
    ``rate`` and takes no ``div``, since a forward already carries it.
    """
    if (spot is None) == (forward is None):
        raise ValueError("pass exactly one of spot and forward")
    if rate is not None and discount is not None:
        raise ValueError("pass at most one of rate and discount")
    if spot is not None:
        if rate is None:
            raise ValueError("spot needs rate")
        if div is None:
            spot, strike, expiry, volatility, rate = align(spot, strike, expiry, volatility, rate, dtype=dtype)
            div = torch.zeros_like(rate)
        else:
            spot, strike, expiry, volatility, rate, div = align(
                spot, strike, expiry, volatility, rate, div, dtype=dtype
            )
        return spot * torch.exp((rate - div) * expiry), strike, expiry, volatility, torch.exp(-rate * expiry)
    if div is not None:
        raise ValueError("div applies to spot only; a forward already includes carry")
    if rate is None and discount is None:
        raise ValueError("forward needs discount or rate")
    carry = rate if rate is not None else discount
    forward, strike, expiry, volatility, carry = align(forward, strike, expiry, volatility, carry, dtype=dtype)
    discount = torch.exp(-carry * expiry) if rate is not None else carry
    return forward, strike, expiry, volatility, discount


def _one_vol(vol: torch.Tensor | None, normal_vol: torch.Tensor | None) -> None:
    if (vol is None) == (normal_vol is None):
        raise ValueError("pass exactly one of vol and normal_vol")


def option_price(
    *,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    is_call: torch.Tensor | bool,
    vol: torch.Tensor | None = None,
    normal_vol: torch.Tensor | None = None,
    spot: torch.Tensor | None = None,
    forward: torch.Tensor | None = None,
    rate: torch.Tensor | None = None,
    div: torch.Tensor | None = None,
    discount: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """European option price under Black-Scholes, Black-76 or Bachelier.

    Args:
        strike: strike
        expiry: time to expiry in years
        is_call: boolean tensor, true for a call and false for a put
        vol: lognormal volatility (Black-Scholes, Black-76)
        normal_vol: normal volatility of the forward (Bachelier)
        spot: underlying spot; needs ``rate``, optional ``div``
        forward: forward to expiry; needs ``discount`` or ``rate``
        rate: continuously compounded risk-free rate
        div: continuous dividend / carry yield, with ``spot`` only
        discount: zero-bond price to expiry, with ``forward`` only

    Returns:
        Option price tensor, broadcast from the inputs.

    Pass exactly one of ``vol`` and ``normal_vol`` and exactly one of ``spot`` and
    ``forward``. With ``spot`` and ``vol``, at ``expiry <= 0`` or ``vol <= 0`` the price
    is its limit, the discounted forward intrinsic value, with finite gradients.

    Example:
        >>> one = torch.tensor(1.0, dtype=torch.float64)
        >>> option_price(spot=100 * one, strike=100 * one, expiry=one, rate=0.05 * one,
        ...              vol=0.2 * one, is_call=True)
    """
    _one_vol(vol, normal_vol)
    if spot is not None and vol is not None and discount is None and forward is None:
        if rate is None:
            raise ValueError("spot needs rate")
        return _black_scholes(spot, strike, expiry, rate, vol, div, is_call, dtype)
    volatility = vol if vol is not None else normal_vol
    forward, strike, expiry, volatility, discount = _market(
        strike=strike, expiry=expiry, volatility=volatility, spot=spot, forward=forward,
        rate=rate, div=div, discount=discount, dtype=dtype,
    )
    is_call = _flag(is_call, forward)
    std_dev = volatility * torch.sqrt(expiry)
    if vol is not None:
        return black_price(forward, strike, std_dev, discount, is_call)
    return bachelier_price(forward, strike, std_dev, discount, is_call)


def _black_scholes(
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor | None,
    is_call: torch.Tensor | bool,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    if div is None:
        spot, strike, expiry, rate, vol = align(spot, strike, expiry, rate, vol, dtype=dtype)
        div = torch.zeros_like(rate)
    else:
        spot, strike, expiry, rate, vol, div = align(spot, strike, expiry, rate, vol, div, dtype=dtype)
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


def binary_price(
    *,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    is_call: torch.Tensor | bool,
    vol: torch.Tensor | None = None,
    normal_vol: torch.Tensor | None = None,
    spot: torch.Tensor | None = None,
    forward: torch.Tensor | None = None,
    rate: torch.Tensor | None = None,
    div: torch.Tensor | None = None,
    discount: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Cash-or-nothing price: pays one unit at expiry if the option ends in the money.

    Takes the same inputs as ``option_price``. A call pays when the terminal price is
    above ``strike``, a put when it is below. With zero volatility or expiry the price is
    the discounted indicator of the forward against the strike.
    """
    _one_vol(vol, normal_vol)
    volatility = vol if vol is not None else normal_vol
    forward, strike, expiry, volatility, discount = _market(
        strike=strike, expiry=expiry, volatility=volatility, spot=spot, forward=forward,
        rate=rate, div=div, discount=discount, dtype=dtype,
    )
    is_call = _flag(is_call, forward)
    sign = torch.where(is_call, torch.ones_like(forward), -torch.ones_like(forward))
    std_dev = volatility * torch.sqrt(torch.clamp(expiry, min=0.0))
    degenerate = std_dev <= 0
    safe = torch.where(degenerate, torch.ones_like(std_dev), std_dev)
    if vol is not None:
        moneyness = torch.log(forward / strike) / safe - 0.5 * safe
    else:
        moneyness = (forward - strike) / safe
    limit = (sign * (forward - strike) > 0).to(forward.dtype)
    probability = torch.where(degenerate, limit, norm_cdf(sign * moneyness))
    return discount * probability


def asset_or_nothing_price(
    *,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    vol: torch.Tensor,
    is_call: torch.Tensor | bool,
    spot: torch.Tensor | None = None,
    forward: torch.Tensor | None = None,
    rate: torch.Tensor | None = None,
    div: torch.Tensor | None = None,
    discount: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Asset-or-nothing price under Black-Scholes: pays the asset at expiry if in the money.

    Lognormal only. Takes ``spot`` or ``forward`` as in ``option_price``.
    """
    forward, strike, expiry, vol, discount = _market(
        strike=strike, expiry=expiry, volatility=vol, spot=spot, forward=forward,
        rate=rate, div=div, discount=discount, dtype=dtype,
    )
    is_call = _flag(is_call, forward)
    sign = torch.where(is_call, torch.ones_like(forward), -torch.ones_like(forward))
    std_dev = vol * torch.sqrt(torch.clamp(expiry, min=0.0))
    degenerate = std_dev <= 0
    safe = torch.where(degenerate, torch.ones_like(std_dev), std_dev)
    d1 = torch.log(forward / strike) / safe + 0.5 * safe
    limit = (sign * (forward - strike) > 0).to(forward.dtype)
    probability = torch.where(degenerate, limit, norm_cdf(sign * d1))
    return discount * forward * probability


def _n(x: torch.Tensor) -> torch.Tensor:
    return norm_cdf(x)


def _guard(power: torch.Tensor, nval: torch.Tensor) -> torch.Tensor:
    return torch.where(nval == 0, torch.zeros_like(power), power * nval)

def barrier_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    barrier: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    rebate: torch.Tensor,
    is_call: torch.Tensor | bool,
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
    is_call = _flag(is_call, spot)
    std = vol * torch.sqrt(expiry)
    mu = (rate - div) / vol**2 - 0.5
    mu_sigma = (1.0 + mu) * std
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    hs = barrier / spot
    pow0 = hs ** (2.0 * mu)
    pow1 = pow0 * hs * hs

    def A(phi: float) -> torch.Tensor:
        x1 = torch.log(spot / strike) / std + mu_sigma
        return phi * (spot * div_df * _n(phi * x1) - strike * rate_df * _n(phi * (x1 - std)))

    def B(phi: float) -> torch.Tensor:
        x2 = torch.log(spot / barrier) / std + mu_sigma
        return phi * (spot * div_df * _n(phi * x2) - strike * rate_df * _n(phi * (x2 - std)))

    def C(eta: float, phi: float) -> torch.Tensor:
        y1 = torch.log(barrier * hs / strike) / std + mu_sigma
        return phi * (
            spot * div_df * _guard(pow1, _n(eta * y1))
            - strike * rate_df * _guard(pow0, _n(eta * (y1 - std)))
        )

    def D(eta: float, phi: float) -> torch.Tensor:
        y2 = torch.log(barrier / spot) / std + mu_sigma
        return phi * (
            spot * div_df * _guard(pow1, _n(eta * y2))
            - strike * rate_df * _guard(pow0, _n(eta * (y2 - std)))
        )

    def E(eta: float) -> torch.Tensor:
        x2 = torch.log(spot / barrier) / std + mu_sigma
        y2 = torch.log(barrier / spot) / std + mu_sigma
        return rebate * rate_df * (
            _n(eta * (x2 - std)) - _guard(pow0, _n(eta * (y2 - std)))
        )

    def F(eta: float) -> torch.Tensor:
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
