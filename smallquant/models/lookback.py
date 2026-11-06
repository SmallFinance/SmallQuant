"""Continuous fixed and floating lookback options."""

from __future__ import annotations

import torch

from ._common import as_tensors, bivariate_west, norm_cdf


def _lambda(rate, div, vol):
    return 2.0 * (rate - div) / vol**2


def fixed_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    minimum: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Continuous fixed-strike lookback.

    ``minimum`` is the realized minimum so far for a call and the realized
    maximum so far for a put.
    """
    spot, strike, minimum, expiry, rate, vol, div = as_tensors(
        spot, strike, minimum, expiry, rate, vol, div
    )
    std = vol * torch.sqrt(expiry)
    lam = _lambda(rate, div, vol)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)

    def part(eta, level):
        ratio = spot / level
        d1 = torch.log(ratio) / std + 0.5 * (lam + 1.0) * std
        pow_ss = ratio ** (-lam)
        return eta * (
            spot * div_df * norm_cdf(eta * d1)
            - level * rate_df * norm_cdf(eta * (d1 - std))
            - spot
            * rate_df
            * (pow_ss * norm_cdf(eta * (d1 - lam * std)) - div_df * norm_cdf(eta * d1) / rate_df)
            / lam
        )

    call = torch.where(strike <= minimum, part(1.0, minimum) + rate_df * (minimum - strike), part(1.0, strike))
    put = torch.where(
        strike >= minimum, part(-1.0, minimum) + rate_df * (strike - minimum), part(-1.0, strike)
    )
    return torch.where(is_call, call, put)


def floating_price(
    *,
    spot: torch.Tensor,
    extremum: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Continuous floating-strike lookback.

    ``extremum`` is the realized minimum for a call and the realized maximum for a put.
    """
    spot, extremum, expiry, rate, vol, div = as_tensors(spot, extremum, expiry, rate, vol, div)
    std = vol * torch.sqrt(expiry)
    lam = _lambda(rate, div, vol)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    ratio = spot / extremum
    d1 = torch.log(ratio) / std + 0.5 * (lam + 1.0) * std
    pow_s = ratio ** (-lam)

    def body(eta):
        return eta * (
            (spot * div_df * norm_cdf(eta * d1) - extremum * rate_df * norm_cdf(eta * (d1 - std)))
            + spot
            * rate_df
            * (
                pow_s * norm_cdf(eta * (-d1 + lam * std))
                - div_df * norm_cdf(eta * -d1) / rate_df
            )
            / lam
        )

    return torch.where(is_call, body(1.0), body(-1.0))


def partial_fixed_price(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    lookback_start: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Partial-time continuous fixed-strike lookback."""
    spot, strike, expiry, lookback_start, rate, vol, div = as_tensors(
        spot, strike, expiry, lookback_start, rate, vol, div
    )
    std = vol * torch.sqrt(expiry)
    carry = rate - div
    x = 2.0 * carry / vol**2
    ratio = spot / strike
    d1 = torch.log(ratio) / std + 0.5 * (x + 1.0) * std
    d2 = d1 - std
    tau = expiry - lookback_start
    e1 = (carry + 0.5 * vol * vol) * tau / (vol * torch.sqrt(tau))
    e2 = e1 - vol * torch.sqrt(tau)
    f1 = (torch.log(ratio) + (carry + 0.5 * vol * vol) * lookback_start) / (
        vol * torch.sqrt(lookback_start)
    )
    f2 = f1 - vol * torch.sqrt(lookback_start)
    rho1 = -torch.sqrt(lookback_start / expiry)
    rho2 = torch.sqrt(tau / expiry)
    rho3 = -rho2
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)

    def body(eta):
        n1 = norm_cdf(eta * d1)
        n2 = norm_cdf(eta * d2)
        n3 = bivariate_west(
            eta * (d1 - x * std),
            eta * (-f1 + 2.0 * carry * torch.sqrt(lookback_start) / vol),
            rho1,
        )
        n4 = bivariate_west(eta * e1, eta * d1, rho2)
        n5 = bivariate_west(-eta * e1, eta * d1, rho3)
        n6 = bivariate_west(eta * f2, -eta * d2, rho1)
        n7 = norm_cdf(eta * f1)
        n8 = norm_cdf(-eta * e2)
        pow_s = ratio ** (-x)
        carry_df = torch.exp(-carry * tau)
        return eta * (
            spot * div_df * n1
            - strike * rate_df * n2
            + spot * rate_df / x * (-pow_s * n3 + div_df / rate_df * n4)
            - spot * div_df * n5
            - strike * rate_df * n6
            + carry_df * div_df * (1.0 - 0.5 * vol * vol / carry) * spot * n7 * n8
        )

    return torch.where(is_call, body(1.0), body(-1.0))


def partial_floating_price(
    *,
    spot: torch.Tensor,
    extremum: torch.Tensor,
    level: torch.Tensor,
    expiry: torch.Tensor,
    lookback_end: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Partial-time continuous floating-strike lookback.

    ``level`` is the lambda strike fraction. ``lookback_end`` is the end of the
    lookback window, at or before ``expiry``.
    """
    spot, extremum, level, expiry, lookback_end, rate, vol, div = as_tensors(
        spot, extremum, level, expiry, lookback_end, rate, vol, div
    )
    std = vol * torch.sqrt(expiry)
    carry = rate - div
    x = 2.0 * carry / vol**2
    ratio = spot / extremum
    d1 = torch.log(ratio) / std + 0.5 * (x + 1.0) * std
    d2 = d1 - std
    full = torch.isclose(lookback_end, expiry)
    tau = torch.clamp(expiry - lookback_end, min=1e-12)
    e1 = (carry + 0.5 * vol * vol) * tau / (vol * torch.sqrt(tau))
    e2 = e1 - vol * torch.sqrt(tau)
    f1 = (torch.log(ratio) + (carry + 0.5 * vol * vol) * lookback_end) / (
        vol * torch.sqrt(lookback_end)
    )
    f2 = f1 - vol * torch.sqrt(lookback_end)
    g1 = torch.log(level) / vol / torch.sqrt(expiry)
    div_df = torch.exp(-div * expiry)
    rate_df = torch.exp(-rate * expiry)
    rho1 = torch.sqrt(lookback_end / expiry)
    rho2 = -torch.sqrt(tau / expiry)
    rho3 = -rho1

    def body(eta):
        n1 = norm_cdf(eta * (d1 - g1))
        n2 = norm_cdf(eta * (d2 - g1))
        n3 = bivariate_west(
            eta * (-f1 + 2.0 * carry * torch.sqrt(lookback_end) / vol),
            eta * (-d1 + x * std - g1),
            rho1,
        )
        g2 = torch.log(level) / vol / torch.sqrt(tau)
        n4 = bivariate_west(-eta * (d1 + g1), eta * (e1 + g2), rho2)
        n5 = bivariate_west(-eta * (d1 - g1), eta * (e1 - g2), rho2)
        n6 = bivariate_west(eta * -f2, eta * (d2 - g1), rho3)
        n7 = norm_cdf(eta * (e2 - g2))
        n8 = norm_cdf(-eta * f1)
        n4_full = norm_cdf(-eta * (d1 + g1))
        pow_s = ratio ** (-x)
        pow_l = level**x
        partial = eta * (
            spot * div_df * n1
            - level * extremum * rate_df * n2
            + spot * rate_df * level / x * (pow_s * n3 - div_df / rate_df * pow_l * n4)
            + spot * div_df * n5
            + rate_df * level * extremum * n6
            - torch.exp(-carry * (expiry - lookback_end))
            * div_df
            * (1.0 + 0.5 * vol * vol / carry)
            * level
            * spot
            * n7
            * n8
        )
        complete = eta * (
            spot * div_df * n1
            - level * extremum * rate_df * n2
            + spot * rate_df * level / x * (pow_s * n3 - div_df / rate_df * pow_l * n4_full)
        )
        return torch.where(full, complete, partial)

    return torch.where(is_call, body(1.0), body(-1.0))
