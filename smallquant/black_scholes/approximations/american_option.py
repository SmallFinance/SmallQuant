"""Closed-form approximations to American option prices under Black-Scholes.

Two families:

* Barone-Adesi and Whaley (1987): quadratic approximation of the early exercise
  premium, with the critical price found by Newton's method.
* Bjerksund and Stensland (1993, 2002): the value of a flat (1993) or two-step
  flat (2002) exercise boundary, which makes the price a lower bound.

References:
    G. Barone-Adesi and R. E. Whaley, "Efficient Analytic Approximation of American
    Option Values", Journal of Finance 42(2), 301-320, 1987.

    P. Bjerksund and G. Stensland, "Closed-Form Approximation of American Options",
    Scandinavian Journal of Management 9, S87-S99, 1993.

    P. Bjerksund and G. Stensland, "Closed Form Valuation of American Options",
    discussion paper, Norwegian School of Economics (NHH), 2002.

    E. G. Haug, "The Complete Guide to Option Pricing Formulas", 2nd ed., McGraw-Hill,
    2007, sections 3.1.3 to 3.1.5.
"""

from __future__ import annotations

import math

import torch

from smallquant._tensor import align, check
from smallquant.black_scholes.vanilla_prices import _flag, option_price
from smallquant.math.distributions import bivariate_west, normal_cdf, normal_pdf

__all__ = ["adesi_whaley", "bjerksund_stensland"]

# Placeholder rate and dividend for elements that never use the approximation, so the
# unused branch stays finite (and so do its gradients).
_SAFE_RATE = 0.05

# Bjerksund-Stensland 2002 splits the life of the option at the golden section.
_GOLDEN = 0.5 * (math.sqrt(5.0) - 1.0)


def _prepare(
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor | bool,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spot, strike, expiry, rate, vol, div = align(spot, strike, expiry, rate, vol, div, dtype=dtype)
    is_call = _flag(is_call, spot)
    check("spot", spot, spot > 0)
    check("strike", strike, strike > 0)
    check("vol", vol, vol > 0)
    check("expiry", expiry, expiry >= 0)
    return torch.broadcast_tensors(spot, strike, expiry, rate, vol, div, is_call)


def _limit(price: torch.Tensor, intrinsic: torch.Tensor, degenerate: torch.Tensor) -> torch.Tensor:
    """At zero expiry the price is the intrinsic value.

    At non-positive vol (outside the documented domain) the larger of the intrinsic
    value and the European limit is returned, a finite lower bound.
    """
    return torch.where(degenerate, torch.maximum(price, intrinsic), price)


def adesi_whaley(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor | bool,
    max_iter: int = 20,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """American option price by the Barone-Adesi and Whaley (1987) approximation.

    The early exercise premium solves the Black-Scholes PDE with the time-derivative
    term scaled away, giving ``A * (S / S*)**q`` above the European price. The
    critical price ``S*`` solves the value-matching condition (BAW 1987, eqs. 24 and
    27 for calls, 26 and 29 for puts) by Newton's method from the seed of section
    II.B. Newton runs ``max_iter`` steps under ``torch.no_grad``, then one step
    with gradients on from the detached root, so derivatives of ``S*`` are the
    implicit-function-theorem ones rather than an unrolled solver.

    Where early exercise is never optimal (a call with ``div <= 0``, a put with
    ``rate <= 0``) the European price is returned.

    Args:
        spot: underlying spot, positive
        strike: strike, positive
        expiry: time to expiry in years, non-negative
        rate: continuously compounded risk-free rate
        vol: lognormal volatility, positive
        div: continuous dividend yield
        is_call: bool or bool tensor, true for a call
        max_iter: Newton steps for the critical price, run without gradients
        dtype: optional dtype for the computation

    Returns:
        American price tensor, broadcast from the inputs. At ``expiry == 0`` it is the
        intrinsic value.

    Example:
        >>> one = torch.tensor(1.0, dtype=torch.float64)
        >>> adesi_whaley(spot=100 * one, strike=100 * one, expiry=0.25 * one,
        ...              rate=0.08 * one, vol=0.2 * one, div=0.12 * one, is_call=False)
    """
    spot, strike, expiry, rate, vol, div, is_call = _prepare(
        spot, strike, expiry, rate, vol, div, is_call, dtype
    )
    one = torch.ones_like(spot)
    sign = torch.where(is_call, one, -one)
    european = option_price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=is_call
    )
    intrinsic = torch.relu(sign * (spot - strike))

    degenerate = (expiry <= 0) | (vol <= 0)
    never_early = torch.where(is_call, div <= 0, rate <= 0)
    plain = degenerate | never_early

    # Elements that skip the approximation get placeholder inputs so it stays finite.
    t = torch.where(degenerate, one, expiry)
    sigma = torch.where(degenerate, one, vol)
    r = torch.where(plain, torch.full_like(rate, _SAFE_RATE), rate)
    q = torch.where(plain, torch.full_like(div, _SAFE_RATE), div)

    variance = sigma * sigma
    sqrt_t = torch.sqrt(t)
    std_dev = sigma * sqrt_t
    carry = r - q
    n = 2.0 * carry / variance  # BAW's N
    m = 2.0 * r / variance  # BAW's M
    # M / K with K = 1 - exp(-rT), written as 2 / (sigma^2 T) * rT / (1 - exp(-rT))
    # so that it has the right limit at r = 0.
    rt = r * t
    small = rt.abs() < 1e-6
    safe_rt = torch.where(small, one, rt)
    ratio = torch.where(small, 1.0 + 0.5 * rt + rt * rt / 12.0, safe_rt / -torch.expm1(-safe_rt))
    m_over_k = 2.0 / (variance * t) * ratio
    q_exp = 0.5 * (1.0 - n + sign * torch.sqrt((n - 1.0) ** 2 + 4.0 * m_over_k))
    div_discount = torch.exp(-q * t)

    def d1(s: torch.Tensor) -> torch.Tensor:
        return (torch.log(s / strike) + (carry + 0.5 * variance) * t) / std_dev

    def premium_factor(s: torch.Tensor) -> torch.Tensor:
        # 1 - exp(-qT) N(sign d1(S)), the bracket in BAW eqs. 25 and 28.
        return 1.0 - div_discount * normal_cdf(sign * d1(s))

    def residual(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Value matching f(S) = 0 and its derivative in S."""
        euro = option_price(spot=s, strike=strike, expiry=t, rate=r, vol=sigma, div=q, is_call=is_call)
        x = d1(s)
        tail = div_discount * normal_cdf(sign * x)
        f = euro + sign * (1.0 - tail) * s / q_exp - sign * (s - strike)
        slope = sign * tail + sign * (1.0 - tail) / q_exp - div_discount * normal_pdf(x) / (std_dev * q_exp) - sign
        return f, slope

    with torch.no_grad():
        # Seed from BAW section II.B: interpolate between the strike and the perpetual
        # critical price S_inf.
        q_inf = 0.5 * (1.0 - n + sign * torch.sqrt(torch.clamp((n - 1.0) ** 2 + 4.0 * m, min=0.0)))
        s_inf = strike / (1.0 - 1.0 / q_inf)
        h = -(sign * carry * t + 2.0 * std_dev) * sign * strike / (s_inf - strike)
        s = torch.where(
            is_call,
            strike + (s_inf - strike) * (-torch.expm1(h)),
            s_inf + (strike - s_inf) * torch.exp(h),
        )
        floor = strike * 1e-12
        for _ in range(max_iter):
            f, slope = residual(s)
            s = torch.maximum(s - f / slope, floor)

    s0 = s.detach()
    f, slope = residual(s0)
    s_crit = s0 - f / slope

    coefficient = sign * s_crit / q_exp * premium_factor(s_crit)
    continuation = european + coefficient * (spot / s_crit) ** q_exp
    approx = torch.where(sign * (s_crit - spot) > 0, continuation, sign * (spot - strike))

    price = torch.where(plain, european, approx)
    return _limit(price, intrinsic, degenerate)


def bjerksund_stensland(
    *,
    spot: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    vol: torch.Tensor,
    div: torch.Tensor,
    is_call: torch.Tensor | bool,
    version: int = 1993,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """American option price by the Bjerksund-Stensland approximation.

    ``version=1993`` values the call with a single flat exercise boundary
    (Bjerksund and Stensland 1993; Haug 2007, section 3.1.4). ``version=2002``
    splits the life of the option at ``t1 = (sqrt(5) - 1) / 2 * T`` and uses one
    flat boundary on each piece, which needs the bivariate normal (Bjerksund and
    Stensland 2002; Haug 2007, section 3.1.5). Either is the value of a feasible
    exercise policy, so a lower bound on the American price.

    Puts use the Bjerksund-Stensland put-call transformation
    ``P(S, K, T, r, q, sigma) = C(K, S, T, q, r, sigma)``: spot swaps with strike
    and rate swaps with dividend yield. Where early exercise is never optimal (a
    call with ``div <= 0``, a put with ``rate <= 0``) the European price is
    returned. (With a negative rate a call can be worth exercising early even at
    ``div <= 0``, a two-boundary case these formulas do not cover; the European
    price is then a lower bound.) Otherwise the result is the largest of the
    approximation, the European price and the intrinsic value, all lower bounds, as
    in QuantLib's ``BjerksundStenslandApproximationEngine``.

    The 2002 triggers follow the paper: the boundary for ``[t1, T]`` is set at its
    remaining life ``T - t1``. Haug (2007) prints ``t1`` there instead, which gives a
    slightly lower price.

    Args:
        spot: underlying spot, positive
        strike: strike, positive
        expiry: time to expiry in years, non-negative
        rate: continuously compounded risk-free rate
        vol: lognormal volatility, positive
        div: continuous dividend yield
        is_call: bool or bool tensor, true for a call
        version: 1993 or 2002
        dtype: optional dtype for the computation

    Returns:
        American price tensor, broadcast from the inputs. At ``expiry == 0`` it is the
        intrinsic value.

    Example:
        >>> one = torch.tensor(1.0, dtype=torch.float64)
        >>> bjerksund_stensland(spot=100 * one, strike=100 * one, expiry=0.25 * one,
        ...                     rate=0.08 * one, vol=0.2 * one, div=0.12 * one,
        ...                     is_call=True, version=2002)
    """
    if version not in (1993, 2002):
        raise ValueError("version must be 1993 or 2002")
    spot, strike, expiry, rate, vol, div, is_call = _prepare(
        spot, strike, expiry, rate, vol, div, is_call, dtype
    )
    one = torch.ones_like(spot)
    european = option_price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=is_call
    )
    intrinsic = torch.relu(torch.where(is_call, spot - strike, strike - spot))

    # Put-call transformation onto an equivalent call.
    s = torch.where(is_call, spot, strike)
    k = torch.where(is_call, strike, spot)
    r = torch.where(is_call, rate, div)
    q = torch.where(is_call, div, rate)

    degenerate = (expiry <= 0) | (vol <= 0)
    plain = degenerate | (q <= 0)
    t = torch.where(degenerate, one, expiry)
    sigma = torch.where(degenerate, one, vol)
    r = torch.where(plain, torch.full_like(r, _SAFE_RATE), r)
    q = torch.where(plain, torch.full_like(q, _SAFE_RATE), q)

    if version == 1993:
        call = _call_1993(s, k, t, r, r - q, sigma)
    else:
        call = _call_2002(s, k, t, r, r - q, sigma)
    # A flat-boundary policy is one feasible policy; so are exercising now and holding to
    # expiry, and the price is the best of the three lower bounds.
    call = torch.maximum(call, torch.maximum(european, intrinsic))
    price = torch.where(plain, european, call)
    return _limit(price, intrinsic, degenerate)


def _beta_and_bounds(
    k: torch.Tensor, r: torch.Tensor, b: torch.Tensor, variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exponent beta, perpetual boundary B_inf and short-maturity boundary B_0."""
    drift = b / variance - 0.5
    beta = -drift + torch.sqrt(drift * drift + 2.0 * r / variance)
    b_inf = beta / (beta - 1.0) * k
    b_zero = torch.maximum(k, r / (r - b) * k)
    return beta, b_inf, b_zero


def _trigger(
    tau: torch.Tensor,
    b: torch.Tensor,
    sigma: torch.Tensor,
    b_inf: torch.Tensor,
    b_zero: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Flat exercise boundary B0 + (B_inf - B0)(1 - exp(h(tau))), with h scaled by ``scale``."""
    h = -(b * tau + 2.0 * sigma * torch.sqrt(tau)) * scale
    return b_zero - (b_inf - b_zero) * torch.expm1(h)


def _phi(
    s: torch.Tensor,
    t: torch.Tensor,
    gamma: torch.Tensor | float,
    h: torch.Tensor,
    x: torch.Tensor,
    r: torch.Tensor,
    b: torch.Tensor,
    variance: torch.Tensor,
    std_dev: torch.Tensor,
) -> torch.Tensor:
    """phi(S, T, gamma, H, X) of Bjerksund and Stensland (1993)."""
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * variance) * t
    kappa = 2.0 * b / variance + 2.0 * gamma - 1.0
    log_xs = torch.log(x / s)
    d = -(torch.log(s / h) + (b + (gamma - 0.5) * variance) * t) / std_dev
    return torch.exp(lam) * s**gamma * (normal_cdf(d) - torch.exp(kappa * log_xs) * normal_cdf(d - 2.0 * log_xs / std_dev))


def _call_1993(
    s: torch.Tensor, k: torch.Tensor, t: torch.Tensor, r: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    variance = sigma * sigma
    std_dev = sigma * torch.sqrt(t)
    beta, b_inf, b_zero = _beta_and_bounds(k, r, b, variance)
    trigger = _trigger(t, b, sigma, b_inf, b_zero, b_zero / (b_inf - b_zero))
    alpha = (trigger - k) * trigger ** (-beta)

    def phi(gamma: torch.Tensor | float, h: torch.Tensor) -> torch.Tensor:
        return _phi(s, t, gamma, h, trigger, r, b, variance, std_dev)

    value = (
        alpha * s**beta
        - alpha * phi(beta, trigger)
        + phi(1.0, trigger)
        - phi(1.0, k)
        - k * phi(0.0, trigger)
        + k * phi(0.0, k)
    )
    return torch.where(s >= trigger, s - k, value)


def _psi(
    s: torch.Tensor,
    big_t: torch.Tensor,
    gamma: torch.Tensor | float,
    h: torch.Tensor,
    x2: torch.Tensor,
    x1: torch.Tensor,
    t1: torch.Tensor,
    r: torch.Tensor,
    b: torch.Tensor,
    variance: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """psi(S, T, gamma, H, X2, X1, t1) of Bjerksund and Stensland (2002)."""
    mu = b + (gamma - 0.5) * variance
    sd1 = torch.sqrt(variance * t1)
    sd = torch.sqrt(variance * big_t)
    log_s_x1 = torch.log(s / x1)
    log_e2 = torch.log(x2 * x2 / (s * x1))
    e1 = (log_s_x1 + mu * t1) / sd1
    e2 = (log_e2 + mu * t1) / sd1
    e3 = (log_s_x1 - mu * t1) / sd1
    e4 = (log_e2 - mu * t1) / sd1
    f1 = (torch.log(s / h) + mu * big_t) / sd
    f2 = (torch.log(x2 * x2 / (s * h)) + mu * big_t) / sd
    f3 = (torch.log(x1 * x1 / (s * h)) + mu * big_t) / sd
    f4 = (torch.log(s * x1 * x1 / (h * x2 * x2)) + mu * big_t) / sd
    kappa = 2.0 * b / variance + 2.0 * gamma - 1.0
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * variance

    # The four bivariate terms in one call, stacked on a trailing axis.
    a = -torch.stack(torch.broadcast_tensors(e1, e2, e3, e4), dim=-1)
    c = -torch.stack(torch.broadcast_tensors(f1, f2, f3, f4), dim=-1)
    m = bivariate_west(a, c, rho.expand_as(a))
    weights = torch.stack(
        torch.broadcast_tensors(
            torch.ones_like(log_s_x1),
            -torch.exp(kappa * torch.log(x2 / s)),
            -torch.exp(kappa * torch.log(x1 / s)),
            torch.exp(kappa * torch.log(x1 / x2)),
        ),
        dim=-1,
    )
    return torch.exp(lam * big_t) * s**gamma * (weights * m).sum(dim=-1)


def _call_2002(
    s: torch.Tensor, k: torch.Tensor, t: torch.Tensor, r: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    variance = sigma * sigma
    t1 = _GOLDEN * t
    std_dev1 = sigma * torch.sqrt(t1)
    beta, b_inf, b_zero = _beta_and_bounds(k, r, b, variance)
    scale = k * k / ((b_inf - b_zero) * b_zero)
    # x1 is the flat boundary on [t1, T], set at its remaining life T - t1; x2 covers
    # [0, t1] and is set at the full life T.
    x1 = _trigger(t - t1, b, sigma, b_inf, b_zero, scale)
    x2 = _trigger(t, b, sigma, b_inf, b_zero, scale)
    alpha1 = (x1 - k) * x1 ** (-beta)
    alpha2 = (x2 - k) * x2 ** (-beta)
    sqrt_golden = math.sqrt(_GOLDEN)
    rho = torch.tensor(
        [sqrt_golden, sqrt_golden, -sqrt_golden, -sqrt_golden], dtype=s.dtype, device=s.device
    )

    def phi(gamma: torch.Tensor | float, h: torch.Tensor) -> torch.Tensor:
        return _phi(s, t1, gamma, h, x2, r, b, variance, std_dev1)

    def psi(gamma: torch.Tensor | float, h: torch.Tensor) -> torch.Tensor:
        return _psi(s, t, gamma, h, x2, x1, t1, r, b, variance, rho)

    value = (
        alpha2 * s**beta
        - alpha2 * phi(beta, x2)
        + phi(1.0, x2)
        - phi(1.0, x1)
        - k * phi(0.0, x2)
        + k * phi(0.0, x1)
        + alpha1 * phi(beta, x1)
        - alpha1 * psi(beta, x1)
        + psi(1.0, x1)
        - psi(1.0, k)
        - k * psi(0.0, x1)
        + k * psi(0.0, k)
    )
    return torch.where(s >= x2, s - k, value)
