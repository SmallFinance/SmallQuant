"""Hull-White one-factor bonds, options and an exact Gaussian step.

Brigo and Mercurio, *Interest Rate Models*. The curve is a
:class:`smallquant.rates.curve.DiscountCurve`. Year fractions only.
"""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.distributions import normal_cdf


def bond_price(curve, short_rate, a, sigma, now, maturity):
    """Zero bond ``P(now, maturity)`` given the short rate at ``now``."""
    short_rate, a, sigma, now, maturity = align(short_rate, a, sigma, now, maturity, curve.times)[:5]
    b = _b(a, now, maturity)
    log_a = _log_a(curve, a, sigma, now, maturity, b)
    return torch.exp(log_a - b * short_rate)


def zero_bond_option(curve, a, sigma, expiry, maturity, strike, *, is_call=True):
    """European option on a zero bond, paid at ``expiry``."""
    a, sigma, expiry, maturity, strike = align(a, sigma, expiry, maturity, strike, curve.times)[:5]
    sigma_p = _bond_volatility(a, sigma, expiry, maturity)
    forward = curve.discount(maturity) / curve.discount(expiry).clamp(min=1e-18)
    d1 = (torch.log(forward / strike) + 0.5 * sigma_p**2) / sigma_p.clamp(min=1e-12)
    d2 = d1 - sigma_p
    call = curve.discount(maturity) * normal_cdf(d1) - strike * curve.discount(expiry) * normal_cdf(d2)
    put = strike * curve.discount(expiry) * normal_cdf(-d2) - curve.discount(maturity) * normal_cdf(-d1)
    flag = is_call if isinstance(is_call, torch.Tensor) else torch.tensor(is_call, device=call.device)
    intrinsic = curve.discount(expiry) * torch.where(flag, torch.relu(forward - strike), torch.relu(strike - forward))
    price = torch.where(flag, call, put)
    return torch.where(sigma_p > 1e-10, price, intrinsic)


def caplet(curve, a, sigma, start, end, strike):
    """Caplet paying ``(end-start) * max(L-strike, 0)`` at ``end``.

    ``L`` is the simply compounded forward from ``start`` to ``end``.
    """
    start, end, strike, a, sigma = align(start, end, strike, a, sigma, curve.times)[:5]
    delta = end - start
    bond_strike = 1.0 / (1.0 + delta * strike)
    # A put on the zero bond maturing at ``end``, struck at bond_strike, times (1+delta K).
    put = zero_bond_option(curve, a, sigma, start, end, bond_strike, is_call=False)
    return (1.0 + delta * strike) * put


def swaption(curve, a, sigma, expiry, pay_times, deltas, strike, *, is_payer=True):
    """Jamshidian decomposition of a European payer or receiver swaption.

    ``pay_times`` are the fixed payment times after ``expiry``. ``deltas`` are
    the accrual fractions. The last payment includes the notional.
    """
    a, sigma, expiry, strike = align(a, sigma, expiry, strike, curve.times)[:4]
    pay_times, deltas = align(pay_times, deltas, curve.times)[:2]
    coupon = deltas * strike
    coupon = coupon.clone()
    coupon[-1] = coupon[-1] + 1.0
    root = _critical_rate(curve, a, sigma, expiry, pay_times, coupon)
    level_strike = bond_price(curve, root, a, sigma, expiry, pay_times)
    if is_payer:
        pieces = coupon * zero_bond_option(curve, a, sigma, expiry, pay_times, level_strike, is_call=True)
    else:
        pieces = coupon * zero_bond_option(curve, a, sigma, expiry, pay_times, level_strike, is_call=False)
    return pieces.sum()


def step(*, short_rate, times, normals, a, sigma, curve):
    """Exact Gaussian short-rate and integrated-rate paths.

    The result has shape ``[num_paths, num_steps + 1, 2]``. Column 0 is the
    short rate and column 1 is the integral of the short rate from time 0,
    so ``exp(-column 1)`` is the stochastic discount.
    """
    short_rate, times, normals, a, sigma = align(short_rate, times, normals, a, sigma, curve.times)[:5]
    num_paths, num_steps, _ = normals.shape
    rate = short_rate.reshape(-1)[0].expand(num_paths).clone()
    integral = torch.zeros(num_paths, dtype=times.dtype, device=times.device)
    rates = [rate]
    integrals = [integral]
    for index in range(num_steps):
        start = times[index]
        end = times[index + 1]
        z_rate = normals[:, index, 0]
        # The integral shock is correlated with the rate shock.
        z_orth = normals[:, index, 1] if normals.shape[-1] > 1 else torch.zeros_like(z_rate)
        rate, integral_step, _ = _transition(rate, start, end, a, sigma, curve, z_rate, z_orth)
        integral = integral + integral_step
        rates.append(rate)
        integrals.append(integral)
    return torch.stack((torch.stack(rates, dim=1), torch.stack(integrals, dim=1)), dim=-1)


def _b(a, start, end):
    dt = end - start
    return torch.where(a.abs() < 1e-8, dt, (1.0 - torch.exp(-a * dt)) / a)


def _instantaneous_forward(curve, t):
    bump = torch.full_like(t, 1e-4)
    return curve.forward_rate(t, t + bump)


def _log_a(curve, a, sigma, now, maturity, b):
    df_ratio = torch.log(curve.discount(maturity).clamp(min=1e-18) / curve.discount(now).clamp(min=1e-18))
    forward = _instantaneous_forward(curve, now)
    variance = (sigma**2) / (4.0 * a.clamp(min=1e-8)) * (1.0 - torch.exp(-2.0 * a * now)) * b**2
    return df_ratio + b * forward - variance


def _bond_volatility(a, sigma, expiry, maturity):
    b = _b(a, expiry, maturity)
    return sigma * torch.sqrt((1.0 - torch.exp(-2.0 * a * expiry)) / (2.0 * a.clamp(min=1e-8))) * b


def _critical_rate(curve, a, sigma, expiry, pay_times, coupon):
    """Short rate that makes the coupon bond worth par at expiry."""

    def value(rate):
        return (coupon * bond_price(curve, rate, a, sigma, expiry, pay_times)).sum() - 1.0

    lo = expiry.new_tensor(-1.0)
    hi = expiry.new_tensor(1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_high = value(mid) > 0
        # Bond prices fall as the short rate rises, so value is decreasing.
        hi = torch.where(too_high, hi, mid)
        lo = torch.where(too_high, mid, lo)
    return 0.5 * (lo + hi)


def _mean_rate(curve, a, sigma, t):
    forward = _instantaneous_forward(curve, t)
    return forward + (sigma**2) / (2.0 * a.clamp(min=1e-8) ** 2) * (1.0 - torch.exp(-a * t)) ** 2


def _transition(rate, start, end, a, sigma, curve, z_rate, z_orth):
    dt = end - start
    decay = torch.exp(-a * dt)
    mean = rate * decay + _mean_rate(curve, a, sigma, end) - _mean_rate(curve, a, sigma, start) * decay
    var = sigma**2 * (1.0 - torch.exp(-2.0 * a * dt)) / (2.0 * a.clamp(min=1e-8))
    next_rate = mean + torch.sqrt(var.clamp(min=0.0)) * z_rate
    # Integral of the deviation plus the deterministic forward integral.
    b = _b(a, start, end)
    forward_integral = -torch.log(curve.discount(end).clamp(min=1e-18) / curve.discount(start).clamp(min=1e-18))
    # Convexity piece of alpha, integrated. See the module docstring.
    convexity = (sigma**2) / (2.0 * a.clamp(min=1e-8) ** 2) * (
        dt + (2.0 / a.clamp(min=1e-8)) * (torch.exp(-a * end) - torch.exp(-a * start))
        - (1.0 / (2.0 * a.clamp(min=1e-8))) * (torch.exp(-2.0 * a * end) - torch.exp(-2.0 * a * start))
    )
    # Stochastic integral shares z_rate and an orthogonal shock.
    var_i = (sigma**2) / a.clamp(min=1e-8) ** 2 * (
        dt - 2.0 * b + (1.0 - torch.exp(-2.0 * a * dt)) / (2.0 * a.clamp(min=1e-8))
    )
    cov = (sigma**2) / a.clamp(min=1e-8) * (
        b - (1.0 - torch.exp(-2.0 * a * dt)) / (2.0 * a.clamp(min=1e-8))
    )
    std_r = torch.sqrt(var.clamp(min=0.0))
    beta = cov / var.clamp(min=1e-18)
    residual = (var_i - beta**2 * var).clamp(min=0.0)
    deviation = rate - _instantaneous_forward(curve, start)
    mean_i = deviation * b + forward_integral + convexity
    integral = mean_i + beta * (next_rate - mean) + torch.sqrt(residual) * z_orth
    return next_rate, integral, var
