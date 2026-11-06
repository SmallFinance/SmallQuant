"""Constant-elasticity-of-variance European option."""

from __future__ import annotations

import torch

from ._common import as_tensors


def _ncx2_cdf(x: torch.Tensor, df: torch.Tensor, noncentrality: torch.Tensor, terms: int = 80) -> torch.Tensor:
    """CDF of the noncentral chi-squared distribution."""
    half = noncentrality / 2.0
    js = torch.arange(terms, dtype=x.dtype, device=x.device)
    log_w = -half.unsqueeze(-1) + js * torch.log(torch.clamp(half, min=1e-30)).unsqueeze(-1)
    log_w = log_w - torch.lgamma(js + 1.0)
    weights = torch.exp(log_w)
    shape = df.unsqueeze(-1) / 2.0 + js
    upper = torch.clamp(x, min=0.0).unsqueeze(-1) / 2.0
    probs = torch.special.gammainc(shape, upper)
    return (weights * probs).sum(dim=-1)


def price(
    *,
    forward: torch.Tensor,
    strike: torch.Tensor,
    expiry: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """CEV price on a forward, Schroder's noncentral chi-squared formula.

    The dynamics are ``dF = alpha F^beta dW``. ``discount`` is the zero bond to expiry.
    """
    forward, strike, expiry, alpha, beta, discount = as_tensors(
        forward, strike, expiry, alpha, beta, discount
    )

    def chi(level: torch.Tensor) -> torch.Tensor:
        return level ** (2.0 * (1.0 - beta)) / (alpha * (1.0 - beta)) ** 2

    x0 = chi(forward)
    k_tilde = chi(strike)
    delta = (1.0 - 2.0 * beta) / (1.0 - beta)
    call_low = forward * (
        1.0 - _ncx2_cdf(k_tilde / expiry, 4.0 - delta, x0 / expiry)
    ) - strike * _ncx2_cdf(x0 / expiry, 2.0 - delta, k_tilde / expiry)
    g = torch.special.gammainc(0.5 * delta - 1.0, x0 / (2.0 * expiry))
    call_high = forward * (
        g - _ncx2_cdf(x0 / expiry, delta - 2.0, k_tilde / expiry)
    ) - strike * _ncx2_cdf(k_tilde / expiry, delta, x0 / expiry)
    call = torch.where(delta < 2.0, call_low, call_high)
    put_low = -forward * _ncx2_cdf(k_tilde / expiry, 4.0 - delta, x0 / expiry) + strike * (
        1.0 - _ncx2_cdf(x0 / expiry, 2.0 - delta, k_tilde / expiry)
    )
    put_high = -forward * _ncx2_cdf(x0 / expiry, delta - 2.0, k_tilde / expiry) + strike * (
        1.0 - _ncx2_cdf(k_tilde / expiry, delta, x0 / expiry)
    )
    put = torch.where(delta < 2.0, put_low, put_high)
    return discount * torch.where(is_call, call, put)
