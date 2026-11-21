"""Variance swap fair strike from a discrete strip of European options.

Laid out after TF Quant Finance's ``tf_quant_finance.black_scholes.variance_swaps``
and implemented from Appendix A of Demeterfi, Derman, Kamal and Zou (1999).

A variance swap pays ``notional * (realized_variance - variance_strike)`` at expiry.
Under continuous sampling, realized variance is replicated by a static position in
the log contract ``f(S_T) = (2 / T) * ((S_T - S*) / S* - log(S_T / S*))`` around a
reference strike ``S*``, plus a dynamically rebalanced forward. The log contract is
approximated by a piecewise-linear payoff through the traded strikes: out-of-the-money
puts below ``S*`` and out-of-the-money calls above it, each strike weighted by the
change in slope of that piecewise-linear payoff.

References:
    Kresimir Demeterfi, Emanuel Derman, Michael Kamal and Joseph Zou. More than you
    ever wanted to know about volatility swaps. Goldman Sachs Quantitative Strategies
    Research Notes, March 1999. Appendix A.
"""

from __future__ import annotations

import torch

from smallquant._tensor import align, check

from .vanilla_prices import option_price

__all__ = ["fair_strike", "replicating_weights"]


def _log_payoff(strike: torch.Tensor, reference_strike: torch.Tensor, expiry: torch.Tensor) -> torch.Tensor:
    """``f(K) = (2 / T) ((K - S*) / S* - log(K / S*))``, the log contract of Appendix A."""
    ratio = strike / reference_strike
    return (2.0 / expiry) * (ratio - 1.0 - torch.log(ratio))


def _diff(values: torch.Tensor) -> torch.Tensor:
    return values[..., 1:] - values[..., :-1]


def replicating_weights(
    ordered_strikes: torch.Tensor,
    *,
    reference_strike: torch.Tensor,
    expiry: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Option weights that replicate the variance swap's log payoff on one strip.

    Following Appendix A of Demeterfi et al. (1999), the log contract
    ``f(K) = (2 / T) ((K - S*) / S* - log(K / S*))`` is replaced by the piecewise-linear
    function through ``(K_i, f(K_i))``. Writing ``s_i = (f(K_{i+1}) - f(K_i)) / (K_{i+1} - K_i)``
    for the slope on the ``i``-th segment, the option struck at ``K_i`` carries weight

        ``w_0 = |s_0|``,  ``w_i = |s_i| - |s_{i-1}|``  for ``i = 1, ..., n - 2``.

    Pass call strikes in ascending order starting at ``reference_strike``, or put
    strikes in descending order starting at ``reference_strike``. The direction is
    read per strip (row) from the strikes themselves: a strictly descending strip is
    taken as puts and has the sign of its slopes flipped, anything else as calls. The
    last strike only closes the final segment and receives no weight. Repeated
    adjacent strikes give a zero slope on that segment rather than NaN.

    Args:
        ordered_strikes: strikes, shape ``batch_shape + [num_strikes]``, all positive
        reference_strike: the strike ``S*`` that splits puts from calls, shape
            broadcastable to ``batch_shape``
        expiry: time to expiry in years, positive, broadcastable to ``batch_shape``
        dtype: optional floating dtype for the computation

    Returns:
        Weights per unit of variance notional, shape ``batch_shape + [num_strikes - 1]``,
        for the options struck at ``ordered_strikes[..., :-1]``.

    Example:
        >>> f64 = torch.float64
        >>> replicating_weights(torch.tensor([100.0, 95.0, 90.0, 85.0], dtype=f64),
        ...                     reference_strike=torch.tensor(100.0, dtype=f64),
        ...                     expiry=torch.tensor(0.25, dtype=f64))
        tensor([0.0021, 0.0044, 0.0049], dtype=torch.float64)
    """
    ordered_strikes, reference_strike, expiry = align(
        ordered_strikes, reference_strike, expiry, dtype=dtype
    )
    check("ordered_strikes", ordered_strikes, ordered_strikes > 0)
    check("reference_strike", reference_strike, reference_strike > 0)
    check("expiry", expiry, expiry > 0)
    reference_strike = reference_strike.unsqueeze(-1)
    expiry = expiry.unsqueeze(-1)

    strike_step = _diff(ordered_strikes)
    payoff_step = _diff(_log_payoff(ordered_strikes, reference_strike, expiry))
    flat = strike_step == 0
    safe_step = torch.where(flat, torch.ones_like(strike_step), strike_step)
    slope = torch.where(flat, torch.zeros_like(payoff_step), payoff_step / safe_step)

    # Puts walk down from S*, where f falls as K rises, so their slopes are negative.
    descending = (strike_step < 0).all(dim=-1, keepdim=True)
    slope = torch.where(descending, -slope, slope)
    previous = torch.cat([torch.zeros_like(slope[..., :1]), slope[..., :-1]], dim=-1)
    return slope - previous


def fair_strike(
    *,
    put_strikes: torch.Tensor,
    put_vols: torch.Tensor,
    call_strikes: torch.Tensor,
    call_vols: torch.Tensor,
    expiry: torch.Tensor,
    rate: torch.Tensor,
    spot: torch.Tensor,
    reference_strike: torch.Tensor,
    div: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fair variance strike of a variance swap, replicated from put and call strips.

    Implements Appendix A of Demeterfi et al. (1999). With forward
    ``F = S exp((r - q) T)`` and reference strike ``S*``,

        ``K_var = (2 / T) (log(F / S*) - F / S* + 1) + exp(r T) * Pi``,

    where ``Pi = sum_i w_i P(K_i) + sum_j w_j C(K_j)`` is today's value of the
    out-of-the-money puts and calls weighted by ``replicating_weights``, each priced
    with ``option_price`` at its own quoted vol. With ``q = 0`` the first term is the
    paper's ``(2 / T) (r T - (S / S*) exp(r T) + 1 - log(S* / S))``; a continuous
    dividend yield enters only through the forward and the option prices, since the
    expected log return of the self-financing forward position is ``(r - q) T``.

    The result is in variance units (vol squared, annualized, decimal), not vol, and
    assumes continuous sampling. Each strip's last strike closes the piecewise-linear
    payoff and is not itself traded, so its vol is unused. Autograd through the
    result gives the sensitivity of ``K_var`` to every quoted vol.

    Args:
        put_strikes: put strikes in descending order, starting at ``reference_strike``,
            shape ``batch_shape + [num_put_strikes]``
        put_vols: lognormal vol for each put strike, broadcastable to ``put_strikes``
        call_strikes: call strikes in ascending order, starting at ``reference_strike``,
            shape ``batch_shape + [num_call_strikes]``
        call_vols: lognormal vol for each call strike, broadcastable to ``call_strikes``
        expiry: time to expiry in years, positive, broadcastable to ``batch_shape``
        rate: continuously compounded risk-free rate, broadcastable to ``batch_shape``
        spot: underlying spot, broadcastable to ``batch_shape``
        reference_strike: the strike ``S*`` splitting puts from calls, usually the
            at-the-money strike, broadcastable to ``batch_shape``
        div: continuous dividend yield, broadcastable to ``batch_shape``; zero if omitted
        dtype: optional floating dtype for the computation

    Returns:
        Fair variance strike, shape ``batch_shape``.

    Example:
        >>> f64 = torch.float64
        >>> fair_strike(put_strikes=torch.tensor([100.0, 95.0, 90.0, 85.0], dtype=f64),
        ...             put_vols=torch.full((4,), 0.2, dtype=f64),
        ...             call_strikes=torch.tensor([100.0, 105.0, 110.0, 115.0], dtype=f64),
        ...             call_vols=torch.full((4,), 0.2, dtype=f64),
        ...             expiry=torch.tensor(0.25, dtype=f64),
        ...             rate=torch.tensor(0.05, dtype=f64),
        ...             spot=torch.tensor(100.0, dtype=f64),
        ...             reference_strike=torch.tensor(100.0, dtype=f64))
        tensor(0.0383, dtype=torch.float64)
    """
    tensors = [put_strikes, put_vols, call_strikes, call_vols, expiry, rate, spot,
               reference_strike]
    if div is not None:
        tensors.append(div)
    aligned = align(*tensors, dtype=dtype)
    put_strikes, put_vols, call_strikes, call_vols, expiry, rate, spot, reference_strike = aligned[:8]
    div = aligned[8] if div is not None else torch.zeros_like(rate)
    put_strikes, put_vols = torch.broadcast_tensors(put_strikes, put_vols)
    call_strikes, call_vols = torch.broadcast_tensors(call_strikes, call_vols)

    put_weights = replicating_weights(put_strikes, reference_strike=reference_strike,
                                      expiry=expiry)
    call_weights = replicating_weights(call_strikes, reference_strike=reference_strike,
                                       expiry=expiry)

    expiry_k, rate_k, spot_k, div_k = (value.unsqueeze(-1) for value in (expiry, rate, spot, div))
    put_prices = option_price(strike=put_strikes[..., :-1], expiry=expiry_k, is_call=False,
                              vol=put_vols[..., :-1], spot=spot_k, rate=rate_k, div=div_k)
    call_prices = option_price(strike=call_strikes[..., :-1], expiry=expiry_k, is_call=True,
                               vol=call_vols[..., :-1], spot=spot_k, rate=rate_k, div=div_k)
    options_value = (put_weights * put_prices).sum(dim=-1) + (call_weights * call_prices).sum(dim=-1)

    log_forward_ratio = torch.log(spot / reference_strike) + (rate - div) * expiry
    centrality = (2.0 / expiry) * (log_forward_ratio - torch.exp(log_forward_ratio) + 1.0)
    return centrality + torch.exp(rate * expiry) * options_value
