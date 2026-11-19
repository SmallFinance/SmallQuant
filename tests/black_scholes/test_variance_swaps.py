"""Variance swap replication against QuantLib, TF Quant Finance and the flat-vol limit."""

import math

import pytest
import QuantLib as ql
import torch

from smallquant.black_scholes import variance_swap_fair_strike, variance_swaps
from smallquant.black_scholes.variance_swaps import fair_strike, replicating_weights

pytestmark = pytest.mark.quantlib

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()
F64 = torch.float64


def _t(value):
    return torch.tensor(value, dtype=F64)


def _smile(strike, reference):
    """A skewed, convex smile so each strike carries its own vol."""
    moneyness = math.log(strike / reference)
    return 0.22 - 0.15 * moneyness + 0.4 * moneyness**2


def _quantlib_variance(*, spot, rate, div, days, put_strikes, call_strikes, put_vol_quotes,
                       call_vol_quotes, dk):
    """ReplicatingVarianceSwapEngine with a strike-dependent smile.

    QuantLib prices every option on one process, so the smile is a BlackVarianceSurface
    flat in time with exactly the quoted vol at each strike.
    """
    ql.Settings.instance().evaluationDate = EVAL
    maturity = EVAL + days
    quotes = dict(zip(put_strikes, put_vol_quotes))
    quotes.update(zip(call_strikes, call_vol_quotes))
    strikes = sorted(quotes)
    matrix = ql.Matrix(len(strikes), 2)
    for row, strike in enumerate(strikes):
        matrix[row][0] = quotes[strike]
        matrix[row][1] = quotes[strike]
    surface = ql.BlackVarianceSurface(EVAL, CALENDAR, [maturity, maturity + 365], strikes,
                                      matrix, DAY_COUNT)
    surface.setInterpolation("bilinear")
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(EVAL, div, DAY_COUNT, ql.Continuous)),
        ql.YieldTermStructureHandle(ql.FlatForward(EVAL, rate, DAY_COUNT, ql.Continuous)),
        ql.BlackVolTermStructureHandle(surface),
    )
    swap = ql.VarianceSwap(ql.Position.Long, 0.04, 50_000.0, EVAL, maturity)
    swap.setPricingEngine(ql.ReplicatingVarianceSwapEngine(process, dk, call_strikes,
                                                           put_strikes))
    # variance() is the fair variance strike (vol squared, decimal), not vol.
    return swap.variance()


QL_CASES = [
    # spot, rate, days, reference, step, num_puts, num_calls
    (100.0, 0.05, 91, 100.0, 5.0, 15, 30),
    (100.0, 0.05, 365, 100.0, 5.0, 19, 40),
    (104.0, 0.03, 200, 100.0, 2.5, 30, 60),
    (95.0, 0.00, 30, 100.0, 1.0, 50, 80),
]


@pytest.mark.parametrize("spot, rate, days, reference, step, num_puts, num_calls", QL_CASES)
def test_matches_quantlib_replicating_engine(spot, rate, days, reference, step, num_puts,
                                             num_calls):
    puts = [reference - step * i for i in range(num_puts)]
    calls = [reference + step * i for i in range(num_calls)]
    put_vol_quotes = [_smile(strike, reference) for strike in puts]
    call_vol_quotes = [_smile(strike, reference) for strike in calls]
    expected = _quantlib_variance(spot=spot, rate=rate, div=0.0, days=days, put_strikes=puts,
                                  call_strikes=calls, put_vol_quotes=put_vol_quotes,
                                  call_vol_quotes=call_vol_quotes, dk=step)
    # QuantLib appends one extra strike dk beyond each strip to close the last segment;
    # here that closing strike is passed explicitly, with an (unused) vol.
    result = fair_strike(
        put_strikes=_t(puts + [puts[-1] - step]),
        put_vols=_t(put_vol_quotes + [put_vol_quotes[-1]]),
        call_strikes=_t(calls + [calls[-1] + step]),
        call_vols=_t(call_vol_quotes + [call_vol_quotes[-1]]),
        expiry=_t(days / 365.0),  # Actual/365 Fixed, the risk-free curve's day count
        rate=_t(rate),
        spot=_t(spot),
        reference_strike=_t(reference),
    )
    assert result.item() == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_quantlib_dividend_gap_is_its_forward_term():
    """QuantLib prices the options with the dividend yield but builds its forward term
    from ``S exp(r T)`` and drift ``r``. The gap is exactly that term's difference."""
    spot, rate, div, days, reference, step = 100.0, 0.05, 0.02, 365, 100.0, 5.0
    puts = [reference - step * i for i in range(19)]
    calls = [reference + step * i for i in range(40)]
    vols_p = [_smile(strike, reference) for strike in puts]
    vols_c = [_smile(strike, reference) for strike in calls]
    quantlib = _quantlib_variance(spot=spot, rate=rate, div=div, days=days, put_strikes=puts,
                                  call_strikes=calls, put_vol_quotes=vols_p,
                                  call_vol_quotes=vols_c, dk=step)
    expiry = days / 365.0
    result = fair_strike(
        put_strikes=_t(puts + [puts[-1] - step]), put_vols=_t(vols_p + [vols_p[-1]]),
        call_strikes=_t(calls + [calls[-1] + step]), call_vols=_t(vols_c + [vols_c[-1]]),
        expiry=_t(expiry), rate=_t(rate), spot=_t(spot), reference_strike=_t(reference),
        div=_t(div),
    ).item()
    log_forward = math.log(spot / reference) + (rate - div) * expiry
    ours_term = 2.0 / expiry * (log_forward - math.exp(log_forward) + 1.0)
    quantlib_term = 2.0 * rate - 2.0 / expiry * (
        spot / reference * math.exp(rate * expiry) - 1.0 + math.log(reference / spot)
    )
    assert quantlib - result == pytest.approx(quantlib_term - ours_term, rel=1e-10, abs=1e-14)


def test_tf_worked_examples():
    # Numbers from the docstrings of tf_quant_finance/black_scholes/variance_swaps.py.
    weights = replicating_weights(_t([100.0, 95.0, 90.0, 85.0]), reference_strike=_t(100.0),
                                  expiry=_t(0.25))
    torch.testing.assert_close(weights, _t([0.00206927, 0.00443828, 0.00494591]),
                               rtol=0.0, atol=5e-9)

    result = variance_swap_fair_strike(
        put_strikes=_t([[100.0, 95.0, 90.0, 85.0], [1000.0, 900.0, 800.0, 700.0]]),
        put_vols=0.2 * torch.ones(2, 4, dtype=F64),
        call_strikes=_t([[100.0, 105.0, 110.0, 115.0], [1000.0, 1100.0, 1200.0, 1300.0]]),
        call_vols=0.2 * torch.ones(2, 4, dtype=F64),
        expiry=_t([0.25, 0.25]),
        rate=_t([0.05, 0.05]),
        spot=_t([100.0, 1000.0]),
        reference_strike=_t([100.0, 1000.0]),
    )
    torch.testing.assert_close(result, _t([0.03825004, 0.04659269]), rtol=0.0, atol=5e-9)


def test_call_weights_are_positive_and_ascending_strip_matches_paper():
    strikes = _t([100.0, 105.0, 110.0, 115.0])
    expiry = _t(0.5)
    weights = replicating_weights(strikes, reference_strike=_t(100.0), expiry=expiry)
    payoff = 2.0 / expiry * (strikes / 100.0 - 1.0 - torch.log(strikes / 100.0))
    slope = (payoff[1:] - payoff[:-1]) / 5.0
    torch.testing.assert_close(weights, torch.cat([slope[:1], slope[1:] - slope[:-1]]))
    assert bool((weights > 0).all())


def test_repeated_strike_gives_finite_weights():
    weights = replicating_weights(_t([100.0, 100.0, 95.0, 90.0]), reference_strike=_t(100.0),
                                  expiry=_t(0.25))
    assert torch.isfinite(weights).all()


def _dense_flat(vol, expiry, rate, spot, div):
    """Flat vol, strips struck around the forward at log spacing 0.001 out to log(K/F) = +-3."""
    forward = spot * torch.exp((rate - div) * expiry)
    put_grid = torch.exp(torch.linspace(0.0, -3.0, 3001, dtype=F64))
    call_grid = torch.exp(torch.linspace(0.0, 3.0, 3001, dtype=F64))
    return fair_strike(
        put_strikes=forward.unsqueeze(-1) * put_grid, put_vols=vol.unsqueeze(-1),
        call_strikes=forward.unsqueeze(-1) * call_grid, call_vols=vol.unsqueeze(-1),
        expiry=expiry, rate=rate, spot=spot, reference_strike=forward, div=div,
    )


def test_flat_vol_dense_strip_converges_to_sigma_squared_batched():
    expiry = _t([0.1, 0.5, 1.0, 2.0, 1.0])
    spot = _t([100.0, 50.0, 100.0, 250.0, 100.0])
    rate = _t([0.05, 0.02, 0.0, 0.04, 0.05])
    div = _t([0.0, 0.01, 0.0, 0.02, 0.03])
    vol = _t([0.2, 0.3, 0.15, 0.25, 0.4])
    result = _dense_flat(vol, expiry, rate, spot, div)
    assert result.shape == (5,)
    torch.testing.assert_close(result, vol**2, rtol=1e-4, atol=0.0)


def test_off_forward_reference_still_converges():
    # S* away from the forward: the centrality term corrects for it.
    vol, expiry, rate, spot, div = _t(0.25), _t(1.0), _t(0.05), _t(100.0), _t(0.02)
    puts = 90.0 * torch.exp(torch.linspace(0.0, -6.0, 1500, dtype=F64))
    calls = 90.0 * torch.exp(torch.linspace(0.0, 6.0, 1500, dtype=F64))
    result = fair_strike(put_strikes=puts, put_vols=vol, call_strikes=calls, call_vols=vol,
                         expiry=expiry, rate=rate, spot=spot, reference_strike=_t(90.0),
                         div=div)
    torch.testing.assert_close(result, vol**2, rtol=1e-4, atol=0.0)


def test_broadcasts_scalar_market_over_batched_strips():
    puts = _t([[100.0, 95.0, 90.0, 85.0], [100.0, 90.0, 80.0, 70.0]])
    calls = _t([[100.0, 105.0, 110.0, 115.0], [100.0, 110.0, 120.0, 130.0]])
    batched = fair_strike(put_strikes=puts, put_vols=_t(0.2), call_strikes=calls,
                          call_vols=_t(0.2), expiry=_t(0.25), rate=_t(0.05), spot=_t(100.0),
                          reference_strike=_t(100.0))
    assert batched.shape == (2,)
    for row in range(2):
        single = fair_strike(put_strikes=puts[row], put_vols=_t(0.2), call_strikes=calls[row],
                             call_vols=_t(0.2), expiry=_t(0.25), rate=_t(0.05),
                             spot=_t(100.0), reference_strike=_t(100.0))
        torch.testing.assert_close(batched[row], single)


def test_gradients_with_respect_to_each_vol():
    puts = _t([100.0, 95.0, 90.0, 85.0, 80.0])
    calls = _t([100.0, 105.0, 110.0, 115.0, 120.0])
    put_vols = _t([0.20, 0.22, 0.24, 0.26, 0.28]).requires_grad_()
    call_vols = _t([0.20, 0.19, 0.185, 0.18, 0.18]).requires_grad_()
    result = fair_strike(put_strikes=puts, put_vols=put_vols, call_strikes=calls,
                         call_vols=call_vols, expiry=_t(0.5), rate=_t(0.03), spot=_t(100.0),
                         reference_strike=_t(100.0), div=_t(0.01))
    put_grad, call_grad = torch.autograd.grad(result, (put_vols, call_vols))
    assert torch.isfinite(put_grad).all() and torch.isfinite(call_grad).all()
    # Every traded option has positive weight and positive vega; the closing strike is unused.
    assert bool((put_grad[:-1] > 0).all()) and bool((call_grad[:-1] > 0).all())
    assert put_grad[-1] == 0 and call_grad[-1] == 0

    # Autograd agrees with a central difference on one vol.
    bump = 1e-6
    def at(vol_value):
        vols = put_vols.detach().clone()
        vols[2] = vol_value
        return fair_strike(put_strikes=puts, put_vols=vols, call_strikes=calls,
                           call_vols=call_vols.detach(), expiry=_t(0.5), rate=_t(0.03),
                           spot=_t(100.0), reference_strike=_t(100.0), div=_t(0.01))
    numeric = (at(0.24 + bump) - at(0.24 - bump)) / (2 * bump)
    torch.testing.assert_close(put_grad[2], numeric, rtol=1e-6, atol=1e-10)


def test_dtype_and_tensor_only():
    result = fair_strike(put_strikes=_t([100.0, 95.0, 90.0]).float(), put_vols=_t(0.2).float(),
                         call_strikes=_t([100.0, 105.0, 110.0]).float(),
                         call_vols=_t(0.2).float(), expiry=_t(0.25).float(),
                         rate=_t(0.05).float(), spot=_t(100.0).float(),
                         reference_strike=_t(100.0).float())
    assert result.dtype == torch.float32
    with pytest.raises(TypeError):
        replicating_weights(_t([100.0, 95.0]), reference_strike=100.0, expiry=_t(0.25))


def test_module_exports():
    assert variance_swap_fair_strike is variance_swaps.fair_strike
