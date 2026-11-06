"""Finite-difference prices against Black-Scholes and QuantLib."""

import pytest
import torch

import QuantLib as ql

from smallquant.engines.finite_difference import price as fd_price
from smallquant.models.black_scholes import price as bs_price

pytestmark = pytest.mark.quantlib

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
NODES = 200


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _process(spot, rate, div, vol):
    ql.Settings.instance().evaluationDate = EVAL
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(EVAL, div, DAY_COUNT, ql.Continuous)),
        ql.YieldTermStructureHandle(ql.FlatForward(EVAL, rate, DAY_COUNT, ql.Continuous)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(EVAL, ql.NullCalendar(), vol, DAY_COUNT)),
    )


def _fd(**kwargs):
    defaults = dict(num_space=NODES, num_time=NODES, rannacher_steps=2, theta=0.5)
    defaults.update(kwargs)
    return fd_price(**defaults)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_matches_black_scholes(is_call):
    spot, strike, expiry, rate, vol, div = _t(100.0), _t(105.0), _t(1.0), _t(0.05), _t(0.25), _t(0.03)
    flag = torch.tensor(is_call)
    finite = _fd(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag
    )
    closed = bs_price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag
    )
    assert abs(float(finite) - float(closed)) < 1e-3


def test_european_delta_and_vega():
    spot = _t(100.0).requires_grad_()
    vol = _t(0.2).requires_grad_()
    strike, expiry, rate, div = _t(100.0), _t(1.0), _t(0.05), _t(0.02)
    flag = torch.tensor(True)
    finite = _fd(spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag)
    delta, vega = torch.autograd.grad(finite, (spot, vol))
    spot_ref = spot.detach().requires_grad_()
    vol_ref = vol.detach().requires_grad_()
    closed = bs_price(
        spot=spot_ref, strike=strike, expiry=expiry, rate=rate, vol=vol_ref, div=div, is_call=flag
    )
    closed_delta, closed_vega = torch.autograd.grad(closed, (spot_ref, vol_ref))
    assert abs(float(delta) - float(closed_delta)) < 1e-3
    assert abs(float(vega) - float(closed_vega)) < 0.05


@pytest.mark.parametrize("is_call", [True, False])
def test_american_matches_quantlib(is_call):
    spot, strike, rate, div, vol = 100.0, 100.0, 0.06, 0.03, 0.2
    kind = ql.Option.Call if is_call else ql.Option.Put
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(kind, strike),
        ql.AmericanExercise(EVAL, EVAL + 365),
    )
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(
            _process(spot, rate, div, vol),
            800,
            800,
            2,
            ql.FdmSchemeDesc.CrankNicolson(),
        )
    )
    finite = _fd(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(1.0),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=torch.tensor(is_call),
        exercise="american",
        num_space=400,
        num_time=400,
    )
    assert abs(float(finite) - option.NPV()) < 1e-3


def test_batch_of_spots_matches_single_prices():
    spots = _t([90.0, 110.0])
    kwargs = dict(
        strike=_t(100.0),
        expiry=_t(1.0),
        rate=_t(0.05),
        vol=_t(0.2),
        div=_t(0.02),
        is_call=torch.tensor(True),
        num_space=80,
        num_time=80,
    )
    batched = fd_price(spot=spots, **kwargs)
    single = torch.stack([fd_price(spot=spot, **kwargs) for spot in spots])
    assert batched.shape == (2,)
    assert torch.allclose(batched, single)
