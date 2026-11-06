"""Monte Carlo prices, the Brownian bridge, and pathwise Greeks."""

import math

import pytest
import torch

import QuantLib as ql

from smallquant.engines.monte_carlo import price as mc_price
from smallquant.math.random import brownian_bridge
from smallquant.models import asian, bachelier, black_scholes
from smallquant.payoffs import (
    arithmetic_average_price,
    cash_or_nothing,
    european,
    geometric_average_price,
)

pytestmark = pytest.mark.quantlib

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()


def _date(days):
    return EVAL + int(days)


def _year(days):
    return days / 365.0


def _flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(EVAL, rate, DAY_COUNT, ql.Continuous))


def _process(spot, rate, div, vol):
    ql.Settings.instance().evaluationDate = EVAL
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        _flat(div),
        _flat(rate),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(EVAL, CALENDAR, vol, DAY_COUNT)),
    )


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _within_stderr(actual, expected, stderr, scale=3.0):
    assert abs(float(actual) - float(expected)) <= scale * float(stderr)


def test_brownian_bridge_matches_quantlib():
    grid = [0.0, 0.30, 0.55, 1.20]
    draws = [
        [0.20, -1.10, 0.70],
        [-0.40, 0.15, 1.25],
    ]
    reference = ql.BrownianBridge(grid[1:])
    expected = torch.tensor(
        [reference.transform(draw) for draw in draws],
        dtype=torch.float64,
    )
    actual = brownian_bridge(torch.tensor(draws, dtype=torch.float64).unsqueeze(-1), _t(grid))
    assert torch.allclose(actual[..., 0], expected, rtol=1e-12, atol=1e-12)


def test_repeated_seed_and_chunks_match():
    times = _t([0.0, 0.5, 1.0])
    spot, rate, vol, div = _t(100.0), _t(0.05), _t(0.2), _t(0.01)
    strike = _t(100.0)
    call = torch.tensor(True)

    def step(normals):
        return black_scholes.step(
            spot=spot, times=times, normals=normals, rate=rate, vol=vol, div=div
        )

    def payoff(paths, grid):
        return european(paths, grid, strike=strike, is_call=call)

    kwargs = dict(
        payoff=payoff,
        step=step,
        times=times,
        discount=torch.exp(-rate * times[-1]),
        num_paths=4096,
        seed=11,
        antithetic=True,
        brownian_bridge=True,
    )
    first = mc_price(**kwargs)
    second = mc_price(**kwargs)
    chunked = mc_price(**kwargs, chunk_size=512)
    assert torch.equal(first, second)
    assert torch.equal(first, chunked)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_black_scholes_within_stderr(is_call):
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.02, 0.2
    times = _t([0.0, 1.0])
    market = dict(
        spot=_t(spot),
        times=times,
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
    )
    flag = torch.tensor(is_call)
    estimate, stderr = mc_price(
        lambda paths, grid: european(paths, grid, strike=_t(strike), is_call=flag),
        lambda normals: black_scholes.step(normals=normals, **market),
        times=times,
        discount=torch.exp(-market["rate"] * times[-1]),
        num_paths=65536,
        seed=3,
        return_stderr=True,
    )
    closed = black_scholes.price(
        spot=market["spot"],
        strike=_t(strike),
        expiry=times[-1],
        rate=market["rate"],
        vol=market["vol"],
        div=market["div"],
        is_call=flag,
    )
    _within_stderr(estimate, closed, stderr)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_bachelier_within_stderr(is_call):
    forward, strike, normal_vol, rate, expiry = 100.0, 100.0, 15.0, 0.03, 1.0
    times = _t([0.0, expiry])
    discount = torch.exp(_t(-rate * expiry))
    flag = torch.tensor(is_call)
    estimate, stderr = mc_price(
        lambda paths, grid: european(paths, grid, strike=_t(strike), is_call=flag),
        lambda normals: bachelier.step(
            forward=_t(forward), times=times, normals=normals, normal_vol=_t(normal_vol)
        ),
        times=times,
        discount=discount,
        num_paths=65536,
        seed=5,
        return_stderr=True,
    )
    closed = bachelier.price(
        forward=_t(forward),
        strike=_t(strike),
        expiry=_t(expiry),
        normal_vol=_t(normal_vol),
        discount=discount,
        is_call=flag,
    )
    _within_stderr(estimate, closed, stderr)


def test_geometric_asian_within_stderr():
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.02, 0.2
    days = list(range(30, 361, 30))
    times = _t([0.0, *[_year(day) for day in days]])
    flag = torch.tensor(True)
    market = dict(spot=_t(spot), times=times, rate=_t(rate), vol=_t(vol), div=_t(div))
    estimate, stderr = mc_price(
        lambda paths, grid: geometric_average_price(paths, grid, strike=_t(strike), is_call=flag),
        lambda normals: black_scholes.step(normals=normals, **market),
        times=times,
        discount=torch.exp(-market["rate"] * times[-1]),
        num_paths=65536,
        seed=9,
        return_stderr=True,
    )
    closed = asian.discrete_price(
        spot=market["spot"],
        strike=_t(strike),
        expiry=times[-1],
        rate=market["rate"],
        vol=market["vol"],
        div=market["div"],
        fixing_times=times[1:],
        is_call=flag,
    )
    _within_stderr(estimate, closed, stderr)


def test_arithmetic_asian_matches_quantlib():
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.02, 0.2
    days = list(range(30, 361, 30))
    times = _t([0.0, *[_year(day) for day in days]])
    flag = torch.tensor(True)
    market = dict(spot=_t(spot), times=times, rate=_t(rate), vol=_t(vol), div=_t(div))
    known = asian.discrete_price(
        spot=market["spot"],
        strike=_t(strike),
        expiry=times[-1],
        rate=market["rate"],
        vol=market["vol"],
        div=market["div"],
        fixing_times=times[1:],
        is_call=flag,
    )
    estimate, stderr = mc_price(
        lambda paths, grid: arithmetic_average_price(paths, grid, strike=_t(strike), is_call=flag),
        lambda normals: black_scholes.step(normals=normals, **market),
        times=times,
        discount=torch.exp(-market["rate"] * times[-1]),
        num_paths=65536,
        seed=13,
        return_stderr=True,
        control=lambda paths, grid: geometric_average_price(
            paths, grid, strike=_t(strike), is_call=flag
        ),
        control_price=known,
    )
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        0.0,
        0,
        [_date(day) for day in days],
        ql.PlainVanillaPayoff(ql.Option.Call, strike),
        ql.EuropeanExercise(_date(days[-1])),
    )
    option.setPricingEngine(
        ql.MCPRDiscreteArithmeticAPEngine(
            _process(spot, rate, div, vol),
            False,
            False,
            True,
            20000,
            None,
            None,
            21,
        )
    )
    reference = option.NPV()
    tolerance = 3.0 * math.hypot(float(stderr), option.errorEstimate())
    assert abs(float(estimate) - reference) <= tolerance


def test_pathwise_delta_and_vega():
    times = _t([0.0, 1.0])
    spot = _t(100.0).requires_grad_()
    vol = _t(0.2).requires_grad_()
    rate, div, strike = _t(0.05), _t(0.02), _t(100.0)
    flag = torch.tensor(True)
    estimate = mc_price(
        lambda paths, grid: european(paths, grid, strike=strike, is_call=flag),
        lambda normals: black_scholes.step(
            spot=spot, times=times, normals=normals, rate=rate, vol=vol, div=div
        ),
        times=times,
        discount=torch.exp(-rate * times[-1]),
        num_paths=131072,
        seed=17,
        antithetic=True,
    )
    delta, vega = torch.autograd.grad(estimate, (spot, vol))
    spot_ref = spot.detach().requires_grad_()
    vol_ref = vol.detach().requires_grad_()
    closed_price = black_scholes.price(
        spot=spot_ref,
        strike=strike,
        expiry=times[-1],
        rate=rate,
        vol=vol_ref,
        div=div,
        is_call=flag,
    )
    closed_delta, closed_vega = torch.autograd.grad(closed_price, (spot_ref, vol_ref))
    assert abs(float(delta) - float(closed_delta)) < 0.01
    assert abs(float(vega) - float(closed_vega)) < 0.15


def test_smoothed_digital_delta_is_not_zero():
    times = _t([0.0, 1.0])
    spot = _t(100.0).requires_grad_()
    rate, div, vol = _t(0.05), _t(0.02), _t(0.2)
    strike, cash, eps = _t(100.0), _t(1.0), _t(1.0)
    flag = torch.tensor(True)

    def step(normals):
        return black_scholes.step(
            spot=spot, times=times, normals=normals, rate=rate, vol=vol, div=div
        )

    raw = mc_price(
        lambda paths, grid: cash_or_nothing(
            paths, grid, strike=strike, is_call=flag, cash=cash
        ),
        step,
        times=times,
        discount=torch.exp(-rate * times[-1]),
        num_paths=32768,
        seed=19,
    )
    assert raw.grad_fn is None
    smoothed = mc_price(
        lambda paths, grid: cash_or_nothing(
            paths, grid, strike=strike, is_call=flag, cash=cash, smoothing="sigmoid", eps=eps
        ),
        step,
        times=times,
        discount=torch.exp(-rate * times[-1]),
        num_paths=131072,
        seed=19,
    )
    smooth_delta = torch.autograd.grad(smoothed, spot)[0]
    expiry = times[-1]
    sigma_sqrt = vol * torch.sqrt(expiry)
    d2 = (torch.log(spot.detach() / strike) + (rate - div - 0.5 * vol * vol) * expiry) / sigma_sqrt
    density = torch.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
    analytic_delta = cash * torch.exp(-rate * expiry) * density / (spot.detach() * sigma_sqrt)
    assert abs(float(smooth_delta) - float(analytic_delta)) < 0.002
