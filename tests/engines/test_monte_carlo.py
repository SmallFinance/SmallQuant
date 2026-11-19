"""Monte Carlo prices, the Brownian bridge, and pathwise Greeks."""

import math

import pytest
import QuantLib as ql
import torch

from smallquant.black_scholes import asian_prices, option_price
from smallquant.engines.monte_carlo import price as mc_price
from smallquant.math.random import brownian_bridge
from smallquant.models import arithmetic_brownian_motion, geometric_brownian_motion
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
        return geometric_brownian_motion.step(
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
        lambda normals: geometric_brownian_motion.step(normals=normals, **market),
        times=times,
        discount=torch.exp(-market["rate"] * times[-1]),
        num_paths=65536,
        seed=3,
        return_stderr=True,
    )
    closed = option_price(
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
        lambda normals: arithmetic_brownian_motion.step(
            forward=_t(forward), times=times, normals=normals, normal_vol=_t(normal_vol)
        ),
        times=times,
        discount=discount,
        num_paths=65536,
        seed=5,
        return_stderr=True,
    )
    closed = option_price(
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
        lambda normals: geometric_brownian_motion.step(normals=normals, **market),
        times=times,
        discount=torch.exp(-market["rate"] * times[-1]),
        num_paths=65536,
        seed=9,
        return_stderr=True,
    )
    closed = asian_prices.discrete_price(
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
    known = asian_prices.discrete_price(
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
        lambda normals: geometric_brownian_motion.step(normals=normals, **market),
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
        lambda normals: geometric_brownian_motion.step(
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
    closed_price = option_price(
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
        return geometric_brownian_motion.step(
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


def test_normal_stream_matches_one_draw():
    from smallquant.math.random import NormalStream, standard_normal

    device = torch.device("cpu")
    for method in ("pseudo", "sobol"):
        for antithetic in (False, True):
            kwargs = dict(
                num_paths=1000,
                num_steps=7,
                dim=2,
                seed=3,
                dtype=torch.float64,
                device=device,
                antithetic=antithetic,
                method=method,
            )
            full = standard_normal(**kwargs)
            stream = NormalStream(**kwargs)
            parts = [stream.take(count) for count in (300, 250, 450)]
            assert torch.equal(full, torch.cat(parts, dim=0))


def _engine_market():
    from smallquant.market import Market
    from smallquant.models.params import BlackScholesParams

    return Market(spot=_t(100.0), rate=_t(0.05), div=_t(0.02)), BlackScholesParams(vol=_t(0.3))


def test_engine_grid_merges_steps_with_instrument_dates():
    """MC-02: the grid is the instrument's dates merged with ``steps`` equal steps."""
    from smallquant.engines.pricing import MonteCarlo
    from smallquant.instruments import AsianOption, EuropeanOption

    market, _ = _engine_market()
    european = EuropeanOption(strike=_t(100.0), expiry=_t(1.0), is_call=torch.tensor(True))
    assert torch.allclose(MonteCarlo(steps=8)._times(european, market), torch.linspace(0.0, 1.0, 9, dtype=torch.float64))
    asian = AsianOption(strike=_t(100.0), expiry=_t(1.0), is_call=torch.tensor(True), fixing_times=_t([0.3, 0.7, 1.0]))
    grid = MonteCarlo(steps=4)._times(asian, market)
    assert torch.equal(grid, _t([0.0, 0.25, 0.3, 0.5, 0.7, 0.75, 1.0]))


def test_asian_averages_fixing_dates_only_on_a_finer_grid():
    from smallquant.instruments import AsianOption

    fixings = _t([0.3, 0.7, 1.0])
    times = _t([0.0, 0.25, 0.3, 0.5, 0.7, 0.75, 1.0])
    paths = torch.linspace(90.0, 110.0, 4 * times.numel(), dtype=torch.float64).reshape(4, -1, 1)
    on_fixings = paths[:, [0, 2, 4, 6], :]
    for average in ("arithmetic", "geometric"):
        option = AsianOption(
            strike=_t(100.0), expiry=_t(1.0), is_call=torch.tensor(True), fixing_times=fixings, average=average
        )
        assert torch.equal(option.payoff(paths, times), option.payoff(on_fixings, _t([0.0, 0.3, 0.7, 1.0])))


@pytest.mark.parametrize("steps", [4, 16])
@pytest.mark.parametrize(("kind", "ql_kind", "barrier", "is_call"), [
    ("down_out", ql.Barrier.DownOut, 90.0, True),
    ("up_in", ql.Barrier.UpIn, 115.0, False),
])
def test_discrete_barrier_monitors_every_step(steps, kind, ql_kind, barrier, is_call):
    """AT-30: a discrete barrier with ``steps=n`` is checked on n dates.

    QuantLib's biased ``MCBarrierEngine`` checks the barrier only on its time
    grid, so it prices the same contract. Both estimates use the same path count
    with antithetics, so the difference is held to 3 combined standard errors.
    """
    from smallquant.engines.pricing import MonteCarlo
    from smallquant.instruments import BarrierOption

    market, params = _engine_market()
    paths = 32768
    option = BarrierOption(
        strike=_t(100.0), expiry=_t(1.0), barrier=_t(barrier), is_call=torch.tensor(is_call), kind=kind
    )
    ours = MonteCarlo(num_paths=paths, steps=steps).price(option, params, market)

    reference = ql.BarrierOption(
        ql_kind,
        barrier,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, 100.0),
        ql.EuropeanExercise(_date(365)),
    )
    reference.setPricingEngine(
        ql.MCBarrierEngine(
            _process(100.0, 0.05, 0.02, 0.3),
            "pseudorandom",
            timeSteps=steps,
            brownianBridge=False,
            antitheticVariate=True,
            requiredSamples=paths,
            isBiased=True,
            seed=42,
        )
    )
    _within_stderr(ours, reference.NPV(), math.sqrt(2.0) * reference.errorEstimate())


def test_discrete_barrier_converges_to_continuous_monitoring():
    """AT-30: knock-out prices fall towards the continuous closed form as steps grow."""
    from smallquant.engines.pricing import Analytic, MonteCarlo
    from smallquant.instruments import BarrierOption

    market, params = _engine_market()

    def option(monitoring):
        return BarrierOption(
            strike=_t(100.0),
            expiry=_t(1.0),
            barrier=_t(90.0),
            is_call=torch.tensor(True),
            kind="down_out",
            monitoring=monitoring,
        )

    prices = [float(MonteCarlo(num_paths=65536, steps=steps).price(option("discrete"), params, market)) for steps in (4, 32, 256)]
    continuous = float(Analytic().price(option("continuous"), params, market))
    # Standard errors are about 0.05; the gaps between these step counts are several times larger.
    assert prices[0] > prices[1] + 0.3 > prices[2] + 0.3 > continuous + 0.3


def test_bermudan_on_expiry_only_is_european():
    """An exercise grid of just the expiry gives no early exercise on a finer simulation grid."""
    from smallquant.engines.pricing import MonteCarlo
    from smallquant.instruments import AmericanOption

    market, params = _engine_market()
    flag = torch.tensor(False)
    bermudan = AmericanOption(strike=_t(100.0), expiry=_t(1.0), is_call=flag, exercise_grid=_t([1.0]))
    american = AmericanOption(strike=_t(100.0), expiry=_t(1.0), is_call=flag)
    engine = MonteCarlo(num_paths=65536, steps=50)
    closed = option_price(
        spot=market.spot, strike=_t(100.0), expiry=_t(1.0), rate=market.rate, vol=params.vol, div=market.div, is_call=flag
    )
    # Discounted put payoff standard deviation is below 15, so the standard error is below 0.06.
    assert abs(float(engine.price(bermudan, params, market)) - float(closed)) < 3 * 0.06
    # Exercising on every step is worth more than the European put.
    assert float(engine.price(american, params, market)) > float(closed) + 0.2
