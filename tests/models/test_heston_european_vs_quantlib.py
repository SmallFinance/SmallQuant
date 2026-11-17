"""Heston European prices (Attari 2004) against QuantLib and the engine path."""

import itertools

import pytest
import torch

import QuantLib as ql

import smallquant as sq
from smallquant.engines.pricing import Analytic, Fourier
from smallquant.instruments import EuropeanOption
from smallquant.market import Market
from smallquant.models.heston.approximations import european_option_price
from smallquant.models.params import HestonParams

pytestmark = pytest.mark.quantlib

# QuantLib's adaptive engine agrees with its Gauss-Laguerre engine to ~1e-13
# relative on these cases, except where the price itself is ~1e-6 or less and
# both sit on the double-precision floor of ``F - K * (...)``. Hence the small
# absolute term next to the 1e-8 relative one.
REL = 1e-8
ABS = 1e-10

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()

SPOT, RATE, DIV = 100.0, 0.03, 0.01

# (v0, kappa, theta, sigma, rho)
PARAMS = {
    "equity": (0.04, 1.5, 0.04, 0.3, -0.7),
    "high_vol_of_vol": (0.09, 0.5, 0.06, 1.2, -0.9),
    "positive_rho": (0.02, 3.0, 0.05, 0.6, 0.3),
    "low_v0_high_vol_of_vol": (0.01, 1.0, 0.02, 1.5, -0.6),
    "zero_rho_fast_reversion": (0.25, 5.0, 0.16, 0.8, 0.0),
}

# 18 days is about 0.05y; 3650 days is 10y.
DAYS = [18, 91, 365, 1825, 3650]
# Strikes in units of 0.2 * sqrt(expiry) of log-moneyness.
MONEYNESS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(EVAL, rate, DAY_COUNT, ql.Continuous))


def _quantlib_price(strike, days, params, is_call):
    ql.Settings.instance().evaluationDate = EVAL
    v0, kappa, theta, sigma, rho = params
    process = ql.HestonProcess(
        _flat(RATE), _flat(DIV), ql.QuoteHandle(ql.SimpleQuote(SPOT)), v0, kappa, theta, sigma, rho
    )
    # Adaptive Gauss-Lobatto with a tight tolerance as the reference.
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-13, 1_000_000)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(EVAL + days))
    option.setPricingEngine(engine)
    return option.NPV()


def _price(params, **kwargs):
    v0, kappa, theta, sigma, rho = (_t(value) for value in params)
    return european_option_price(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho, **kwargs)


@pytest.mark.parametrize("name", list(PARAMS))
def test_prices_match_quantlib(name):
    params = PARAMS[name]
    cases = [
        (SPOT * float(torch.exp(_t(m * 0.2 * (days / 365.0) ** 0.5))), days, is_call)
        for days, m, is_call in itertools.product(DAYS, MONEYNESS, (True, False))
    ]
    strike = _t([case[0] for case in cases])
    expiry = _t([case[1] / 365.0 for case in cases])
    is_call = torch.tensor([case[2] for case in cases])
    ours = _price(
        params, spot=_t(SPOT), strike=strike, expiry=expiry, rate=_t(RATE), div=_t(DIV), is_call=is_call
    )
    ref = _t([_quantlib_price(k, days, params, call) for k, days, call in cases])
    assert ours.shape == ref.shape
    error = (ours - ref).abs()
    bound = REL * ref.abs() + ABS
    worst = int(torch.argmax(error / bound))
    assert bool((error <= bound).all()), (
        f"{name}: strike={cases[worst][0]:.4f} days={cases[worst][1]} call={cases[worst][2]} "
        f"ours={float(ours[worst])!r} quantlib={float(ref[worst])!r}"
    )


@pytest.mark.parametrize("days", [18, 365, 3650])
@pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
def test_short_and_long_expiry_relative_accuracy(days, strike):
    # Prices that are not tiny agree to 1e-8 relative on their own.
    params = PARAMS["equity"]
    for is_call in (True, False):
        ref = _quantlib_price(strike, days, params, is_call)
        if abs(ref) < 1e-3:
            continue
        ours = _price(
            params,
            spot=_t(SPOT),
            strike=_t(strike),
            expiry=_t(days / 365.0),
            rate=_t(RATE),
            div=_t(DIV),
            is_call=is_call,
        )
        assert float(ours) == pytest.approx(ref, rel=REL, abs=0.0)


# The COS engine is accurate for moderate parameters only, so the comparison
# uses the equity set. It prices one strike at a time.
@pytest.mark.parametrize("expiry", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("strike,is_call", [(80.0, True), (100.0, False), (120.0, True)])
def test_matches_engine_path(expiry, strike, is_call):
    v0, kappa, theta, sigma, rho = PARAMS["equity"]
    params = HestonParams(v0=_t(v0), kappa=_t(kappa), theta=_t(theta), sigma=_t(sigma), rho=_t(rho))
    market = Market(spot=_t(SPOT), rate=_t(RATE), div=_t(DIV))
    flag = torch.tensor(is_call)
    option = EuropeanOption(strike=_t(strike), expiry=_t(expiry), is_call=flag)
    ours = european_option_price(
        spot=_t(SPOT),
        strike=_t(strike),
        expiry=_t(expiry),
        rate=_t(RATE),
        div=_t(DIV),
        v0=params.v0,
        kappa=params.kappa,
        theta=params.theta,
        sigma=params.sigma,
        rho=params.rho,
        is_call=flag,
    )
    assert float(sq.price(option, params, market)) == pytest.approx(float(ours), abs=1e-8)
    assert float(Analytic().price(option, params, market)) == pytest.approx(float(ours), abs=1e-8)
    fine = Fourier(method="cos", num_terms=256).price(option, params, market)
    assert float(fine) == pytest.approx(float(ours), abs=1e-10)


def test_batched_equals_stacked_single_calls():
    strike = _t([80.0, 100.0, 125.0]).reshape(1, 1, 3)
    expiry = _t([0.05, 1.0, 10.0]).reshape(1, 3, 1)
    rows = [PARAMS["equity"], PARAMS["high_vol_of_vol"]]
    v0, kappa, theta, sigma, rho = (_t([row[i] for row in rows]).reshape(2, 1, 1) for i in range(5))
    is_call = torch.tensor([True, False, True]).reshape(1, 1, 3)
    batched = european_option_price(
        spot=_t(SPOT),
        strike=strike,
        expiry=expiry,
        rate=_t(RATE),
        div=_t(DIV),
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        is_call=is_call,
    )
    assert batched.shape == (2, 3, 3)
    singles = torch.stack(
        [
            european_option_price(
                spot=_t(SPOT),
                strike=strike[0, 0, k],
                expiry=expiry[0, j, 0],
                rate=_t(RATE),
                div=_t(DIV),
                v0=v0[i, 0, 0],
                kappa=kappa[i, 0, 0],
                theta=theta[i, 0, 0],
                sigma=sigma[i, 0, 0],
                rho=rho[i, 0, 0],
                is_call=bool(is_call[0, 0, k]),
            )
            for i in range(2)
            for j in range(3)
            for k in range(3)
        ]
    ).reshape(2, 3, 3)
    torch.testing.assert_close(batched, singles, rtol=1e-13, atol=1e-13)


def _greek_inputs():
    return dict(
        spot=_t([100.0, 100.0, 100.0]),
        strike=_t([90.0, 100.0, 115.0]),
        expiry=_t([0.05, 1.0, 10.0]),
        rate=_t(RATE),
        div=_t(DIV),
        v0=_t([0.04, 0.09, 0.02]),
        kappa=_t([1.5, 0.5, 3.0]),
        theta=_t([0.04, 0.06, 0.05]),
        sigma=_t([0.3, 1.2, 0.6]),
        rho=_t([-0.7, -0.9, 0.3]),
        is_call=torch.tensor([True, False, True]),
    )


def test_gradients_are_finite():
    inputs = _greek_inputs()
    spot = inputs["spot"].clone().requires_grad_()
    v0 = inputs["v0"].clone().requires_grad_()
    rho = inputs["rho"].clone().requires_grad_()
    price = european_option_price(**{**inputs, "spot": spot, "v0": v0, "rho": rho})
    grads = torch.autograd.grad(price.sum(), (spot, v0, rho))
    for grad in grads:
        assert grad.shape == (3,)
        assert bool(torch.isfinite(grad).all())
    # Calls rise with spot, puts fall; both rise with the initial variance.
    assert grads[0][0] > 0 and grads[0][1] < 0 and grads[0][2] > 0
    assert bool((grads[1] > 0).all())


@pytest.mark.parametrize("name,step", [("spot", 1e-3), ("v0", 1e-6), ("rho", 1e-5)])
def test_autograd_matches_central_bump(name, step):
    inputs = _greek_inputs()
    leaf = inputs[name].clone().requires_grad_()
    price = european_option_price(**{**inputs, name: leaf})
    (grad,) = torch.autograd.grad(price.sum(), leaf)
    up = european_option_price(**{**inputs, name: inputs[name] + step})
    down = european_option_price(**{**inputs, name: inputs[name] - step})
    bump = (up - down) / (2.0 * step)
    torch.testing.assert_close(grad, bump, rtol=1e-6, atol=1e-8)


def test_put_call_parity_and_zero_expiry():
    common = dict(
        spot=_t(SPOT),
        strike=_t([90.0, 110.0]),
        rate=_t(RATE),
        div=_t(DIV),
        v0=_t(0.04),
        kappa=_t(1.5),
        theta=_t(0.04),
        sigma=_t(0.3),
        rho=_t(-0.7),
    )
    expiry = _t(2.0)
    call = european_option_price(expiry=expiry, is_call=True, **common)
    put = european_option_price(expiry=expiry, is_call=False, **common)
    forward_gap = SPOT * torch.exp(-_t(DIV) * expiry) - common["strike"] * torch.exp(-_t(RATE) * expiry)
    torch.testing.assert_close(call - put, forward_gap, rtol=0.0, atol=1e-12)
    at_expiry = european_option_price(expiry=_t(0.0), is_call=True, **common)
    torch.testing.assert_close(at_expiry, _t([10.0, 0.0]))


def test_dtype_and_tensor_contract():
    kwargs = dict(
        spot=torch.tensor(100.0, dtype=torch.float32),
        strike=torch.tensor(100.0, dtype=torch.float32),
        expiry=torch.tensor(1.0, dtype=torch.float32),
        rate=torch.tensor(0.03, dtype=torch.float32),
        div=torch.tensor(0.0, dtype=torch.float32),
        v0=torch.tensor(0.04, dtype=torch.float32),
        kappa=torch.tensor(1.5, dtype=torch.float32),
        theta=torch.tensor(0.04, dtype=torch.float32),
        sigma=torch.tensor(0.3, dtype=torch.float32),
        rho=torch.tensor(-0.7, dtype=torch.float32),
        is_call=True,
    )
    single = european_option_price(**kwargs)
    assert single.dtype == torch.float32
    double = european_option_price(**kwargs, dtype=torch.float64)
    assert double.dtype == torch.float64
    assert float(single) == pytest.approx(float(double), rel=1e-5)
    with pytest.raises(TypeError):
        european_option_price(**{**kwargs, "spot": 100.0})
