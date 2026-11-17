"""Regression tests for the Fourier engine and the binomial/trinomial trees.

The Heston reference is the Attari single-integral price, which matches QuantLib's
AnalyticHestonEngine to about 1e-12 (tests/models/test_heston_european_vs_quantlib.py).
"""

import pytest
import torch

from smallquant.black_scholes import option_price, option_price_binomial
from smallquant.engines import fourier
from smallquant.engines.pricing import Analytic, Fourier
from smallquant.instruments import EuropeanOption
from smallquant.market import Market
from smallquant.models.heston.approximations import european_option_price as heston_reference
from smallquant.models.params import HestonParams


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


HESTON_SETS = {
    "equity": dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7),
    "high_vol_of_vol": dict(v0=0.04, kappa=1.0, theta=0.09, sigma=1.2, rho=-0.9),
    "positive_rho": dict(v0=0.04, kappa=3.0, theta=0.05, sigma=0.6, rho=0.3),
}
MARKET = dict(spot=_t(100.0), rate=_t(0.03), div=_t(0.01))
STRIKES = _t([80.0, 90.0, 100.0, 115.0, 130.0])


def _heston(name):
    return {key: _t(value) for key, value in HESTON_SETS[name].items()}


def _fourier(name, expiry, method, num_terms, strike=STRIKES):
    params = _heston(name)
    characteristic = fourier.heston_characteristic(expiry=expiry, **MARKET, **params)
    return fourier.price(
        strike=strike, expiry=expiry, is_call=True, characteristic=characteristic,
        method=method, num_terms=num_terms, **MARKET,
    )


def _reference(name, expiry, strike=STRIKES):
    return heston_reference(strike=strike, expiry=expiry, is_call=True, **MARKET, **_heston(name))


@pytest.mark.parametrize("method, num_terms", [("cos", 256), ("carr_madan", 4096)])
def test_fourier_prices_a_batch_of_strikes(method, num_terms):
    expiry = _t(1.0)
    batched = _fourier("equity", expiry, method, num_terms)
    assert batched.shape == STRIKES.shape
    stacked = torch.stack([_fourier("equity", expiry, method, num_terms, strike=k) for k in STRIKES])
    assert torch.allclose(batched, stacked, rtol=0, atol=1e-12)


def test_fourier_broadcasts_strikes_against_expiries():
    expiry = _t([[0.5], [2.0]])
    prices = _fourier("equity", expiry, "cos", 512)
    assert prices.shape == (2, STRIKES.numel())
    assert torch.allclose(prices, _reference("equity", expiry), rtol=0, atol=1e-8)


def test_engines_price_a_batch_of_strikes():
    option = EuropeanOption(strike=STRIKES, expiry=_t(1.0), is_call=torch.tensor(True))
    params = HestonParams(**_heston("equity"))
    market = Market(**MARKET)
    reference = _reference("equity", _t(1.0))
    for engine in (Analytic(), Fourier(), Fourier(method="carr_madan", num_terms=4096)):
        assert torch.allclose(engine.price(option, params, market), reference, rtol=0, atol=1e-7)


@pytest.mark.parametrize("expiry", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("name, tolerance", [("equity", 1e-9), ("positive_rho", 1e-9), ("high_vol_of_vol", 1e-5)])
def test_cos_matches_heston_reference(name, tolerance, expiry):
    prices = _fourier(name, _t(expiry), "cos", 1024)
    assert torch.allclose(prices, _reference(name, _t(expiry)), rtol=0, atol=tolerance)


@pytest.mark.parametrize("expiry", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("name", list(HESTON_SETS))
def test_carr_madan_matches_heston_reference(name, expiry):
    prices = _fourier(name, _t(expiry), "carr_madan", 8192)
    assert torch.allclose(prices, _reference(name, _t(expiry)), rtol=0, atol=1e-10)


@pytest.mark.parametrize("method, num_terms", [("cos", 256), ("carr_madan", 4096)])
@pytest.mark.parametrize("is_call", [True, False])
def test_fourier_matches_black_scholes(method, num_terms, is_call):
    expiry, vol = _t(0.75), _t(0.25)
    characteristic = fourier.black_characteristic(vol=vol, expiry=expiry, **MARKET)
    prices = fourier.price(
        strike=STRIKES, expiry=expiry, is_call=is_call, characteristic=characteristic,
        method=method, num_terms=num_terms, **MARKET,
    )
    closed = option_price(strike=STRIKES, expiry=expiry, vol=vol, is_call=is_call, **MARKET)
    assert torch.allclose(prices, closed, rtol=0, atol=1e-10)


@pytest.mark.parametrize("kind", ["crr", "trinomial"])
@pytest.mark.parametrize("exercise", ["european", "american"])
def test_trees_price_a_batch_of_spots(kind, exercise):
    spots = _t([80.0, 95.0, 100.0, 120.0])
    kwargs = dict(strike=_t(100.0), expiry=_t(1.0), rate=_t(0.05), vol=_t(0.25), div=_t(0.02),
                  is_call=False, exercise=exercise, num_steps=150, kind=kind)
    batched = option_price_binomial(spot=spots, **kwargs)
    stacked = torch.stack([option_price_binomial(spot=spot, **kwargs) for spot in spots])
    assert batched.shape == spots.shape
    assert torch.allclose(batched, stacked, rtol=0, atol=1e-12)


def test_tree_broadcasts_spots_against_strikes_and_flags():
    spots = _t([[90.0], [110.0]])
    strikes = _t([95.0, 100.0, 105.0])
    is_call = torch.tensor([True, False, True])
    prices = option_price_binomial(
        spot=spots, strike=strikes, expiry=_t(0.5), rate=_t(0.03), vol=_t(0.2), div=_t(0.0),
        is_call=is_call, num_steps=400,
    )
    closed = option_price(spot=spots, strike=strikes, expiry=_t(0.5), rate=_t(0.03), vol=_t(0.2),
                          div=_t(0.0), is_call=is_call)
    assert prices.shape == (2, 3)
    assert torch.allclose(prices, closed, rtol=0, atol=2e-2)


def test_tree_accepts_a_python_bool():
    kwargs = dict(spot=_t(100.0), strike=_t(100.0), expiry=_t(1.0), rate=_t(0.05), vol=_t(0.2), div=_t(0.0))
    assert float(option_price_binomial(is_call=True, **kwargs)) == pytest.approx(
        float(option_price_binomial(is_call=torch.tensor(True), **kwargs))
    )
