"""Vanilla Black-Scholes calls/puts against QuantLib's AnalyticEuropeanEngine."""

import pytest
import torch

from smallquant.models.black_scholes import price

ql = pytest.importorskip("tests.reference.ql")

pytestmark = pytest.mark.quantlib

REL = 1e-9
ABS = 1e-10

# (spot, strike, days, rate, vol)
CASES = {
    "atm_1y": (100.0, 100.0, 365, 0.05, 0.20),
    "itm": (120.0, 100.0, 182, 0.03, 0.25),
    "otm": (80.0, 100.0, 182, 0.03, 0.25),
    "deep_itm": (200.0, 100.0, 365, 0.05, 0.20),
    "deep_otm": (50.0, 100.0, 91, 0.05, 0.20),
    "one_week": (100.0, 100.0, 7, 0.05, 0.20),
    "five_years": (100.0, 110.0, 1825, 0.04, 0.30),
    "high_vol": (100.0, 100.0, 365, 0.05, 1.00),
    "low_vol": (100.0, 105.0, 365, 0.05, 0.05),
    "zero_rate": (100.0, 100.0, 365, 0.00, 0.20),
    "negative_rate": (100.0, 100.0, 365, -0.01, 0.20),
}


def price_and_greeks(spot, strike, days, rate, vol, div=0.0, is_call=True):
    """SmallQuant price, first-order Greeks and gamma, all from autograd."""
    f64 = dict(dtype=torch.float64)
    S = torch.as_tensor(spot, **f64).requires_grad_()
    K = torch.as_tensor(strike, **f64)
    T = (torch.as_tensor(days, **f64) / 365.0).requires_grad_()
    r = torch.as_tensor(rate, **f64).requires_grad_()
    sigma = torch.as_tensor(vol, **f64).requires_grad_()
    q = torch.as_tensor(div, **f64)

    value = price(spot=S, strike=K, expiry=T, rate=r, vol=sigma, div=q, is_call=is_call)
    delta, dexpiry, rho, vega = torch.autograd.grad(
        value.sum(), (S, T, r, sigma), create_graph=True
    )
    (gamma,) = torch.autograd.grad(delta.sum(), S)
    return {
        "price": value.detach(),
        "delta": delta.detach(),
        "gamma": gamma,
        "vega": vega.detach(),
        "rho": rho.detach(),
        "dexpiry": dexpiry.detach(),
    }


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_matches_quantlib(case):
    spot, strike, days, rate, vol = case
    ours = price_and_greeks(spot, strike, days, rate, vol)
    ref = ql.european(spot=spot, strike=strike, days=days, rate=rate, vol=vol)

    for name, value in ours.items():
        assert value.item() == pytest.approx(getattr(ref, name), rel=REL, abs=ABS), name


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_put_matches_quantlib(case):
    spot, strike, days, rate, vol = case
    ours = price_and_greeks(spot, strike, days, rate, vol, is_call=False)
    ref = ql.european(
        spot=spot, strike=strike, days=days, rate=rate, vol=vol, is_call=False
    )

    for name, value in ours.items():
        assert value.item() == pytest.approx(getattr(ref, name), rel=REL, abs=ABS), name


def test_dividend_yield_matches_quantlib():
    ours = price_and_greeks(100.0, 100.0, 365, 0.05, 0.20, div=0.03)
    ref = ql.european(spot=100.0, strike=100.0, days=365, rate=0.05, vol=0.20, div=0.03)
    for name, value in ours.items():
        assert value.item() == pytest.approx(getattr(ref, name), rel=REL, abs=ABS), name


def test_textbook_value():
    # Hull, Options, Futures, and Other Derivatives: S=K=100, T=1, r=5%, vol=20%.
    ours = price_and_greeks(100.0, 100.0, 365, 0.05, 0.20)["price"].item()
    ref = ql.european(spot=100.0, strike=100.0, days=365, rate=0.05, vol=0.20).price
    assert ours == pytest.approx(10.4506, abs=5e-5)
    assert ref == pytest.approx(10.4506, abs=5e-5)


def test_batch_matches_quantlib():
    gen = torch.Generator().manual_seed(0)
    n = 200
    spot = 50.0 + 100.0 * torch.rand(n, generator=gen, dtype=torch.float64)
    strike = 50.0 + 100.0 * torch.rand(n, generator=gen, dtype=torch.float64)
    days = torch.randint(1, 3 * 365, (n,), generator=gen).to(torch.float64)
    rate = -0.01 + 0.09 * torch.rand(n, generator=gen, dtype=torch.float64)
    vol = 0.05 + 0.75 * torch.rand(n, generator=gen, dtype=torch.float64)

    ours = price_and_greeks(spot, strike, days, rate, vol)

    for i in range(n):
        ref = ql.european(
            spot=spot[i].item(),
            strike=strike[i].item(),
            days=int(days[i].item()),
            rate=rate[i].item(),
            vol=vol[i].item(),
        )
        for name, values in ours.items():
            assert values[i].item() == pytest.approx(
                getattr(ref, name), rel=REL, abs=ABS
            ), (name, i)


def test_python_floats_default_to_float64():
    value = price(spot=100.0, strike=100.0, expiry=1.0, rate=0.05, vol=0.2)
    assert value.dtype == torch.float64
