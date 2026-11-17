"""Stefanica-Radoicic fast implied volatility approximation.

Accuracy tolerances come from the paper (Stefanica and Radoicic, "An explicit implied
volatility formula", IJTAF 20(7), 2017): its abstract states the relative error of the
approximate implied volatility is under 10% for maturities below five years and implied
volatilities below 150%, and is uniformly bounded for any moneyness, maturity and
volatility. The approximation also solves the Black equation exactly with N replaced by
Polya's CDF, which is within 0.0032 of N, so the true Black price at the returned
volatility can differ from the input price by at most 0.0032 * (F + K) * D.
"""

import math
import time

import pytest
import torch

import smallquant.black_scholes as bs
from smallquant.black_scholes import ImpliedVolMethod, ImpliedVolUnderlyingDistribution, option_price
from smallquant.black_scholes.implied_vol_approximation import implied_vol
from smallquant.math.distributions import normal_cdf

F64 = torch.float64
PAPER_REL_TOL = 0.10  # paper: relative error < 10% for T < 5y and vol < 150%
POLYA_CDF_ERROR = 0.0032  # max |N(t) - Polya(t)| is 0.00315


def _t(value):
    return torch.as_tensor(value, dtype=F64)


@pytest.fixture(scope="module")
def grid():
    """Strike x expiry x vol grid with nonzero rate and dividend, calls and puts."""
    moneyness = torch.linspace(0.5, 2.0, 31, dtype=F64)
    expiries = _t([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    vols = torch.linspace(0.05, 1.0, 20, dtype=F64)
    strike, expiry, vol = torch.meshgrid(100.0 * moneyness, expiries, vols, indexing="ij")
    is_call = torch.tensor([True, False]).view(2, 1, 1, 1)
    strike, expiry, vol, is_call = torch.broadcast_tensors(strike, expiry, vol, is_call)
    spot = torch.full_like(strike, 100.0)
    rate = torch.full_like(strike, 0.03)
    div = torch.full_like(strike, 0.01)
    market = dict(spot=spot, strike=strike, expiry=expiry, rate=rate, div=div)
    price = option_price(vol=vol, is_call=is_call, **market)
    # Deep in-the-money, short, low-vol points whose time value is below the solvers'
    # intrinsic tolerance carry no volatility information; both solvers return zero there.
    call = option_price(vol=vol, is_call=True, **market)
    put = option_price(vol=vol, is_call=False, **market)
    informative = torch.minimum(call, put) > 1e-8
    return dict(market=market, price=price, vol=vol, is_call=is_call, informative=informative)


def test_polya_cdf_error_bound():
    x = torch.linspace(-10.0, 10.0, 200_001, dtype=F64)
    polya = 0.5 + 0.5 * torch.sign(x) * torch.sqrt(-torch.expm1(-2.0 * x * x / math.pi))
    assert (polya - normal_cdf(x)).abs().max().item() < POLYA_CDF_ERROR


def test_accuracy_on_grid(grid):
    vol = implied_vol(grid["price"], is_call=grid["is_call"], **grid["market"])
    ok = grid["informative"]
    assert ok.float().mean() > 0.85
    assert torch.isfinite(vol[ok]).all()
    abs_err = (vol - grid["vol"]).abs()[ok]
    rel_err = abs_err / grid["vol"][ok]
    print(
        f"\nFAST_APPROX over {int(ok.sum())} grid options: max abs err {abs_err.max():.4f}, "
        f"max rel err {rel_err.max():.4f}, median rel err {rel_err.median():.4f}"
    )
    assert rel_err.max() < PAPER_REL_TOL

    # Price-space check: Black price at the approximate vol vs the input price.
    m = grid["market"]
    repriced = option_price(vol=vol, is_call=grid["is_call"], **m)
    forward = m["spot"] * torch.exp((m["rate"] - m["div"]) * m["expiry"])
    discount = torch.exp(-m["rate"] * m["expiry"])
    bound = POLYA_CDF_ERROR * (forward + m["strike"]) * discount
    assert ((repriced - grid["price"]).abs() <= bound).all()


def test_relative_error_uniformly_bounded_beyond_trading_range():
    """Wide random sample (strike/forward e^-3..e^3, T to 30y, vol to 300%)."""
    gen = torch.Generator().manual_seed(7)
    n = 100_000
    strike = torch.exp(torch.empty(n, dtype=F64).uniform_(-3.0, 3.0, generator=gen))
    expiry = torch.exp(torch.empty(n, dtype=F64).uniform_(math.log(0.01), math.log(30.0), generator=gen))
    vol = torch.exp(torch.empty(n, dtype=F64).uniform_(math.log(0.01), math.log(3.0), generator=gen))
    forward = torch.ones_like(strike)
    discount = torch.full_like(strike, 0.9)
    market = dict(forward=forward, strike=strike, expiry=expiry, discount=discount)
    call = option_price(vol=vol, is_call=True, **market)
    put = option_price(vol=vol, is_call=False, **market)
    informative = torch.minimum(call, put) > 1e-8
    approx = implied_vol(put, is_call=False, **market)
    rel_err = ((approx - vol).abs() / vol)[informative]
    assert torch.isfinite(rel_err).all()
    assert rel_err.max() < PAPER_REL_TOL


def test_at_the_money_forward():
    one = torch.ones(4, dtype=F64)
    vol = _t([0.01, 0.2, 0.5, 1.0])
    price = option_price(forward=one, strike=one, expiry=one, discount=one, vol=vol, is_call=True)
    approx = implied_vol(price, forward=one, strike=one, expiry=one, discount=one)
    torch.testing.assert_close(approx, vol, rtol=5e-3, atol=0.0)


def test_lib_dispatch_matches(grid):
    direct = implied_vol(grid["price"], is_call=grid["is_call"], **grid["market"])
    via_lib = bs.implied_vol(
        grid["price"], is_call=grid["is_call"], method=ImpliedVolMethod.FAST_APPROX, **grid["market"]
    )
    torch.testing.assert_close(via_lib, direct, rtol=0.0, atol=0.0, equal_nan=True)
    via_alias = bs.implied_vol_approx(grid["price"], is_call=grid["is_call"], **grid["market"])
    torch.testing.assert_close(via_alias, direct, rtol=0.0, atol=0.0, equal_nan=True)


def test_lib_fast_approx_rejects_normal():
    one = _t(1.0)
    with pytest.raises(ValueError, match="lognormal"):
        bs.implied_vol(
            0.1 * one, forward=one, strike=one, expiry=one, discount=one,
            method=ImpliedVolMethod.FAST_APPROX,
            underlying_distribution=ImpliedVolUnderlyingDistribution.NORMAL,
        )


def test_spot_and_forward_forms_agree(grid):
    m = grid["market"]
    from_spot = implied_vol(grid["price"], is_call=grid["is_call"], **m)
    forward = m["spot"] * torch.exp((m["rate"] - m["div"]) * m["expiry"])
    discount = torch.exp(-m["rate"] * m["expiry"])
    from_discount = implied_vol(
        grid["price"], forward=forward, strike=m["strike"], expiry=m["expiry"], discount=discount,
        is_call=grid["is_call"],
    )
    from_rate = implied_vol(
        grid["price"], forward=forward, strike=m["strike"], expiry=m["expiry"], rate=m["rate"],
        is_call=grid["is_call"],
    )
    ok = grid["informative"]
    torch.testing.assert_close(from_discount[ok], from_spot[ok], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(from_rate[ok], from_spot[ok], rtol=1e-10, atol=1e-12)


def test_spot_without_div_means_zero_div():
    one = _t(1.0)
    market = dict(spot=100 * one, strike=110 * one, expiry=one, rate=0.02 * one)
    price = option_price(vol=0.3 * one, is_call=True, **market)
    torch.testing.assert_close(
        implied_vol(price, **market), implied_vol(price, div=0 * one, **market), rtol=0.0, atol=0.0
    )


def test_out_of_bounds_and_intrinsic():
    spot = torch.full((6,), 100.0, dtype=F64)
    strike = _t([90.0, 90.0, 110.0, 110.0, 90.0, 110.0])
    expiry = torch.ones(6, dtype=F64)
    rate = torch.full((6,), 0.05, dtype=F64)
    is_call = torch.tensor([True, True, False, False, True, False])
    discounted_strike = strike * math.exp(-0.05)
    intrinsic = torch.where(is_call, spot - discounted_strike, discounted_strike - spot)
    upper = torch.where(is_call, spot, discounted_strike)
    price = torch.stack([
        intrinsic[0] - 1e-3,  # call below intrinsic
        upper[1] + 1e-3,  # call above spot
        intrinsic[2] - 1e-3,  # put below intrinsic
        upper[3] + 1e-3,  # put above discounted strike
        intrinsic[4],  # call at intrinsic
        intrinsic[5],  # put at intrinsic
    ])
    vol = implied_vol(price, spot=spot, strike=strike, expiry=expiry, rate=rate, is_call=is_call)
    assert torch.isnan(vol[:4]).all()
    torch.testing.assert_close(vol[4:], torch.zeros(2, dtype=F64))

    newton = bs.implied_vol_newton(price, spot=spot, strike=strike, expiry=expiry, rate=rate, is_call=is_call)
    torch.testing.assert_close(vol, newton, equal_nan=True)


def test_negative_price_is_nan():
    one = _t(1.0)
    vol = implied_vol(-one, forward=100 * one, strike=120 * one, expiry=one, discount=one)
    assert torch.isnan(vol)


def test_gradients_finite_in_interior(grid):
    m = grid["market"]
    ok = grid["informative"]
    price = grid["price"][ok].clone().requires_grad_(True)
    spot = m["spot"][ok].clone().requires_grad_(True)
    strike = m["strike"][ok].clone().requires_grad_(True)
    expiry = m["expiry"][ok].clone().requires_grad_(True)
    rate = m["rate"][ok].clone().requires_grad_(True)
    div = m["div"][ok].clone().requires_grad_(True)
    vol = implied_vol(price, spot=spot, strike=strike, expiry=expiry, rate=rate, div=div,
                      is_call=grid["is_call"][ok])
    grads = torch.autograd.grad(vol.sum(), (price, spot, strike, expiry, rate, div))
    for grad in grads:
        assert torch.isfinite(grad).all()
    # Volatility rises with the option price.
    assert (grads[0] > 0).all()


def test_gradients_finite_with_masked_elements():
    """Out-of-bounds and intrinsic elements must not leak NaN into the backward pass."""
    one = torch.ones(4, dtype=F64)
    price = _t([12.0, -1.0, 200.0, 10.0]).requires_grad_(True)
    strike = (100 * one).requires_grad_(True)
    vol = implied_vol(price, forward=100 * one, strike=strike, expiry=one, discount=one)
    assert torch.isfinite(vol[0]) and torch.isnan(vol[1:3]).all()
    grads = torch.autograd.grad(vol[0] + vol[3], (price, strike))
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_bool_flag_and_dtype():
    one32 = torch.tensor(1.0, dtype=torch.float32)
    price = option_price(forward=one32, strike=1.1 * one32, expiry=one32, discount=one32,
                         vol=0.25 * one32, is_call=False)
    vol = implied_vol(price, forward=one32, strike=1.1 * one32, expiry=one32, discount=one32, is_call=False)
    assert vol.dtype == torch.float32
    assert abs(vol.item() - 0.25) < PAPER_REL_TOL * 0.25
    vol64 = implied_vol(price, forward=one32, strike=1.1 * one32, expiry=one32, discount=one32,
                        is_call=False, dtype=F64)
    assert vol64.dtype == F64


def test_rejects_python_numbers():
    one = _t(1.0)
    with pytest.raises(TypeError):
        implied_vol(0.1, forward=one, strike=one, expiry=one, discount=one)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(), "pass exactly one of spot and forward"),
        (dict(spot=True, forward=True, rate=True), "pass exactly one of spot and forward"),
        (dict(spot=True, rate=True, discount=True), "pass at most one of rate and discount"),
        (dict(spot=True), "spot needs rate"),
        (dict(forward=True, rate=True, div=True), "div applies to spot only"),
        (dict(forward=True), "forward needs discount or rate"),
    ],
)
def test_invalid_combinations_match_newton(kwargs, message):
    one = _t(1.0)
    kwargs = {key: one for key in kwargs}
    for solver in (implied_vol, bs.implied_vol_newton):
        with pytest.raises(ValueError, match=message):
            solver(0.1 * one, strike=one, expiry=one, **kwargs)


def test_faster_than_newton_on_large_chain():
    n = 1_000_000
    gen = torch.Generator().manual_seed(1)
    spot = torch.full((n,), 100.0, dtype=F64)
    strike = 100.0 * torch.empty(n, dtype=F64).uniform_(0.5, 2.0, generator=gen)
    expiry = torch.empty(n, dtype=F64).uniform_(0.05, 5.0, generator=gen)
    vol = torch.empty(n, dtype=F64).uniform_(0.05, 1.0, generator=gen)
    rate = torch.full((n,), 0.03, dtype=F64)
    div = torch.full((n,), 0.01, dtype=F64)
    is_call = torch.rand(n, generator=gen) < 0.5
    market = dict(spot=spot, strike=strike, expiry=expiry, rate=rate, div=div, is_call=is_call)
    price = option_price(vol=vol, **market)

    def timed(fn):
        fn()  # warm up
        start = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - start

    approx, approx_s = timed(lambda: bs.implied_vol(price, method=ImpliedVolMethod.FAST_APPROX, **market))
    exact, newton_s = timed(lambda: bs.implied_vol(price, method=ImpliedVolMethod.NEWTON, **market))
    print(
        f"\n{n:,} options on {price.device}, torch {torch.__version__}, float64: "
        f"FAST_APPROX {approx_s * 1e3:.1f} ms, NEWTON {newton_s * 1e3:.1f} ms, "
        f"speedup {newton_s / approx_s:.1f}x"
    )
    assert approx_s < newton_s
    both = torch.isfinite(approx) & torch.isfinite(exact) & (exact > 0)
    assert ((approx - exact).abs() / exact)[both].max() < PAPER_REL_TOL
