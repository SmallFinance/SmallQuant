"""Coverage for the pricing surface, Greeks, engines, curves and schedules."""

import math

import pytest
import torch

import smallquant as sq
from smallquant.calibration import calibrate
from smallquant.engines.pricing import Analytic, Fourier, Lattice, MonteCarlo, PDE
from smallquant.instruments import AmericanOption, EuropeanOption
from smallquant.market import Market
from smallquant.math.interp import linear
from smallquant.math.quadrature import gauss_legendre, simpson
from smallquant.math.roots import newton
from smallquant.models.params import BlackScholesParams, HestonParams, SABRParams
from smallquant.rates.bootstrap import bootstrap, par_swap_rate
from smallquant.rates.curve import DiscountCurve
from smallquant.time import (
    TARGET,
    ActualActual,
    Business252,
    BusinessDayConvention,
    Date,
    DateGeneration,
    Period,
    Schedule,
    TimeUnit,
)
from smallquant.vol.surface import dupire_local_vol, svi_butterfly_ok, svi_total_variance

ql = pytest.importorskip("QuantLib")


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def test_package_surface():
    for name in (
        "math",
        "rates",
        "vol",
        "models",
        "instruments",
        "engines",
        "greeks",
        "calibration",
        "price",
        "Market",
        "set_default_dtype",
        "set_validation",
    ):
        assert hasattr(sq, name)


def test_dtype_promotion_and_rejection():
    with pytest.raises(TypeError):
        sq.models.black_scholes.price(
            spot=1.0,
            strike=_t(1.0),
            expiry=_t(1.0),
            rate=_t(0.0),
            vol=_t(0.2),
            div=_t(0.0),
            is_call=torch.tensor(True),
        )
    mixed = sq.models.black_scholes.price(
        spot=torch.tensor(100.0, dtype=torch.float32),
        strike=_t(100.0),
        expiry=_t(1.0),
        rate=_t(0.01),
        vol=_t(0.2),
        div=_t(0.0),
        is_call=torch.tensor(True),
    )
    assert mixed.dtype == torch.float64
    with pytest.raises(ValueError):
        sq.models.black_scholes.price(
            spot=torch.tensor(100.0, device="cpu"),
            strike=torch.tensor(100.0, device="meta"),
            expiry=_t(1.0),
            rate=_t(0.0),
            vol=_t(0.2),
            div=_t(0.0),
            is_call=torch.tensor(True),
        )


def test_validation_names_the_argument():
    with sq.validation(True):
        with pytest.raises(ValueError, match="vol"):
            sq.models.black_scholes.price(
                spot=_t(100.0),
                strike=_t(100.0),
                expiry=_t(1.0),
                rate=_t(0.0),
                vol=_t(-0.2),
                div=_t(0.0),
                is_call=torch.tensor(True),
            )


def test_implied_vol_round_trip_and_vega():
    spot, strike, expiry, rate, div = _t(100.0), _t(100.0), _t(1.0), _t(0.03), _t(0.01)
    vol = _t(0.2)
    flag = torch.tensor(True)
    value = sq.models.black_scholes.price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag
    )
    price = value.detach().requires_grad_(True)
    solved = sq.vol.implied_vol(price, spot=spot, strike=strike, expiry=expiry, rate=rate, div=div, is_call=flag)
    assert float(solved.detach()) == pytest.approx(0.2, rel=1e-8, abs=1e-8)
    (grad,) = torch.autograd.grad(solved, price)
    spot_g = spot.detach().requires_grad_(True)
    model = sq.models.black_scholes.price(
        spot=spot_g, strike=strike, expiry=expiry, rate=rate, vol=solved.detach(), div=div, is_call=flag
    )
    (vega,) = torch.autograd.grad(model, spot_g)
    # Vega here is delta. Recompute vega from the solved volatility.
    vol_g = solved.detach().requires_grad_(True)
    model = sq.models.black_scholes.price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol_g, div=div, is_call=flag
    )
    (vega,) = torch.autograd.grad(model, vol_g)
    assert float(grad) == pytest.approx(1.0 / float(vega), rel=1e-6, abs=1e-6)


def test_greeks_match_autograd():
    spot = _t(100.0).requires_grad_()
    vol = _t(0.25).requires_grad_()
    inputs = {
        "spot": spot,
        "strike": _t(105.0),
        "expiry": _t(1.0),
        "rate": _t(0.02),
        "vol": vol,
        "div": _t(0.0),
    }

    def black(data):
        return sq.models.black_scholes.price(
            spot=data["spot"],
            strike=data["strike"],
            expiry=data["expiry"],
            rate=data["rate"],
            vol=data["vol"],
            div=data["div"],
            is_call=torch.tensor(True),
        )

    value = black(inputs)
    first = sq.greeks.first_order(value, {"spot": spot, "vol": vol})
    (delta,) = torch.autograd.grad(value, spot, retain_graph=True)
    assert torch.allclose(first["spot"], delta)
    bumped = sq.greeks.bump(
        black,
        {name: value.detach() for name, value in inputs.items()},
    )
    assert float(bumped["spot"]) == pytest.approx(float(delta), rel=1e-3, abs=1e-3)
    frozen = {name: value.detach() for name, value in inputs.items()}

    def spot_vol(data):
        return black({**frozen, "spot": data["spot"], "vol": data["vol"]})

    gamma = sq.greeks.second_order(
        spot_vol,
        {"spot": spot.detach(), "vol": vol.detach()},
        pairs=[("spot", "spot")],
    )[("spot", "spot")]
    assert gamma.shape == spot.shape
    bump_spot = spot.detach()
    step = _t(1e-2)
    up = black({**{k: v.detach() for k, v in inputs.items()}, "spot": bump_spot + step})
    mid = black({**{k: v.detach() for k, v in inputs.items()}, "spot": bump_spot})
    down = black({**{k: v.detach() for k, v in inputs.items()}, "spot": bump_spot - step})
    finite = (up - 2 * mid + down) / step**2
    assert float(gamma) == pytest.approx(float(finite), rel=1e-3, abs=1e-4)


def test_engines_agree_on_a_european_call():
    spot, strike, expiry, rate, vol, div = _t(100.0), _t(100.0), _t(1.0), _t(0.05), _t(0.2), _t(0.0)
    flag = torch.tensor(True)
    closed = sq.models.black_scholes.price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag
    )
    market = Market(spot=spot, rate=rate, div=div)
    option = EuropeanOption(strike=strike, expiry=expiry, is_call=flag)
    params = BlackScholesParams(vol=vol)
    analytic = sq.price(option, params, market)
    cosine = Fourier(method="cos", num_terms=160).price(option, params, market)
    tree = Lattice(num_steps=400, kind="crr").price(option, params, market)
    pde = PDE(num_space=160, num_time=160).price(option, params, market)
    assert float(analytic) == pytest.approx(float(closed), rel=1e-12)
    assert float(cosine) == pytest.approx(float(closed), abs=1e-3)
    assert float(tree) == pytest.approx(float(closed), abs=2e-2)
    assert float(pde) == pytest.approx(float(closed), abs=2e-2)


def test_sabr_atm_is_finite():
    forward = _t(100.0).requires_grad_()
    vol = sq.models.sabr.implied_vol(
        forward=forward,
        strike=forward.detach(),
        expiry=_t(1.0),
        alpha=_t(0.2),
        beta=_t(0.7),
        rho=_t(-0.3),
        nu=_t(0.4),
    )
    assert torch.isfinite(vol)
    (grad,) = torch.autograd.grad(vol, forward)
    assert torch.isfinite(grad)


def test_heston_characteristic_matches_black_when_vol_of_vol_vanishes():
    spot, rate, div, expiry, vol = _t(100.0), _t(0.01), _t(0.0), _t(0.5), _t(0.2)
    u = torch.tensor([0.5, 1.5, 3.0], dtype=torch.float64)
    heston = sq.models.heston.characteristic_function(
        u, spot=spot, rate=rate, div=div, expiry=expiry, v0=vol**2, kappa=_t(1.0), theta=vol**2, sigma=_t(1e-6), rho=_t(0.0)
    )
    black = sq.engines.fourier.black_characteristic(spot=spot, rate=rate, div=div, vol=vol, expiry=expiry)(u)
    assert torch.allclose(heston.real, black.real, atol=1e-4)
    assert torch.allclose(heston.imag, black.imag, atol=1e-4)


def test_merton_with_no_jumps_matches_black_scholes():
    kwargs = dict(
        spot=_t(100.0),
        strike=_t(110.0),
        expiry=_t(0.75),
        rate=_t(0.02),
        vol=_t(0.18),
        div=_t(0.0),
        is_call=torch.tensor(True),
    )
    plain = sq.models.black_scholes.price(**kwargs)
    jumped = sq.models.merton.price(
        **kwargs, jump_intensity=_t(0.0), jump_mean=_t(0.0), jump_vol=_t(0.1), num_terms=4
    )
    assert float(jumped) == pytest.approx(float(plain), rel=1e-10, abs=1e-10)


def test_bootstrap_reprices_a_flat_curve():
    times = _t([0.5, 1.0, 1.5, 2.0])
    rate = 0.04
    deposits = times[:1]
    # One deposit, then par swaps on the same grid.
    factors = torch.exp(-_t(rate) * times)
    deltas = _t([0.5, 0.5, 0.5, 0.5])
    swap_rates = []
    for index in range(1, 4):
        level = (deltas[: index + 1] * factors[: index + 1]).sum()
        swap_rates.append((1.0 - factors[index]) / level)
    rates = torch.stack([_t(rate), *swap_rates])
    curve = bootstrap(pay_times=times, rates=rates, kinds=["deposit", "swap", "swap", "swap"], deltas=deltas)
    for index in range(1, 4):
        recovered = par_swap_rate(curve, times[: index + 1], deltas[: index + 1])
        assert float(recovered) == pytest.approx(float(swap_rates[index - 1]), abs=1e-10)
    # Bucketed delta is finite.
    quote = rates.detach().requires_grad_(True)
    built = bootstrap(pay_times=times, rates=quote, kinds=["deposit", "swap", "swap", "swap"], deltas=deltas)
    (grad,) = torch.autograd.grad(built.discount(_t(2.0)), quote)
    assert torch.isfinite(grad).all()


def test_discount_curve_flat_zero():
    times = _t([0.0, 1.0, 2.0])
    curve = DiscountCurve(times, torch.exp(-0.05 * times))
    assert float(curve.discount(_t(1.5))) == pytest.approx(math.exp(-0.05 * 1.5), rel=1e-12)


def test_schedule_and_business252_match_quantlib():
    effective = Date(15, 1, 2025)
    termination = Date(15, 1, 2027)
    ours = Schedule(
        effective,
        termination,
        Period(6, TimeUnit.Months),
        TARGET(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.ModifiedFollowing,
        DateGeneration.Backward,
        False,
    )
    ref = ql.Schedule(
        ql.Date(15, 1, 2025),
        ql.Date(15, 1, 2027),
        ql.Period(6, ql.Months),
        ql.TARGET(),
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )
    assert len(ours) == len(ref)
    for index, date in enumerate(ours.dates):
        assert (date.day, int(date.month), date.year) == (ref[index].dayOfMonth(), ref[index].month(), ref[index].year())
    day_count = ActualActual(ActualActual.Convention.ISMA)
    assert day_count.year_fraction(ours[0], ours[1], schedule=ours) == pytest.approx(
        ql.ActualActual(ql.ActualActual.ISMA, ref).yearFraction(ref[0], ref[1])
    )
    counter = Business252(TARGET())
    ref_counter = ql.Business252(ql.TARGET())
    start, end = Date(1, 1, 2024), Date(1, 7, 2024)
    assert counter.name == ref_counter.name()
    assert counter.day_count(start, end) == ref_counter.dayCount(ql.Date(1, 1, 2024), ql.Date(1, 7, 2024))


def test_newton_implicit_gradient():
    target = _t(2.0).requires_grad_()

    def residual(x, value):
        return x * x - value

    root, converged = newton(residual, _t(1.0), (target,))
    assert bool(converged)
    assert float(root.detach()) == pytest.approx(math.sqrt(2.0), rel=1e-8)
    (grad,) = torch.autograd.grad(root, target)
    assert float(grad) == pytest.approx(0.5 / math.sqrt(2.0), rel=1e-5)


def test_linear_interpolation_and_quadrature():
    knots = _t([0.0, 1.0, 2.0])
    values = _t([0.0, 2.0, 2.0])
    query = _t(0.25).requires_grad_()
    result = linear(query, knots, values)
    assert float(result.detach()) == pytest.approx(0.5)
    (grad,) = torch.autograd.grad(result, query)
    assert float(grad) == pytest.approx(2.0)
    integral = gauss_legendre(lambda x: x**2, _t(0.0), _t(1.0), n=8)
    simpson_integral = simpson(lambda x: x**2, _t(0.0), _t(1.0), n=64)
    assert float(integral) == pytest.approx(1.0 / 3.0, rel=1e-10)
    assert float(simpson_integral) == pytest.approx(1.0 / 3.0, rel=1e-8)


def test_dupire_recovers_flat_volatility():
    spot, rate, div, vol = _t(100.0), _t(0.01), _t(0.0), _t(0.2)

    def calls(strike, expiry):
        return sq.models.black_scholes.price(
            spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=torch.tensor(True)
        )

    local = dupire_local_vol(calls, spot=spot, times=_t(1.0), strikes=_t(100.0), rate=rate, div=div)
    assert float(local.detach()) == pytest.approx(0.2, abs=1e-3)


def test_svi_slice_is_positive():
    moneyness = torch.linspace(-0.3, 0.3, 7, dtype=torch.float64)
    total = svi_total_variance(moneyness, _t(0.04), _t(0.1), _t(-0.3), _t(0.0), _t(0.2))
    assert torch.all(total > 0)
    assert svi_butterfly_ok(moneyness, _t(0.04), _t(0.1), _t(-0.3), _t(0.0), _t(0.2)).dtype == torch.bool


def test_american_put_lattice_and_pde_are_close():
    spot, strike, expiry, rate, vol, div = _t(100.0), _t(100.0), _t(1.0), _t(0.05), _t(0.2), _t(0.0)
    flag = torch.tensor(False)
    market = Market(spot=spot, rate=rate, div=div)
    option = AmericanOption(strike=strike, expiry=expiry, is_call=flag)
    params = BlackScholesParams(vol=vol)
    tree = Lattice(num_steps=300).price(option, params, market)
    pde = PDE(num_space=120, num_time=120).price(option, params, market)
    assert float(tree) == pytest.approx(float(pde), abs=0.08)
    european = sq.models.black_scholes.price(
        spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, div=div, is_call=flag
    )
    assert float(tree) > float(european)


def test_calibration_matches_the_quote_with_unit_sensitivity():
    quotes = _t(0.2).requires_grad_()
    params = BlackScholesParams(vol=_t(0.5))

    def pricer(model):
        return model.vol

    result = calibrate(params, quotes, pricer, bounds={"vol": "positive"}, max_iter=20)
    assert float(result.params.vol.detach()) == pytest.approx(0.2, abs=1e-6)
    (sensitivity,) = torch.autograd.grad(result.params.vol, quotes)
    assert float(sensitivity) == pytest.approx(1.0, abs=1e-5)


def test_pytree_params_round_trip():
    params = SABRParams(alpha=_t(0.2), beta=_t(0.5), rho=_t(-0.2), nu=_t(0.3))
    flat, context = torch.utils._pytree.tree_flatten(params)
    rebuilt = torch.utils._pytree.tree_unflatten(flat, context)
    assert float(rebuilt.alpha) == 0.2
    assert isinstance(rebuilt, SABRParams)
    assert HestonParams(v0=_t(0.04), kappa=_t(1.0), theta=_t(0.04), sigma=_t(0.3), rho=_t(-0.5)).v0.dtype == torch.float64
