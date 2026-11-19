"""American closed-form approximations against QuantLib and a CRR tree.

* Barone-Adesi-Whaley against ``ql.BaroneAdesiWhaleyApproximationEngine``.
* Bjerksund-Stensland 1993 against ``ql.BjerksundStenslandApproximationEngine``,
  which implements the 1993 single-boundary formula (it matches ours to about 1e-11).
* Bjerksund-Stensland 2002 against a 2,000-step CRR tree, since no published table
  could be sourced reliably (see ``test_bjerksund_stensland_2002_vs_crr``).
"""

import itertools

import pytest
import QuantLib as ql
import torch

from smallquant.black_scholes.approximations import adesi_whaley, bjerksund_stensland
from smallquant.black_scholes.crr_binomial_tree import option_price_binomial
from smallquant.black_scholes.vanilla_prices import option_price

pytestmark = pytest.mark.quantlib

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()
STRIKE = 100.0


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


def _quantlib(engine, spot, days, rate, div, vol, is_call):
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, STRIKE),
        ql.AmericanExercise(EVAL, EVAL + int(days)),
    )
    option.setPricingEngine(engine(_process(spot, rate, div, vol)))
    return option.NPV()


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _batch(rows):
    """Columns of a list of (is_call, spot, vol, rate, div, days) rows as tensors."""
    is_call, spot, vol, rate, div, days = zip(*rows)
    return dict(
        spot=_t(list(spot)),
        strike=_t(STRIKE),
        expiry=_t(list(days)) / 365.0,
        rate=_t(list(rate)),
        vol=_t(list(vol)),
        div=_t(list(div)),
        is_call=torch.tensor(list(is_call)),
    )


SPOTS = (80.0, 95.0, 100.0, 105.0, 120.0)
VOLS = (0.1, 0.25, 0.4)


# Barone-Adesi-Whaley ---------------------------------------------------------------

# QuantLib stops its critical-price Newton loop once |LHS - RHS| / K <= 1e-6, so its
# critical price, and with it the price, carries an error up to about 1e-6 * K. A pure
# Python copy of QuantLib's loop with that stopping rule reproduces QuantLib to 5e-12,
# and the same copy run to full convergence reproduces ours to 3e-12; the largest gap on
# this grid is 2.8e-5 on a price of 5.4. So the tolerance is 1e-8 relative plus
# 1e-6 * K absolute, the size of QuantLib's own stopping error, instead of 1e-8 alone.
BAW_REL = 1e-8
BAW_ABS = 1e-6 * STRIKE

BAW_GRID = [
    (is_call, spot, vol, rate, div, days)
    for is_call, spot, vol, rate, div, days in itertools.product(
        (True, False), SPOTS, VOLS, (0.01, 0.05, 0.1), (0.0, 0.03, 0.08, 0.12), (30, 182, 730)
    )
]


def test_adesi_whaley_vs_quantlib():
    ours = adesi_whaley(**_batch(BAW_GRID))
    for value, row in zip(ours.tolist(), BAW_GRID):
        is_call, spot, vol, rate, div, days = row
        ref = _quantlib(ql.BaroneAdesiWhaleyApproximationEngine, spot, days, rate, div, vol, is_call)
        assert value == pytest.approx(ref, rel=BAW_REL, abs=BAW_ABS), row


def test_adesi_whaley_newton_converged():
    """Twenty Newton steps already sit on the root: more steps change nothing."""
    inputs = _batch(BAW_GRID)
    assert torch.allclose(
        adesi_whaley(**inputs, max_iter=20), adesi_whaley(**inputs, max_iter=60), rtol=1e-13, atol=1e-13
    )


def test_adesi_whaley_no_early_exercise_is_european():
    """Call with div <= 0 and put with rate <= 0 return the European price.

    QuantLib's engine only special-cases the call, so for a put at rate 0 it adds a
    spurious premium; ours returns the European price there by design.
    """
    rows = [(True, 100.0, 0.3, 0.05, 0.0, 365), (True, 90.0, 0.2, 0.02, -0.01, 182),
            (False, 100.0, 0.3, 0.0, 0.04, 365), (False, 110.0, 0.2, -0.01, 0.0, 182)]
    inputs = _batch(rows)
    torch.testing.assert_close(adesi_whaley(**inputs), option_price(**inputs), rtol=0.0, atol=0.0)


# Bjerksund-Stensland 1993 ----------------------------------------------------------

BS_REL = 1e-8
BS_ABS = 1e-10

BS_GRID = [
    (is_call, spot, vol, rate, div, days)
    for is_call, spot, vol, rate, div, days in itertools.product(
        (True, False), (60.0,) + SPOTS + (150.0,), VOLS, (0.0, 0.01, 0.05, 0.1),
        (0.0, 0.03, 0.08, 0.12), (30, 182, 730)
    )
]


def test_bjerksund_stensland_1993_vs_quantlib():
    ours = bjerksund_stensland(**_batch(BS_GRID), version=1993)
    for value, row in zip(ours.tolist(), BS_GRID):
        is_call, spot, vol, rate, div, days = row
        ref = _quantlib(ql.BjerksundStenslandApproximationEngine, spot, days, rate, div, vol, is_call)
        assert value == pytest.approx(ref, rel=BS_REL, abs=BS_ABS), row


def test_quantlib_bjerksund_stensland_is_1993():
    """QuantLib's engine is the 1993 formula, not 2002: it matches one and not the other."""
    row = (True, 110.0, 0.2, 0.08, 0.12, 365)
    is_call, spot, vol, rate, div, days = row
    ref = _quantlib(ql.BjerksundStenslandApproximationEngine, spot, days, rate, div, vol, is_call)
    inputs = _batch([row])
    assert float(bjerksund_stensland(**inputs, version=1993)[0]) == pytest.approx(ref, rel=1e-10)
    assert abs(float(bjerksund_stensland(**inputs, version=2002)[0]) - ref) > 1e-3


# Bjerksund-Stensland 2002 ----------------------------------------------------------

# No reliable table of the 2002 paper's own numbers was available, so the reference is
# a 2,000-step CRR tree. The tree is within about 1e-3 of the American price here; the
# rest of the tolerance is the approximation's own error, which reaches 0.6% (0.053 on
# 8.90) and 1.6% (on 0.40) on a wider grid. The check is |BS2002 - CRR| <= 2% * CRR +
# 0.01, plus the lower-bound property: BS2002 <= CRR + 0.01 and BS2002 >= BS1993.
CRR_GRID = [
    (is_call, spot, vol, rate, div, days)
    for is_call, spot, vol, rate, div, days in itertools.product(
        (True, False), (85.0, 100.0, 115.0), (0.2, 0.3), (0.03, 0.08), (0.0, 0.05), (91, 365)
    )
]


def test_bjerksund_stensland_2002_vs_crr():
    inputs = _batch(CRR_GRID)
    ours = bjerksund_stensland(**inputs, version=2002)
    old = bjerksund_stensland(**inputs, version=1993)
    tree = torch.stack(
        [
            option_price_binomial(
                spot=_t(spot), strike=_t(STRIKE), expiry=_t(days / 365.0), rate=_t(rate),
                vol=_t(vol), div=_t(div), is_call=torch.tensor(is_call),
                exercise="american", num_steps=2000,
            )
            for is_call, spot, vol, rate, div, days in CRR_GRID
        ]
    )
    assert torch.all((ours - tree).abs() <= 0.02 * tree + 0.01)
    assert torch.all(ours <= tree + 0.01)
    assert torch.all(ours >= old - 1e-12)
    # 2002 is the better approximation on average.
    assert (ours - tree).abs().mean() < (old - tree).abs().mean()


# Shared properties -----------------------------------------------------------------

PRICERS = [
    pytest.param(adesi_whaley, id="baw"),
    pytest.param(lambda **kw: bjerksund_stensland(**kw, version=1993), id="bs1993"),
    pytest.param(lambda **kw: bjerksund_stensland(**kw, version=2002), id="bs2002"),
]

PROPERTY_GRID = [
    (is_call, spot, vol, rate, div, days)
    for is_call, spot, vol, rate, div, days in itertools.product(
        (True, False), SPOTS, (0.15, 0.35), (0.0, 0.04, 0.09), (0.0, 0.03, 0.1), (45, 365, 1095)
    )
]


@pytest.mark.parametrize("pricer", PRICERS)
def test_batched_equals_single_calls(pricer):
    inputs = _batch(PROPERTY_GRID)
    batched = pricer(**inputs)
    singles = torch.stack(
        [
            pricer(
                spot=_t(spot), strike=_t(STRIKE), expiry=_t(days / 365.0), rate=_t(rate),
                vol=_t(vol), div=_t(div), is_call=is_call,
            )
            for is_call, spot, vol, rate, div, days in PROPERTY_GRID
        ]
    )
    torch.testing.assert_close(batched, singles, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pricer", PRICERS)
def test_broadcasts_over_inputs(pricer):
    spot = _t([[90.0], [100.0], [110.0]])
    vol = _t([0.2, 0.3])
    out = pricer(spot=spot, strike=_t(100.0), expiry=_t(0.5), rate=_t(0.05), vol=vol,
                 div=_t(0.03), is_call=torch.tensor([True, False]))
    assert out.shape == (3, 2)
    assert out.dtype == torch.float64


@pytest.mark.parametrize("pricer", PRICERS)
def test_at_least_european_and_intrinsic(pricer):
    inputs = _batch(PROPERTY_GRID)
    american = pricer(**inputs)
    european = option_price(**inputs)
    intrinsic = torch.where(inputs["is_call"], inputs["spot"] - STRIKE, STRIKE - inputs["spot"]).clamp(min=0)
    assert torch.all(american >= european - 1e-12)
    assert torch.all(american >= intrinsic - 1e-12)


@pytest.mark.parametrize("pricer", PRICERS)
def test_zero_expiry_is_intrinsic(pricer):
    spot = _t([80.0, 100.0, 120.0])
    for is_call in (True, False):
        out = pricer(spot=spot, strike=_t(100.0), expiry=_t(0.0), rate=_t(0.05), vol=_t(0.2),
                     div=_t(0.03), is_call=is_call)
        expected = (spot - 100.0 if is_call else 100.0 - spot).clamp(min=0)
        torch.testing.assert_close(out, expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("pricer", PRICERS)
@pytest.mark.parametrize(
    "is_call,spot,div",
    [(True, 105.0, 0.08), (False, 95.0, 0.02), (True, 90.0, 0.0), (False, 110.0, 0.05)],
)
def test_gradcheck(pricer, is_call, spot, div):
    def price(s, v, r):
        return pricer(spot=s, strike=_t(100.0), expiry=_t(0.75), rate=r, vol=v, div=_t(div), is_call=is_call)

    inputs = (
        _t([spot, spot - 7.0]).requires_grad_(),
        _t([0.25, 0.3]).requires_grad_(),
        _t([0.06, 0.04]).requires_grad_(),
    )
    assert torch.autograd.gradcheck(price, inputs)


@pytest.mark.parametrize("pricer", PRICERS)
def test_gradients_finite_on_masked_branches(pricer):
    """Elements that skip the approximation (no early exercise, zero expiry) keep finite gradients."""
    spot = _t([100.0, 100.0, 100.0, 100.0]).requires_grad_()
    vol = _t([0.2, 0.2, 0.2, 0.2]).requires_grad_()
    rate = _t([0.05, 0.0, 0.05, 0.05]).requires_grad_()
    div = _t([0.0, 0.03, 0.03, 0.03]).requires_grad_()
    expiry = _t([1.0, 1.0, 0.0, 1.0]).requires_grad_()
    out = pricer(spot=spot, strike=_t(100.0), expiry=expiry, rate=rate, vol=vol, div=div,
                 is_call=torch.tensor([True, False, True, False]))
    out.sum().backward()
    for tensor in (spot, vol, rate, div, expiry):
        assert torch.isfinite(tensor.grad).all()


def test_adesi_whaley_float32():
    out = adesi_whaley(spot=torch.tensor([90.0, 110.0]), strike=torch.tensor(100.0), expiry=torch.tensor(0.5),
                       rate=torch.tensor(0.05), vol=torch.tensor(0.25), div=torch.tensor(0.03), is_call=False)
    assert out.dtype == torch.float32
    ref = adesi_whaley(spot=_t([90.0, 110.0]), strike=_t(100.0), expiry=_t(0.5), rate=_t(0.05),
                       vol=_t(0.25), div=_t(0.03), is_call=False)
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


def test_rejects_python_floats_and_bad_version():
    with pytest.raises(TypeError):
        adesi_whaley(spot=100.0, strike=_t(100.0), expiry=_t(1.0), rate=_t(0.05), vol=_t(0.2),
                     div=_t(0.03), is_call=True)
    with pytest.raises(ValueError):
        bjerksund_stensland(spot=_t(100.0), strike=_t(100.0), expiry=_t(1.0), rate=_t(0.05),
                            vol=_t(0.2), div=_t(0.03), is_call=True, version=2000)
