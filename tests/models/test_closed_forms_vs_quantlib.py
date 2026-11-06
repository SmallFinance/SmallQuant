"""Closed-form prices against the matching QuantLib analytic engines."""

import pytest
import torch

import QuantLib as ql

from smallquant.models import (
    asian,
    bachelier,
    barrier,
    binary_barrier,
    black76,
    cev,
    chooser,
    cliquet,
    compound,
    digital_american,
    dividend,
    double_barrier,
    double_barrier_binary,
    extensible,
    forward_start,
    lookback,
    margrabe,
    partial_time_barrier,
    quanto,
    soft_barrier,
    stulz,
    two_asset_barrier,
    two_asset_correlation,
)

pytestmark = pytest.mark.quantlib

REL = 1e-6
ABS = 1e-6

EVAL = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()


def _date(days):
    return EVAL + int(days)


def _year(days):
    return days / 365.0


def _flat(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(EVAL, rate, DAY_COUNT, ql.Continuous))


def _vol(value):
    return ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(EVAL, CALENDAR, value, DAY_COUNT)
    )


def _process(spot, rate, div, vol):
    ql.Settings.instance().evaluationDate = EVAL
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        _flat(div),
        _flat(rate),
        _vol(vol),
    )


def _euro(days):
    return ql.EuropeanExercise(_date(days))


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _call_flag(is_call):
    return torch.tensor(is_call)


def _check(ours, ref):
    assert float(ours) == pytest.approx(ref, rel=REL, abs=ABS)


@pytest.mark.parametrize("is_call", [True, False])
def test_black76(is_call):
    forward, strike, days, vol, rate = 100.0, 100.0, 365, 0.2, 0.05
    discount = torch.exp(_t(-rate * _year(days)))
    ours = black76.price(
        forward=_t(forward),
        strike=_t(strike),
        expiry=_t(_year(days)),
        vol=_t(vol),
        discount=discount,
        is_call=_call_flag(is_call),
    )
    ref = ql.blackFormula(
        ql.Option.Call if is_call else ql.Option.Put,
        strike,
        forward,
        vol * _year(days) ** 0.5,
        float(discount),
    )
    _check(ours, ref)


@pytest.mark.parametrize("is_call", [True, False])
def test_bachelier(is_call):
    forward, strike, days, normal_vol, rate = 100.0, 98.0, 365, 15.0, 0.03
    discount = torch.exp(_t(-rate * _year(days)))
    ours = bachelier.price(
        forward=_t(forward),
        strike=_t(strike),
        expiry=_t(_year(days)),
        normal_vol=_t(normal_vol),
        discount=discount,
        is_call=_call_flag(is_call),
    )
    ref = ql.bachelierBlackFormula(
        ql.Option.Call if is_call else ql.Option.Put,
        strike,
        forward,
        normal_vol * _year(days) ** 0.5,
        float(discount),
    )
    _check(ours, ref)


@pytest.mark.parametrize("kind", ["down_in", "up_in", "down_out", "up_out"])
@pytest.mark.parametrize("is_call", [True, False])
def test_barrier(kind, is_call):
    spot, strike, days, rate, div, vol = 100.0, 100.0, 365, 0.08, 0.04, 0.25
    level = 90.0 if kind.startswith("down") else 110.0
    rebate = 3.0
    ql_kind = {
        "down_in": ql.Barrier.DownIn,
        "up_in": ql.Barrier.UpIn,
        "down_out": ql.Barrier.DownOut,
        "up_out": ql.Barrier.UpOut,
    }[kind]
    option = ql.BarrierOption(
        ql_kind,
        level,
        rebate,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        _euro(days),
    )
    option.setPricingEngine(ql.AnalyticBarrierEngine(_process(spot, rate, div, vol)))
    ours = barrier.price(
        spot=_t(spot),
        strike=_t(strike),
        barrier=_t(level),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        rebate=_t(rebate),
        is_call=_call_flag(is_call),
        kind=kind,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("kind", ["down_in", "up_in", "down_out", "up_out"])
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("asset", [False, True])
def test_binary_barrier(kind, is_call, asset):
    spot, strike, days, rate, div, vol, cash = 100.0, 100.0, 365, 0.08, 0.04, 0.25, 10.0
    level = 95.0 if kind.startswith("down") else 105.0
    ql_kind = {
        "down_in": ql.Barrier.DownIn,
        "up_in": ql.Barrier.UpIn,
        "down_out": ql.Barrier.DownOut,
        "up_out": ql.Barrier.UpOut,
    }[kind]
    option_type = ql.Option.Call if is_call else ql.Option.Put
    payoff = (
        ql.AssetOrNothingPayoff(option_type, strike)
        if asset
        else ql.CashOrNothingPayoff(option_type, strike, cash)
    )
    exercise = ql.AmericanExercise(EVAL, _date(days), True)
    option = ql.BarrierOption(ql_kind, level, 0.0, payoff, exercise)
    option.setPricingEngine(ql.AnalyticBinaryBarrierEngine(_process(spot, rate, div, vol)))
    ours = binary_barrier.price(
        spot=_t(spot),
        strike=_t(strike),
        barrier=_t(level),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        cash=_t(cash),
        is_call=_call_flag(is_call),
        kind=kind,
        asset_or_nothing=asset,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("kind", ["knock_in", "knock_out"])
@pytest.mark.parametrize("is_call", [True, False])
def test_double_barrier(kind, is_call):
    spot, strike, days, rate, div, vol = 100.0, 100.0, 365, 0.08, 0.04, 0.25
    ql_kind = ql.DoubleBarrier.KnockIn if kind == "knock_in" else ql.DoubleBarrier.KnockOut
    option = ql.DoubleBarrierOption(
        ql_kind,
        80.0,
        120.0,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        _euro(days),
    )
    option.setPricingEngine(ql.AnalyticDoubleBarrierEngine(_process(spot, rate, div, vol)))
    ours = double_barrier.price(
        spot=_t(spot),
        strike=_t(strike),
        barrier_lo=_t(80.0),
        barrier_hi=_t(120.0),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(is_call),
        kind=kind,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("kind", ["knock_in", "knock_out", "kiko", "koki"])
def test_double_barrier_binary(kind):
    spot, days, rate, div, vol, cash = 100.0, 365, 0.08, 0.04, 0.2, 10.0
    ql_kind = {
        "knock_in": ql.DoubleBarrier.KnockIn,
        "knock_out": ql.DoubleBarrier.KnockOut,
        "kiko": ql.DoubleBarrier.KIKO,
        "koki": ql.DoubleBarrier.KOKI,
    }[kind]
    exercise = _euro(days) if kind in ("knock_in", "knock_out") else ql.AmericanExercise(EVAL, _date(days))
    option = ql.DoubleBarrierOption(
        ql_kind, 80.0, 120.0, 0.0, ql.CashOrNothingPayoff(ql.Option.Call, 100.0, cash), exercise
    )
    option.setPricingEngine(ql.AnalyticDoubleBarrierBinaryEngine(_process(spot, rate, div, vol)))
    ours = double_barrier_binary.price(
        spot=_t(spot),
        strike=_t(100.0),
        barrier_lo=_t(80.0),
        barrier_hi=_t(120.0),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        cash=_t(cash),
        kind=kind,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("kind", ["down_out", "up_out", "down_in", "up_in"])
def test_soft_barrier(kind):
    spot, strike, days, rate, div, vol = 100.0, 100.0, 365, 0.08, 0.04, 0.25
    lo, hi = (90.0, 95.0) if kind.startswith("down") else (105.0, 115.0)
    ql_kind = {
        "down_in": ql.Barrier.DownIn,
        "up_in": ql.Barrier.UpIn,
        "down_out": ql.Barrier.DownOut,
        "up_out": ql.Barrier.UpOut,
    }[kind]
    option = ql.SoftBarrierOption(
        ql_kind, lo, hi, ql.PlainVanillaPayoff(ql.Option.Call, strike), _euro(days)
    )
    option.setPricingEngine(ql.AnalyticSoftBarrierEngine(_process(spot, rate, div, vol)))
    ours = soft_barrier.price(
        spot=_t(spot),
        strike=_t(strike),
        barrier_lo=_t(lo),
        barrier_hi=_t(hi),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
        kind=kind,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("kind,barrier_range", [("down_out", "start"), ("up_out", "end_b1"), ("down_in", "start")])
def test_partial_time_barrier(kind, barrier_range):
    spot, strike, days, cover_days, rate, div, vol = 100.0, 90.0, 365, 182, 0.08, 0.04, 0.25
    level = 95.0
    ql_kind = {
        "down_out": ql.Barrier.DownOut,
        "up_out": ql.Barrier.UpOut,
        "down_in": ql.Barrier.DownIn,
    }[kind]
    ql_range = {"start": ql.PartialBarrier.Start, "end_b1": ql.PartialBarrier.EndB1}[barrier_range]
    option = ql.PartialTimeBarrierOption(
        ql_kind,
        ql_range,
        level,
        0.0,
        _date(cover_days),
        ql.PlainVanillaPayoff(ql.Option.Call, strike),
        _euro(days),
    )
    option.setPricingEngine(ql.AnalyticPartialTimeBarrierOptionEngine(_process(spot, rate, div, vol)))
    ours = partial_time_barrier.price(
        spot=_t(spot),
        strike=_t(strike),
        barrier=_t(level),
        expiry=_t(_year(days)),
        cover=_t(_year(cover_days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
        kind=kind,
        barrier_range=barrier_range,
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("is_call", [True, False])
def test_continuous_geometric_asian(is_call):
    spot, strike, days, rate, div, vol = 100.0, 100.0, 365, 0.05, 0.02, 0.2
    option = ql.ContinuousAveragingAsianOption(
        ql.Average.Geometric,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        _euro(days),
    )
    option.setPricingEngine(
        ql.AnalyticContinuousGeometricAveragePriceAsianEngine(_process(spot, rate, div, vol))
    )
    ours = asian.continuous_price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(is_call),
    )
    _check(ours, option.NPV())


def test_discrete_geometric_asian():
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.02, 0.2
    days = list(range(30, 361, 30))
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric,
        1.0,
        0,
        [_date(day) for day in days],
        ql.PlainVanillaPayoff(ql.Option.Call, strike),
        _euro(days[-1]),
    )
    option.setPricingEngine(
        ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(_process(spot, rate, div, vol))
    )
    ours = asian.discrete_price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(days[-1])),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        fixing_times=_t([_year(day) for day in days]),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_discrete_geometric_strike_asian():
    spot, rate, div, vol = 100.0, 0.05, 0.02, 0.2
    days = list(range(0, 361, 30))
    option = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric,
        1.0,
        0,
        [_date(day) for day in days],
        ql.PlainVanillaPayoff(ql.Option.Call, 0.0),
        _euro(365),
    )
    option.setPricingEngine(
        ql.AnalyticDiscreteGeometricAverageStrikeAsianEngine(_process(spot, rate, div, vol))
    )
    ours = asian.discrete_strike_price(
        spot=_t(spot),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        fixing_times=_t([_year(day) for day in days]),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("is_call", [True, False])
def test_fixed_lookback(is_call):
    spot, strike, extremum, days, rate, div, vol = 100.0, 100.0, 95.0 if is_call else 105.0, 365, 0.08, 0.04, 0.2
    option = ql.ContinuousFixedLookbackOption(
        extremum,
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        _euro(days),
    )
    option.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(_process(spot, rate, div, vol)))
    ours = lookback.fixed_price(
        spot=_t(spot),
        strike=_t(strike),
        minimum=_t(extremum),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(is_call),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("is_call", [True, False])
def test_floating_lookback(is_call):
    spot, extremum, days, rate, div, vol = 100.0, 90.0 if is_call else 110.0, 365, 0.08, 0.04, 0.2
    option = ql.ContinuousFloatingLookbackOption(
        extremum,
        ql.FloatingTypePayoff(ql.Option.Call if is_call else ql.Option.Put),
        _euro(days),
    )
    option.setPricingEngine(
        ql.AnalyticContinuousFloatingLookbackEngine(_process(spot, rate, div, vol))
    )
    ours = lookback.floating_price(
        spot=_t(spot),
        extremum=_t(extremum),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(is_call),
    )
    _check(ours, option.NPV())


def test_partial_fixed_lookback():
    spot, strike, days, start, rate, div, vol = 100.0, 100.0, 365, 182, 0.08, 0.04, 0.2
    option = ql.ContinuousPartialFixedLookbackOption(
        _date(start), ql.PlainVanillaPayoff(ql.Option.Call, strike), _euro(days)
    )
    option.setPricingEngine(
        ql.AnalyticContinuousPartialFixedLookbackEngine(_process(spot, rate, div, vol))
    )
    ours = lookback.partial_fixed_price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(days)),
        lookback_start=_t(_year(start)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_partial_floating_lookback():
    spot, extremum, days, end, rate, div, vol = 100.0, 95.0, 365, 182, 0.08, 0.04, 0.2
    option = ql.ContinuousPartialFloatingLookbackOption(
        extremum,
        1.0,
        _date(end),
        ql.FloatingTypePayoff(ql.Option.Call),
        _euro(days),
    )
    option.setPricingEngine(
        ql.AnalyticContinuousPartialFloatingLookbackEngine(_process(spot, rate, div, vol))
    )
    ours = lookback.partial_floating_price(
        spot=_t(spot),
        extremum=_t(extremum),
        level=_t(1.0),
        expiry=_t(_year(days)),
        lookback_end=_t(_year(end)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("at_expiry", [True, False])
@pytest.mark.parametrize("is_call", [True, False])
def test_digital_american(at_expiry, is_call):
    spot, strike, days, rate, div, vol, cash = 100.0, 105.0 if is_call else 95.0, 365, 0.08, 0.04, 0.2, 10.0
    exercise = ql.AmericanExercise(EVAL, _date(days), at_expiry)
    option = ql.VanillaOption(ql.CashOrNothingPayoff(ql.Option.Call if is_call else ql.Option.Put, strike, cash), exercise)
    option.setPricingEngine(ql.AnalyticDigitalAmericanEngine(_process(spot, rate, div, vol)))
    ours = digital_american.price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        cash=_t(cash),
        is_call=_call_flag(is_call),
        payoff_at_expiry=at_expiry,
    )
    _check(ours, option.NPV())


def test_compound():
    spot, rate, div, vol = 100.0, 0.05, 0.02, 0.2
    mother, daughter = 10.0, 100.0
    option = ql.CompoundOption(
        ql.PlainVanillaPayoff(ql.Option.Call, mother),
        _euro(182),
        ql.PlainVanillaPayoff(ql.Option.Call, daughter),
        _euro(365),
    )
    option.setPricingEngine(ql.AnalyticCompoundOptionEngine(_process(spot, rate, div, vol)))
    ours = compound.price(
        spot=_t(spot),
        strike=_t(mother),
        underlying_strike=_t(daughter),
        expiry=_t(_year(182)),
        underlying_expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
        underlying_is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_simple_chooser():
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.02, 0.2
    option = ql.SimpleChooserOption(_date(182), strike, _euro(365))
    option.setPricingEngine(ql.AnalyticSimpleChooserEngine(_process(spot, rate, div, vol)))
    ours = chooser.simple_price(
        spot=_t(spot),
        strike=_t(strike),
        choose=_t(_year(182)),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
    )
    _check(ours, option.NPV())


def test_complex_chooser():
    spot, rate, div, vol = 100.0, 0.08, 0.04, 0.25
    option = ql.ComplexChooserOption(_date(90), 100.0, 100.0, _euro(365), _euro(365))
    option.setPricingEngine(ql.AnalyticComplexChooserEngine(_process(spot, rate, div, vol)))
    ours = chooser.complex_price(
        spot=_t(spot),
        strike_call=_t(100.0),
        strike_put=_t(100.0),
        choose=_t(_year(90)),
        call_expiry=_t(_year(365)),
        put_expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
    )
    _check(ours, option.NPV())


def test_forward_start():
    spot, rate, div, vol = 100.0, 0.05, 0.02, 0.2
    option = ql.ForwardVanillaOption(
        1.1, _date(182), ql.PlainVanillaPayoff(ql.Option.Call, 0.0), _euro(365)
    )
    option.setPricingEngine(ql.ForwardEuropeanEngine(_process(spot, rate, div, vol)))
    ours = forward_start.price(
        spot=_t(spot),
        moneyness=_t(1.1),
        reset=_t(_year(182)),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_cliquet():
    spot, rate, div, vol = 100.0, 0.05, 0.02, 0.2
    resets = [91, 182, 273]
    option = ql.CliquetOption(
        ql.PercentageStrikePayoff(ql.Option.Call, 1.0),
        _euro(365),
        [_date(day) for day in resets],
    )
    option.setPricingEngine(ql.AnalyticCliquetEngine(_process(spot, rate, div, vol)))
    ours = cliquet.price(
        spot=_t(spot),
        moneyness=_t(1.0),
        reset_times=_t([_year(day) for day in resets]),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_performance_option():
    spot, rate, div, vol = 100.0, 0.05, 0.02, 0.2
    resets = [91, 182, 273]
    option = ql.CliquetOption(
        ql.PercentageStrikePayoff(ql.Option.Call, 1.0),
        _euro(365),
        [_date(day) for day in resets],
    )
    option.setPricingEngine(ql.AnalyticPerformanceEngine(_process(spot, rate, div, vol)))
    ours = cliquet.performance_price(
        spot=_t(spot),
        moneyness=_t(1.0),
        reset_times=_t([_year(day) for day in resets]),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_margrabe():
    spot1, spot2, rate, div1, div2, vol1, vol2, rho = 100.0, 90.0, 0.05, 0.02, 0.03, 0.2, 0.25, 0.4
    option = ql.MargrabeOption(1, 1, _euro(365))
    option.setPricingEngine(
        ql.AnalyticEuropeanMargrabeEngine(
            _process(spot1, rate, div1, vol1), _process(spot2, rate, div2, vol2), rho
        )
    )
    ours = margrabe.price(
        spot1=_t(spot1),
        spot2=_t(spot2),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol1=_t(vol1),
        vol2=_t(vol2),
        div1=_t(div1),
        div2=_t(div2),
        correlation=_t(rho),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("on_max", [False, True])
def test_stulz(on_max):
    spot1, spot2, strike, rate, div, vol1, vol2, rho = 100.0, 100.0, 100.0, 0.05, 0.0, 0.2, 0.3, 0.5
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    basket = ql.MaxBasketPayoff(payoff) if on_max else ql.MinBasketPayoff(payoff)
    option = ql.BasketOption(basket, _euro(365))
    option.setPricingEngine(
        ql.StulzEngine(_process(spot1, rate, div, vol1), _process(spot2, rate, div, vol2), rho)
    )
    ours = stulz.price(
        spot1=_t(spot1),
        spot2=_t(spot2),
        strike=_t(strike),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol1=_t(vol1),
        vol2=_t(vol2),
        div1=_t(div),
        div2=_t(div),
        correlation=_t(rho),
        is_call=_call_flag(True),
        on_max=on_max,
    )
    _check(ours, option.NPV())


def test_two_asset_correlation():
    spot1, spot2, rate, div, vol1, vol2, rho = 100.0, 100.0, 0.05, 0.02, 0.2, 0.3, 0.5
    option = ql.TwoAssetCorrelationOption(ql.Option.Call, 100.0, 100.0, _euro(365))
    option.setPricingEngine(
        ql.AnalyticTwoAssetCorrelationEngine(
            _process(spot1, rate, div, vol1),
            _process(spot2, rate, div, vol2),
            ql.QuoteHandle(ql.SimpleQuote(rho)),
        )
    )
    ours = two_asset_correlation.price(
        spot1=_t(spot1),
        spot2=_t(spot2),
        strike1=_t(100.0),
        strike2=_t(100.0),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol1=_t(vol1),
        vol2=_t(vol2),
        div1=_t(div),
        div2=_t(div),
        correlation=_t(rho),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_two_asset_barrier():
    spot1, spot2, rate, div, vol1, vol2, rho = 100.0, 100.0, 0.08, 0.04, 0.2, 0.2, 0.5
    option = ql.TwoAssetBarrierOption(
        ql.Barrier.DownOut, 95.0, ql.PlainVanillaPayoff(ql.Option.Call, 100.0), _euro(365)
    )
    option.setPricingEngine(
        ql.AnalyticTwoAssetBarrierEngine(
            _process(spot1, rate, div, vol1),
            _process(spot2, rate, div, vol2),
            ql.QuoteHandle(ql.SimpleQuote(rho)),
        )
    )
    ours = two_asset_barrier.price(
        spot1=_t(spot1),
        spot2=_t(spot2),
        strike=_t(100.0),
        barrier=_t(95.0),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol1=_t(vol1),
        vol2=_t(vol2),
        div1=_t(div),
        div2=_t(div),
        correlation=_t(rho),
        is_call=_call_flag(True),
        kind="down_out",
    )
    _check(ours, option.NPV())


def test_quanto_european():
    spot, strike, rate, foreign, div, vol, fx_vol, rho = 100.0, 100.0, 0.05, 0.03, 0.01, 0.2, 0.1, 0.3
    option = ql.QuantoVanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike), _euro(365))
    option.setPricingEngine(
        ql.QuantoEuropeanEngine(
            _process(spot, rate, div, vol),
            _flat(foreign),
            _vol(fx_vol),
            ql.QuoteHandle(ql.SimpleQuote(rho)),
        )
    )
    ours = quanto.price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        foreign_rate=_t(foreign),
        fx_vol=_t(fx_vol),
        correlation=_t(rho),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_dividend_european():
    spot, strike, rate, div, vol = 100.0, 100.0, 0.05, 0.0, 0.2
    dividends = ql.DividendSchedule()
    dividends.push_back(ql.FixedDividend(2.0, _date(90)))
    dividends.push_back(ql.FixedDividend(2.0, _date(180)))
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike), _euro(365))
    option.setPricingEngine(
        ql.AnalyticDividendEuropeanEngine(_process(spot, rate, div, vol), dividends)
    )
    ours = dividend.price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        dividend_times=_t([_year(90), _year(180)]),
        dividend_amounts=_t([2.0, 2.0]),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize("is_call", [True, False])
def test_cev(is_call):
    forward, strike, alpha, beta, rate, days = 100.0, 100.0, 2.0, 0.5, 0.03, 365
    engine_option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike), _euro(days)
    )
    engine_option.setPricingEngine(ql.AnalyticCEVEngine(forward, alpha, beta, _flat(rate)))
    discount = torch.exp(_t(-rate * _year(days)))
    ours = cev.price(
        forward=_t(forward),
        strike=_t(strike),
        expiry=_t(_year(days)),
        alpha=_t(alpha),
        beta=_t(beta),
        discount=discount,
        is_call=_call_flag(is_call),
    )
    _check(ours, engine_option.NPV())


def test_writer_extensible():
    spot, rate, div, vol = 100.0, 0.08, 0.04, 0.25
    option = ql.WriterExtensibleOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        _euro(180),
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        _euro(365),
    )
    option.setPricingEngine(ql.AnalyticWriterExtensibleOptionEngine(_process(spot, rate, div, vol)))
    ours = extensible.writer_price(
        spot=_t(spot),
        strike=_t(100.0),
        extended_strike=_t(100.0),
        expiry=_t(_year(180)),
        extended_expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


@pytest.mark.parametrize(
    "is_call,premium",
    [(True, 2.0), (True, 8.0), (False, 2.0)],
)
def test_holder_extensible(is_call, premium):
    spot, rate, div, vol = 100.0, 0.08, 0.04, 0.25
    kind = ql.Option.Call if is_call else ql.Option.Put
    option = ql.HolderExtensibleOption(
        kind,
        premium,
        _date(365),
        100.0,
        ql.PlainVanillaPayoff(kind, 100.0),
        _euro(180),
    )
    option.setPricingEngine(ql.AnalyticHolderExtensibleOptionEngine(_process(spot, rate, div, vol)))
    ours = extensible.holder_price(
        spot=_t(spot),
        strike=_t(100.0),
        extended_strike=_t(100.0),
        premium=_t(premium),
        expiry=_t(_year(180)),
        extended_expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        is_call=_call_flag(is_call),
    )
    _check(ours, option.NPV())


def test_quanto_forward():
    spot, rate, foreign, div, vol, fx_vol, rho = 100.0, 0.05, 0.03, 0.01, 0.2, 0.1, 0.3
    option = ql.ForwardVanillaOption(
        1.1, _date(182), ql.PlainVanillaPayoff(ql.Option.Call, 0.0), _euro(365)
    )
    option.setPricingEngine(
        ql.QuantoForwardEuropeanEngine(
            _process(spot, rate, div, vol),
            _flat(foreign),
            _vol(fx_vol),
            ql.QuoteHandle(ql.SimpleQuote(rho)),
        )
    )
    ours = quanto.forward_start_price(
        spot=_t(spot),
        moneyness=_t(1.1),
        reset=_t(_year(182)),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        foreign_rate=_t(foreign),
        fx_vol=_t(fx_vol),
        correlation=_t(rho),
        is_call=_call_flag(True),
    )
    _check(ours, option.NPV())


def test_quanto_barrier():
    spot, rate, foreign, div, vol, fx_vol, rho = 100.0, 0.05, 0.03, 0.01, 0.2, 0.1, 0.3
    option = ql.BarrierOption(
        ql.Barrier.DownOut,
        90.0,
        0.0,
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
        _euro(365),
    )
    option.setPricingEngine(
        ql.QuantoBarrierEngine(
            _process(spot, rate, div, vol),
            _flat(foreign),
            _vol(fx_vol),
            ql.QuoteHandle(ql.SimpleQuote(rho)),
        )
    )
    ours = quanto.barrier_option_price(
        spot=_t(spot),
        strike=_t(100.0),
        barrier=_t(90.0),
        expiry=_t(_year(365)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        rebate=_t(0.0),
        foreign_rate=_t(foreign),
        fx_vol=_t(fx_vol),
        correlation=_t(rho),
        is_call=_call_flag(True),
        kind="down_out",
    )
    _check(ours, option.NPV())


def test_digital_american_knock_out():
    spot, strike, days, rate, div, vol, cash = 100.0, 105.0, 365, 0.08, 0.04, 0.2, 10.0
    exercise = ql.AmericanExercise(EVAL, _date(days), True)
    option = ql.VanillaOption(ql.CashOrNothingPayoff(ql.Option.Call, strike, cash), exercise)
    option.setPricingEngine(ql.AnalyticDigitalAmericanKOEngine(_process(spot, rate, div, vol)))
    ours = digital_american.price(
        spot=_t(spot),
        strike=_t(strike),
        expiry=_t(_year(days)),
        rate=_t(rate),
        vol=_t(vol),
        div=_t(div),
        cash=_t(cash),
        is_call=_call_flag(True),
        payoff_at_expiry=True,
        knock_in=False,
    )
    _check(ours, option.NPV())
