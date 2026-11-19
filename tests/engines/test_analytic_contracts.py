"""Analytic engine prices for contracts that already have closed forms."""

import torch

from smallquant.black_scholes import asian_prices, barrier_price, binary_price
from smallquant.engines.pricing import Analytic
from smallquant.exotics import (
    chooser_prices,
    cliquet_prices,
    compound_prices,
    double_barrier_prices,
    forward_start_prices,
    lookback_prices,
)
from smallquant.instruments import (
    AsianOption,
    BarrierOption,
    ChooserOption,
    CliquetOption,
    CompoundOption,
    DigitalOption,
    DoubleBarrierOption,
    ForwardStartOption,
    LookbackOption,
)
from smallquant.market import Market
from smallquant.models.params import BlackScholesParams
from smallquant.price import price


def _t(value):
    return torch.tensor(value, dtype=torch.float64)


def _market():
    return Market(spot=_t(100.0), rate=_t(0.05), div=_t(0.02))


def _params():
    return BlackScholesParams(vol=_t(0.2))


def _same(actual, expected):
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_digital_scales_the_unit_cash():
    instrument = DigitalOption(
        strike=_t(100.0), expiry=_t(1.0), is_call=torch.tensor(True), cash=_t(15.0)
    )
    market, params = _market(), _params()
    expected = 15.0 * binary_price(
        spot=market.spot,
        strike=instrument.strike,
        expiry=instrument.expiry,
        rate=market.rate,
        vol=params.vol,
        div=market.div,
        is_call=instrument.is_call,
    )
    _same(price(instrument, params, market), expected)


def test_continuous_barrier():
    instrument = BarrierOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        barrier=_t(90.0),
        is_call=torch.tensor(False),
        kind="down_and_out",
        rebate=_t(3.0),
        monitoring="continuous",
    )
    market, params = _market(), _params()
    expected = barrier_price(
        spot=market.spot,
        strike=instrument.strike,
        barrier=instrument.barrier,
        expiry=instrument.expiry,
        rate=market.rate,
        vol=params.vol,
        div=market.div,
        rebate=instrument.rebate,
        is_call=instrument.is_call,
        kind="down_out",
    )
    _same(price(instrument, params, market), expected)


def test_discrete_barrier_is_not_priced_as_continuous():
    instrument = BarrierOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        barrier=_t(90.0),
        is_call=torch.tensor(True),
        kind="down_out",
    )
    try:
        price(instrument, _params(), _market())
    except NotImplementedError as error:
        assert "continuous" in str(error)
    else:
        raise AssertionError("discrete barrier must not use the continuous formula")


def test_geometric_asians():
    market, params = _market(), _params()
    fixings = _t([0.25, 0.5, 0.75, 1.0])
    discrete = AsianOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        is_call=torch.tensor(True),
        fixing_times=fixings,
        average="geometric",
    )
    _same(
        price(discrete, params, market),
        asian_prices.discrete_price(
            spot=market.spot,
            strike=discrete.strike,
            expiry=discrete.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            fixing_times=fixings,
            is_call=discrete.is_call,
        ),
    )
    continuous = AsianOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        is_call=torch.tensor(False),
        average="geometric",
        sampling="continuous",
    )
    _same(
        price(continuous, params, market),
        asian_prices.continuous_price(
            spot=market.spot,
            strike=continuous.strike,
            expiry=continuous.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=continuous.is_call,
        ),
    )


def test_arithmetic_asian_needs_monte_carlo():
    instrument = AsianOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        is_call=torch.tensor(True),
        fixing_times=_t([0.5, 1.0]),
    )
    try:
        Analytic().price(instrument, _params(), _market())
    except NotImplementedError as error:
        assert "arithmetic" in str(error)
    else:
        raise AssertionError("arithmetic Asian has no analytic price")


def test_lookbacks():
    market, params = _market(), _params()
    floating = LookbackOption(expiry=_t(1.0), is_call=torch.tensor(True), extremum=_t(90.0))
    _same(
        price(floating, params, market),
        lookback_prices.floating_price(
            spot=market.spot,
            extremum=floating.extremum,
            expiry=floating.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=floating.is_call,
        ),
    )
    fixed = LookbackOption(
        expiry=_t(1.0), is_call=torch.tensor(False), strike=_t(100.0), extremum=_t(105.0)
    )
    _same(
        price(fixed, params, market),
        lookback_prices.fixed_price(
            spot=market.spot,
            strike=fixed.strike,
            minimum=fixed.extremum,
            expiry=fixed.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=fixed.is_call,
        ),
    )


def test_forward_start_compound_chooser_cliquet_and_double_barrier():
    market, params = _market(), _params()
    forward = ForwardStartOption(
        moneyness=_t(1.1), reset=_t(0.5), expiry=_t(1.0), is_call=torch.tensor(True)
    )
    _same(
        price(forward, params, market),
        forward_start_prices.price(
            spot=market.spot,
            moneyness=forward.moneyness,
            reset=forward.reset,
            expiry=forward.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=forward.is_call,
        ),
    )
    compound = CompoundOption(
        strike=_t(10.0),
        underlying_strike=_t(100.0),
        expiry=_t(0.5),
        underlying_expiry=_t(1.0),
        is_call=torch.tensor(True),
        underlying_is_call=torch.tensor(True),
    )
    _same(
        price(compound, params, market),
        compound_prices.price(
            spot=market.spot,
            strike=compound.strike,
            underlying_strike=compound.underlying_strike,
            expiry=compound.expiry,
            underlying_expiry=compound.underlying_expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=compound.is_call,
            underlying_is_call=compound.underlying_is_call,
        ),
    )
    chooser = ChooserOption(strike=_t(100.0), choose=_t(0.5), expiry=_t(1.0))
    _same(
        price(chooser, params, market),
        chooser_prices.simple_price(
            spot=market.spot,
            strike=chooser.strike,
            choose=chooser.choose,
            expiry=chooser.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
        ),
    )
    complex_chooser = ChooserOption(
        strike=_t(100.0),
        choose=_t(0.25),
        expiry=_t(1.0),
        strike_put=_t(95.0),
        put_expiry=_t(1.0),
    )
    _same(
        price(complex_chooser, params, market),
        chooser_prices.complex_price(
            spot=market.spot,
            strike_call=complex_chooser.strike,
            strike_put=complex_chooser.strike_put,
            choose=complex_chooser.choose,
            call_expiry=complex_chooser.expiry,
            put_expiry=complex_chooser.put_expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
        ),
    )
    resets = _t([0.25, 0.5, 0.75])
    cliquet = CliquetOption(
        moneyness=_t(1.0), reset_times=resets, expiry=_t(1.0), is_call=torch.tensor(True)
    )
    _same(
        price(cliquet, params, market),
        cliquet_prices.price(
            spot=market.spot,
            moneyness=cliquet.moneyness,
            reset_times=resets,
            expiry=cliquet.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=cliquet.is_call,
        ),
    )
    performance = CliquetOption(
        moneyness=_t(1.0),
        reset_times=resets,
        expiry=_t(1.0),
        is_call=torch.tensor(True),
        kind="performance",
    )
    _same(
        price(performance, params, market),
        cliquet_prices.performance_price(
            spot=market.spot,
            moneyness=performance.moneyness,
            reset_times=resets,
            expiry=performance.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=performance.is_call,
        ),
    )
    double = DoubleBarrierOption(
        strike=_t(100.0),
        expiry=_t(1.0),
        barrier_lo=_t(80.0),
        barrier_hi=_t(120.0),
        is_call=torch.tensor(True),
        kind="knock_out",
    )
    _same(
        price(double, params, market),
        double_barrier_prices.price(
            spot=market.spot,
            strike=double.strike,
            barrier_lo=double.barrier_lo,
            barrier_hi=double.barrier_hi,
            expiry=double.expiry,
            rate=market.rate,
            vol=params.vol,
            div=market.div,
            is_call=double.is_call,
            kind=double.kind,
        ),
    )
