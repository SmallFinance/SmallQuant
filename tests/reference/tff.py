"""TF Quant Finance reference values in SmallQuant conventions.

Expiries are year fractions (`days / 365`), the same Actual/365 Fixed year fraction
QuantLib uses in `tests.reference.ql`. Rates are continuously compounded. Greeks are
TensorFlow autodiff of `option_price`, the same partials QuantLib reports.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# tf-quant-finance's checked-in protobufs predate the installed protobuf runtime.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tf_quant_finance.black_scholes import vanilla_prices


@dataclass(frozen=True)
class EuropeanResult:
    """Price and Greeks, with Greeks as plain partial derivatives."""

    price: float | np.ndarray
    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    rho: float | np.ndarray
    dexpiry: float | np.ndarray


def _as_float(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _out(value):
    array = np.array(value.numpy(), dtype=np.float64, copy=True)
    if array.ndim == 0:
        return float(array)
    return array


def european(*, spot, strike, days, rate, vol, div=0.0, is_call=True) -> EuropeanResult:
    """Black-Scholes-Merton price and Greeks from TF Quant Finance."""
    spot_t = tf.constant(_as_float(spot), dtype=tf.float64)
    strike_t = tf.constant(_as_float(strike), dtype=tf.float64)
    expiry_t = tf.constant(_as_float(days), dtype=tf.float64) / 365.0
    rate_t = tf.constant(_as_float(rate), dtype=tf.float64)
    vol_t = tf.constant(_as_float(vol), dtype=tf.float64)
    div_t = tf.constant(_as_float(div), dtype=tf.float64)

    with tf.GradientTape() as outer:
        outer.watch(spot_t)
        with tf.GradientTape(persistent=True) as inner:
            inner.watch((spot_t, expiry_t, rate_t, vol_t))
            price = vanilla_prices.option_price(
                volatilities=vol_t,
                strikes=strike_t,
                expiries=expiry_t,
                spots=spot_t,
                discount_rates=rate_t,
                dividend_rates=div_t,
                is_call_options=is_call,
                dtype=tf.float64,
            )
        delta, dexpiry, rho, vega = inner.gradient(price, (spot_t, expiry_t, rate_t, vol_t))
        del inner
    gamma = outer.gradient(delta, spot_t)

    return EuropeanResult(
        price=_out(price),
        delta=_out(delta),
        gamma=_out(gamma),
        vega=_out(vega),
        rho=_out(rho),
        dexpiry=_out(dexpiry),
    )


def _contract(*, spot, strike, days, rate, vol, div):
    """Inputs for ``binary_price``.

    That function ignores ``dividend_rates`` when ``spots`` is passed and builds the
    forward as ``spot * exp(rate * expiry)``. The dividend-adjusted forward is passed
    instead, so the reference prices the same contract as QuantLib.
    """
    expiry = _as_float(days) / 365.0
    spot = _as_float(spot)
    rate = _as_float(rate)
    div = _as_float(div)
    forward = spot * np.exp((rate - div) * expiry)
    return dict(
        volatilities=_as_float(vol),
        strikes=_as_float(strike),
        expiries=expiry,
        forwards=forward,
        discount_rates=rate,
        dtype=tf.float64,
    )


def binary(*, spot, strike, days, rate, vol, div=0.0, is_call=True) -> float:
    """Cash-or-nothing price that pays one unit of cash."""
    value = vanilla_prices.binary_price(is_call_options=is_call, **_contract(
        spot=spot, strike=strike, days=days, rate=rate, vol=vol, div=div
    ))
    return float(value.numpy())


def asset_or_nothing(*, spot, strike, days, rate, vol, div=0.0, is_call=True) -> float:
    """Asset-or-nothing price. Spot inputs do carry the dividend yield."""
    expiry = _as_float(days) / 365.0
    value = vanilla_prices.asset_or_nothing_price(
        volatilities=_as_float(vol),
        strikes=_as_float(strike),
        expiries=expiry,
        spots=_as_float(spot),
        discount_rates=_as_float(rate),
        dividend_rates=_as_float(div),
        is_call_options=is_call,
        dtype=tf.float64,
    )
    return float(value.numpy())
