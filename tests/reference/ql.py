"""QuantLib reference values in SmallQuant conventions.

SmallQuant works in year fractions with continuously compounded rates. QuantLib works in
dates, so every expiry is expressed as a whole number of days from a fixed evaluation date
under Actual/365 Fixed. Pass expiries as `days / 365` to get an exact match.
"""

from dataclasses import dataclass

import QuantLib as ql

EVALUATION_DATE = ql.Date(15, ql.January, 2025)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()


@dataclass(frozen=True)
class EuropeanResult:
    """Price and Greeks, with Greeks as plain partial derivatives.

    QuantLib's vega and rho are already per unit (1.0 = 100%). Its theta is the derivative
    with respect to calendar time, i.e. -dV/d(expiry); `dexpiry` flips that sign so it can be
    compared directly with autograd on the expiry input.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    rho: float
    dexpiry: float


def european(*, spot, strike, days, rate, vol, div=0.0, is_call=True) -> EuropeanResult:
    """Analytic Black-Scholes-Merton price and Greeks of a European option."""
    ql.Settings.instance().evaluationDate = EVALUATION_DATE

    def flat(r):
        return ql.YieldTermStructureHandle(
            ql.FlatForward(EVALUATION_DATE, r, DAY_COUNT, ql.Continuous)
        )

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        flat(div),
        flat(rate),
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(EVALUATION_DATE, CALENDAR, vol, DAY_COUNT)
        ),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, strike),
        ql.EuropeanExercise(EVALUATION_DATE + int(days)),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return EuropeanResult(
        price=option.NPV(),
        delta=option.delta(),
        gamma=option.gamma(),
        vega=option.vega(),
        rho=option.rho(),
        dexpiry=-option.theta(),
    )
