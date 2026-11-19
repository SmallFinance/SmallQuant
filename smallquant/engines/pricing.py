"""Engine objects. Each ``price`` takes an instrument, parameters and a market."""

from __future__ import annotations

import torch

from smallquant.black_scholes import asian_prices
from smallquant.black_scholes.crr_binomial_tree import option_price_binomial
from smallquant.black_scholes.vanilla_prices import barrier_price, binary_price, option_price
from smallquant.exotics import (
    chooser_prices,
    cliquet_prices,
    compound_prices,
    double_barrier_prices,
    forward_start_prices,
    lookback_prices,
)
from smallquant.instruments import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    ChooserOption,
    CliquetOption,
    CompoundOption,
    DigitalOption,
    DoubleBarrierOption,
    EuropeanOption,
    ForwardStartOption,
    LookbackOption,
)
from smallquant.market import Market
from smallquant.models import geometric_brownian_motion, heston, local_vol, merton, sabr
from smallquant.models.params import (
    BachelierParams,
    Black76Params,
    BlackScholesParams,
    HestonParams,
    LocalVolParams,
    MertonParams,
    SABRParams,
)

from . import finite_difference, fourier
from .longstaff import discounted_exercise, exercise_index
from .monte_carlo import price as monte_carlo_price


class _Engine:
    registry: dict

    def supports(self, instrument, model_params) -> bool:
        return (type(instrument), type(model_params)) in self.registry

    def supported_pairs(self) -> list[tuple[str, str]]:
        return [(instrument.__name__, params.__name__) for instrument, params in self.registry]

    def price(self, instrument, model_params, market: Market):
        key = (type(instrument), type(model_params))
        if key not in self.registry:
            listed = ", ".join(f"{left}/{right}" for left, right in self.supported_pairs())
            raise NotImplementedError(
                f"{type(self).__name__} does not price {type(instrument).__name__} "
                f"with {type(model_params).__name__}. Supported: {listed}"
            )
        return self.registry[key](self, instrument, model_params, market)


def _rate(market, expiry):
    return market.rate_at(expiry)


def _div(market, expiry):
    return market.div_at(expiry)


class Analytic(_Engine):
    """Closed-form and characteristic-function prices.

    ``num_terms`` is the Merton jump series length; ``fourier_terms`` is the COS term
    count for Heston.
    """

    def __init__(self, num_terms: int = 128, fourier_terms: int = 256):
        self.num_terms = num_terms
        self.fourier_terms = fourier_terms

    def _european_black(self, instrument: EuropeanOption, params: BlackScholesParams, market: Market):
        return option_price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=instrument.expiry,
            rate=_rate(market, instrument.expiry),
            vol=params.vol,
            div=_div(market, instrument.expiry),
            is_call=instrument.is_call,
        )

    def _european_black76(self, instrument: EuropeanOption, params: Black76Params, market: Market):
        expiry = instrument.expiry
        forward = market.spot * torch.exp((_rate(market, expiry) - _div(market, expiry)) * expiry)
        return option_price(
            forward=forward,
            strike=instrument.strike,
            expiry=expiry,
            vol=params.vol,
            discount=market.discount_at(expiry),
            is_call=instrument.is_call,
        )

    def _european_bachelier(self, instrument: EuropeanOption, params: BachelierParams, market: Market):
        expiry = instrument.expiry
        forward = market.spot * torch.exp((_rate(market, expiry) - _div(market, expiry)) * expiry)
        return option_price(
            forward=forward,
            strike=instrument.strike,
            expiry=expiry,
            normal_vol=params.normal_vol,
            discount=market.discount_at(expiry),
            is_call=instrument.is_call,
        )

    def _european_merton(self, instrument: EuropeanOption, params: MertonParams, market: Market):
        return merton.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=instrument.expiry,
            rate=_rate(market, instrument.expiry),
            vol=params.vol,
            div=_div(market, instrument.expiry),
            is_call=instrument.is_call,
            jump_intensity=params.jump_intensity,
            jump_mean=params.jump_mean,
            jump_vol=params.jump_vol,
            num_terms=self.num_terms,
        )

    def _european_heston(self, instrument: EuropeanOption, params: HestonParams, market: Market):
        expiry = instrument.expiry
        return fourier.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=_rate(market, expiry),
            div=_div(market, expiry),
            is_call=instrument.is_call,
            characteristic=fourier.heston_characteristic(
                spot=market.spot,
                rate=_rate(market, expiry),
                div=_div(market, expiry),
                expiry=expiry,
                v0=params.v0,
                kappa=params.kappa,
                theta=params.theta,
                sigma=params.sigma,
                rho=params.rho,
            ),
            method="cos",
            num_terms=self.fourier_terms,
        )

    def _european_sabr(self, instrument: EuropeanOption, params: SABRParams, market: Market):
        expiry = instrument.expiry
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        forward = market.spot * torch.exp((rate - div) * expiry)
        return sabr.approximations.european_option_price(
            forward=forward,
            strike=instrument.strike,
            expiry=expiry,
            discount=market.discount_at(expiry),
            alpha=params.alpha,
            beta=params.beta,
            rho=params.rho,
            nu=params.nu,
            is_call=instrument.is_call,
        )

    def _digital_black(self, instrument: DigitalOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        unit = binary_price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
        )
        return instrument.cash * unit

    def _barrier_black(self, instrument: BarrierOption, params: BlackScholesParams, market: Market):
        if instrument.monitoring != "continuous":
            raise NotImplementedError(
                "Analytic prices the continuous barrier. Set monitoring='continuous', "
                "or pass a MonteCarlo engine for discrete monitoring."
            )
        expiry = instrument.expiry
        rebate = instrument.rebate
        if rebate is None:
            rebate = torch.zeros((), dtype=market.spot.dtype, device=market.spot.device)
        return barrier_price(
            spot=market.spot,
            strike=instrument.strike,
            barrier=instrument.barrier,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            rebate=rebate,
            is_call=instrument.is_call,
            kind=instrument.kind,
        )

    def _asian_black(self, instrument: AsianOption, params: BlackScholesParams, market: Market):
        if instrument.average != "geometric":
            raise NotImplementedError(
                "Analytic prices geometric Asians. Pass a MonteCarlo engine for an arithmetic Asian."
            )
        expiry = instrument.expiry
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        if instrument.sampling == "continuous":
            return asian_prices.continuous_price(
                spot=market.spot,
                strike=instrument.strike,
                expiry=expiry,
                rate=rate,
                vol=params.vol,
                div=div,
                is_call=instrument.is_call,
            )
        if instrument.sampling != "discrete":
            raise ValueError("sampling must be 'discrete' or 'continuous'")
        if instrument.fixing_times is None:
            raise ValueError("a discrete geometric Asian requires fixing_times")
        return asian_prices.discrete_price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=rate,
            vol=params.vol,
            div=div,
            fixing_times=instrument.fixing_times,
            is_call=instrument.is_call,
        )

    def _lookback_black(self, instrument: LookbackOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        extremum = market.spot if instrument.extremum is None else instrument.extremum
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        if instrument.strike is None:
            return lookback_prices.floating_price(
                spot=market.spot,
                extremum=extremum,
                expiry=expiry,
                rate=rate,
                vol=params.vol,
                div=div,
                is_call=instrument.is_call,
            )
        return lookback_prices.fixed_price(
            spot=market.spot,
            strike=instrument.strike,
            minimum=extremum,
            expiry=expiry,
            rate=rate,
            vol=params.vol,
            div=div,
            is_call=instrument.is_call,
        )

    def _forward_black(self, instrument: ForwardStartOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        return forward_start_prices.price(
            spot=market.spot,
            moneyness=instrument.moneyness,
            reset=instrument.reset,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
        )

    def _compound_black(self, instrument: CompoundOption, params: BlackScholesParams, market: Market):
        return compound_prices.price(
            spot=market.spot,
            strike=instrument.strike,
            underlying_strike=instrument.underlying_strike,
            expiry=instrument.expiry,
            underlying_expiry=instrument.underlying_expiry,
            rate=_rate(market, instrument.expiry),
            vol=params.vol,
            div=_div(market, instrument.expiry),
            is_call=instrument.is_call,
            underlying_is_call=instrument.underlying_is_call,
        )

    def _chooser_black(self, instrument: ChooserOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        if instrument.strike_put is None:
            return chooser_prices.simple_price(
                spot=market.spot,
                strike=instrument.strike,
                choose=instrument.choose,
                expiry=expiry,
                rate=rate,
                vol=params.vol,
                div=div,
            )
        put_expiry = instrument.expiry if instrument.put_expiry is None else instrument.put_expiry
        return chooser_prices.complex_price(
            spot=market.spot,
            strike_call=instrument.strike,
            strike_put=instrument.strike_put,
            choose=instrument.choose,
            call_expiry=expiry,
            put_expiry=put_expiry,
            rate=rate,
            vol=params.vol,
            div=div,
        )

    def _cliquet_black(self, instrument: CliquetOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        kwargs = dict(
            spot=market.spot,
            moneyness=instrument.moneyness,
            reset_times=instrument.reset_times,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
        )
        if instrument.kind == "performance":
            return cliquet_prices.performance_price(**kwargs)
        return cliquet_prices.price(**kwargs)

    def _double_barrier_black(self, instrument: DoubleBarrierOption, params: BlackScholesParams, market: Market):
        expiry = instrument.expiry
        return double_barrier_prices.price(
            spot=market.spot,
            strike=instrument.strike,
            barrier_lo=instrument.barrier_lo,
            barrier_hi=instrument.barrier_hi,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
            kind=instrument.kind,
        )


Analytic.registry = {
    (EuropeanOption, BlackScholesParams): Analytic._european_black,
    (EuropeanOption, Black76Params): Analytic._european_black76,
    (EuropeanOption, BachelierParams): Analytic._european_bachelier,
    (EuropeanOption, MertonParams): Analytic._european_merton,
    (EuropeanOption, HestonParams): Analytic._european_heston,
    (EuropeanOption, SABRParams): Analytic._european_sabr,
    (DigitalOption, BlackScholesParams): Analytic._digital_black,
    (BarrierOption, BlackScholesParams): Analytic._barrier_black,
    (AsianOption, BlackScholesParams): Analytic._asian_black,
    (LookbackOption, BlackScholesParams): Analytic._lookback_black,
    (ForwardStartOption, BlackScholesParams): Analytic._forward_black,
    (CompoundOption, BlackScholesParams): Analytic._compound_black,
    (ChooserOption, BlackScholesParams): Analytic._chooser_black,
    (CliquetOption, BlackScholesParams): Analytic._cliquet_black,
    (DoubleBarrierOption, BlackScholesParams): Analytic._double_barrier_black,
}


class MonteCarlo(_Engine):
    """Batched Monte Carlo. Same seed, device and dtype reproduce the paths."""

    def __init__(
        self,
        num_paths: int = 8192,
        seed: int = 1,
        antithetic: bool = True,
        brownian_bridge: bool = False,
        chunk_size: int | None = None,
        moment_match: bool = False,
        greeks_method: str = "pathwise",
        degree: int = 2,
        steps: int = 50,
    ):
        self.num_paths = num_paths
        self.seed = seed
        self.antithetic = antithetic
        self.brownian_bridge = brownian_bridge
        self.chunk_size = chunk_size
        self.moment_match = moment_match
        self.greeks_method = greeks_method
        self.degree = degree
        self.steps = steps

    def _times(self, instrument, market):
        if hasattr(instrument, "observation_times"):
            grid = instrument.observation_times
            if grid.numel() >= 2:
                return grid
        count = self.steps
        expiry = instrument.expiry
        return torch.arange(count + 1, dtype=expiry.dtype, device=expiry.device) * (expiry / count)

    def _european_black(self, instrument, params, market):
        times = self._times(instrument, market)
        rate = _rate(market, instrument.expiry)
        div = _div(market, instrument.expiry)

        def step(normals):
            return geometric_brownian_motion.step(
                spot=market.spot, times=times, normals=normals, rate=rate, vol=params.vol, div=div
            )

        score = None
        if self.greeks_method == "likelihood_ratio":

            def score(normals):
                return geometric_brownian_motion.likelihood_weight(
                    normals=normals, times=times, spot=market.spot, vol=params.vol
                )

        return monte_carlo_price(
            instrument.payoff,
            step,
            times=times,
            discount=market.discount_at(instrument.expiry),
            num_paths=self.num_paths,
            seed=self.seed,
            antithetic=self.antithetic,
            brownian_bridge=self.brownian_bridge,
            chunk_size=self.chunk_size,
            moment_match=self.moment_match,
            greeks_method=self.greeks_method,
            score=score,
        )

    def _american_black(self, instrument: AmericanOption, params: BlackScholesParams, market: Market):
        times = self._times(instrument, market)
        rate = _rate(market, instrument.expiry)
        div = _div(market, instrument.expiry)

        def step(normals):
            return geometric_brownian_motion.step(
                spot=market.spot, times=times, normals=normals, rate=rate, vol=params.vol, div=div
            )

        from smallquant.math.random import standard_normal

        normals = standard_normal(
            num_paths=self.num_paths,
            num_steps=times.numel() - 1,
            dim=1,
            seed=self.seed,
            dtype=times.dtype,
            device=times.device,
            antithetic=self.antithetic,
        )
        paths = step(normals)
        when = exercise_index(paths, times, instrument.intrinsic, rate, self.degree)
        # Re-simulate under grad with the frozen policy by stepping again.
        paths = step(normals)
        return discounted_exercise(paths, times, instrument.intrinsic, rate, when).mean()

    def _heston(self, instrument, params: HestonParams, market: Market):
        times = self._times(instrument, market)
        rate = _rate(market, instrument.expiry)
        div = _div(market, instrument.expiry)

        def step(normals):
            return heston.step(
                spot=market.spot,
                v0=params.v0,
                times=times,
                normals=normals,
                rate=rate,
                div=div,
                kappa=params.kappa,
                theta=params.theta,
                sigma=params.sigma,
                rho=params.rho,
            )

        return monte_carlo_price(
            instrument.payoff,
            step,
            times=times,
            discount=market.discount_at(instrument.expiry),
            num_paths=self.num_paths,
            seed=self.seed,
            dim=2,
            antithetic=self.antithetic,
            chunk_size=self.chunk_size,
            moment_match=self.moment_match,
        )

    def _local(self, instrument, params: LocalVolParams, market: Market):
        times = self._times(instrument, market)
        rate = _rate(market, instrument.expiry)
        div = _div(market, instrument.expiry)

        def step(normals):
            return local_vol.step(
                spot=market.spot,
                times=times,
                normals=normals,
                rate=rate,
                div=div,
                strikes=params.strikes,
                expiries=params.expiries,
                local_vols=params.local_vols,
            )

        return monte_carlo_price(
            instrument.payoff,
            step,
            times=times,
            discount=market.discount_at(instrument.expiry),
            num_paths=self.num_paths,
            seed=self.seed,
            antithetic=self.antithetic,
            chunk_size=self.chunk_size,
        )


MonteCarlo.registry = {
    (EuropeanOption, BlackScholesParams): MonteCarlo._european_black,
    (DigitalOption, BlackScholesParams): MonteCarlo._european_black,
    (AsianOption, BlackScholesParams): MonteCarlo._european_black,
    (BarrierOption, BlackScholesParams): MonteCarlo._european_black,
    (ForwardStartOption, BlackScholesParams): MonteCarlo._european_black,
    (DoubleBarrierOption, BlackScholesParams): MonteCarlo._european_black,
    (AmericanOption, BlackScholesParams): MonteCarlo._american_black,
    (EuropeanOption, HestonParams): MonteCarlo._heston,
    (EuropeanOption, LocalVolParams): MonteCarlo._local,
}


class PDE(_Engine):
    def __init__(self, num_space: int = 200, num_time: int = 200, theta: float = 0.5, rannacher_steps: int = 2):
        self.num_space = num_space
        self.num_time = num_time
        self.theta = theta
        self.rannacher_steps = rannacher_steps

    def _vanilla(self, instrument, params, market, exercise):
        expiry = instrument.expiry
        return finite_difference.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
            exercise=exercise,
            num_space=self.num_space,
            num_time=self.num_time,
            theta=self.theta,
            rannacher_steps=self.rannacher_steps,
            barrier=getattr(instrument, "barrier", None),
            barrier_kind=getattr(instrument, "kind", None) if isinstance(instrument, BarrierOption) else None,
            rebate=getattr(instrument, "rebate", None),
        )

    def _european(self, instrument, params, market):
        return self._vanilla(instrument, params, market, "european")

    def _american(self, instrument, params, market):
        return self._vanilla(instrument, params, market, "american")

    def _local(self, instrument, params: LocalVolParams, market):
        expiry = instrument.expiry

        def local(spots, time):
            return local_vol._local(spots, time, params.strikes, params.expiries, params.local_vols)

        return finite_difference.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.local_vols.mean(),
            div=_div(market, expiry),
            is_call=instrument.is_call,
            num_space=self.num_space,
            num_time=self.num_time,
            theta=self.theta,
            rannacher_steps=self.rannacher_steps,
            local_vol=local,
        )


PDE.registry = {
    (EuropeanOption, BlackScholesParams): PDE._european,
    (AmericanOption, BlackScholesParams): PDE._american,
    (BarrierOption, BlackScholesParams): PDE._european,
    (EuropeanOption, LocalVolParams): PDE._local,
}


class Fourier(_Engine):
    def __init__(self, method: str = "cos", num_terms: int = 256):
        self.method = method
        self.num_terms = num_terms

    def _heston(self, instrument, params: HestonParams, market):
        expiry = instrument.expiry
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        return fourier.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=rate,
            div=div,
            is_call=instrument.is_call,
            characteristic=fourier.heston_characteristic(
                spot=market.spot,
                rate=rate,
                div=div,
                expiry=expiry,
                v0=params.v0,
                kappa=params.kappa,
                theta=params.theta,
                sigma=params.sigma,
                rho=params.rho,
            ),
            method=self.method,
            num_terms=self.num_terms,
        )

    def _black(self, instrument, params: BlackScholesParams, market):
        expiry = instrument.expiry
        rate = _rate(market, expiry)
        div = _div(market, expiry)
        return fourier.price(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=rate,
            div=div,
            is_call=instrument.is_call,
            characteristic=fourier.black_characteristic(
                spot=market.spot, rate=rate, div=div, vol=params.vol, expiry=expiry
            ),
            method=self.method,
            num_terms=self.num_terms,
        )


Fourier.registry = {
    (EuropeanOption, HestonParams): Fourier._heston,
    (EuropeanOption, BlackScholesParams): Fourier._black,
}


class Lattice(_Engine):
    def __init__(self, num_steps: int = 200, kind: str = "crr"):
        self.num_steps = num_steps
        self.kind = kind

    def _price(self, instrument, params, market, exercise):
        expiry = instrument.expiry
        return option_price_binomial(
            spot=market.spot,
            strike=instrument.strike,
            expiry=expiry,
            rate=_rate(market, expiry),
            vol=params.vol,
            div=_div(market, expiry),
            is_call=instrument.is_call,
            exercise=exercise,
            num_steps=self.num_steps,
            kind=self.kind,
        )

    def _european(self, instrument, params, market):
        return self._price(instrument, params, market, "european")

    def _american(self, instrument, params, market):
        return self._price(instrument, params, market, "american")


Lattice.registry = {
    (EuropeanOption, BlackScholesParams): Lattice._european,
    (AmericanOption, BlackScholesParams): Lattice._american,
}
