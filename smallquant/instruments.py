"""Contract terms. Instruments hold no pricing logic."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .payoffs import (
    arithmetic_average_price,
    barrier_survival,
    cash_or_nothing,
    european,
    geometric_average_price,
)


def _grid(expiry: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
    origin = torch.zeros((), dtype=expiry.dtype, device=expiry.device)
    if extra is None:
        return torch.stack((origin, expiry.reshape(-1)[0]))
    return torch.unique(torch.cat((origin.reshape(1), extra.reshape(-1), expiry.reshape(1))), sorted=True)


@dataclass(frozen=True)
class EuropeanOption:
    """Vanilla call or put exercised at ``expiry``."""

    strike: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return european(paths, times, strike=self.strike, is_call=self.is_call)

    def intrinsic(self, spot: torch.Tensor) -> torch.Tensor:
        return torch.where(self.is_call, torch.relu(spot - self.strike), torch.relu(self.strike - spot))

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class DigitalOption:
    """Cash-or-nothing. ``smoothing`` is ``None`` or ``"sigmoid"``."""

    strike: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor
    cash: torch.Tensor
    smoothing: str | None = None
    eps: torch.Tensor | None = None

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return cash_or_nothing(
            paths,
            times,
            strike=self.strike,
            is_call=self.is_call,
            cash=self.cash,
            smoothing=self.smoothing,
            eps=self.eps,
        )

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class BarrierOption:
    """Single barrier. ``kind`` is down/up crossed with in/out, with or without ``_and_``.

    ``monitoring="survival"`` uses the one-step Brownian-bridge probability.
    ``monitoring="discrete"`` knocks on the grid. ``discrete_correction=True``
    applies the Broadie-Glasserman-Kou barrier shift on a discrete grid.
    """

    strike: torch.Tensor
    expiry: torch.Tensor
    barrier: torch.Tensor
    is_call: torch.Tensor
    kind: str
    rebate: torch.Tensor | None = None
    monitoring: str = "discrete"
    vol: torch.Tensor | None = None
    discrete_correction: bool = False

    def __post_init__(self) -> None:
        normalized = self.kind.replace("_and_", "_")
        if normalized not in ("down_in", "down_out", "up_in", "up_out"):
            raise ValueError(f"unknown barrier kind {self.kind!r}")
        object.__setattr__(self, "kind", normalized)

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        barrier = self._barrier(times)
        if self.monitoring == "survival":
            if self.vol is None:
                raise ValueError("survival monitoring requires vol")
            survival_kind = "down_and_out" if self.kind == "down_out" else "up_and_out"
            if self.kind not in ("down_out", "up_out"):
                raise ValueError("survival monitoring is implemented for knock-outs")
            return barrier_survival(
                paths,
                times,
                strike=self.strike,
                is_call=self.is_call,
                barrier=barrier,
                vol=self.vol,
                kind=survival_kind,
            )
        spot = paths[..., 0]
        if self.kind.startswith("down"):
            hit = (spot <= barrier).any(dim=-1)
        else:
            hit = (spot >= barrier).any(dim=-1)
        vanilla = european(paths, times, strike=self.strike, is_call=self.is_call)
        rebate = self.rebate if self.rebate is not None else torch.zeros_like(vanilla)
        knocked_out = self.kind.endswith("out")
        alive = torch.where(knocked_out, ~hit, hit)
        return torch.where(alive, vanilla, rebate)

    def _barrier(self, times: torch.Tensor) -> torch.Tensor:
        if not self.discrete_correction or self.vol is None:
            return self.barrier
        dt = (times[1:] - times[:-1]).mean()
        beta = 0.5826
        shift = torch.exp(beta * self.vol * torch.sqrt(dt))
        if self.kind.startswith("down"):
            return self.barrier * shift
        return self.barrier / shift

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class AsianOption:
    """Average-price option. ``average`` is ``"arithmetic"`` or ``"geometric"``."""

    strike: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor
    fixing_times: torch.Tensor | None = None
    average: str = "arithmetic"
    sampling: str = "discrete"

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        if self.average == "geometric":
            return geometric_average_price(paths, times, strike=self.strike, is_call=self.is_call)
        if self.average == "arithmetic":
            return arithmetic_average_price(paths, times, strike=self.strike, is_call=self.is_call)
        raise ValueError("average must be 'arithmetic' or 'geometric'")

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry, self.fixing_times)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class LookbackOption:
    """Floating-strike lookback on the running extremum, or a fixed strike.

    ``extremum`` is the minimum seen so far for a call and the maximum seen so
    far for a put. Omit it when the option is new and the extremum is the spot.
    """

    expiry: torch.Tensor
    is_call: torch.Tensor
    strike: torch.Tensor | None = None
    extremum: torch.Tensor | None = None

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del times
        spot = paths[..., -1, 0]
        minimum = paths[..., 0].min(dim=-1).values
        maximum = paths[..., 0].max(dim=-1).values
        floating = torch.where(self.is_call, torch.relu(spot - minimum), torch.relu(maximum - spot))
        if self.strike is None:
            return floating
        fixed = torch.where(self.is_call, torch.relu(maximum - self.strike), torch.relu(self.strike - minimum))
        return fixed

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class AmericanOption:
    """Early exercise on ``exercise_times``. The intrinsic value is the exercise payoff."""

    strike: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor
    exercise_grid: torch.Tensor | None = None

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return european(paths, times, strike=self.strike, is_call=self.is_call)

    def intrinsic(self, spot: torch.Tensor) -> torch.Tensor:
        return torch.where(self.is_call, torch.relu(spot - self.strike), torch.relu(self.strike - spot))

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry, self.exercise_grid)

    @property
    def exercise_times(self) -> torch.Tensor:
        if self.exercise_grid is None:
            return self.observation_times
        return self.exercise_grid


@dataclass(frozen=True)
class ForwardStartOption:
    """Vanilla struck at ``moneyness`` times the spot on the reset date."""

    moneyness: torch.Tensor
    reset: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        index = int(torch.argmin(torch.abs(times - self.reset)).item())
        strike = self.moneyness * paths[:, index, 0]
        terminal = paths[:, -1, 0]
        return torch.where(self.is_call, torch.relu(terminal - strike), torch.relu(strike - terminal))

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry, self.reset)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class CompoundOption:
    """Option on an option. Priced by the analytic engine."""

    strike: torch.Tensor
    underlying_strike: torch.Tensor
    expiry: torch.Tensor
    underlying_expiry: torch.Tensor
    is_call: torch.Tensor
    underlying_is_call: torch.Tensor

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del paths, times
        raise NotImplementedError("CompoundOption is priced by the analytic engine")

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.underlying_expiry, self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class ChooserOption:
    """Chooser. Omit ``strike_put`` for a simple chooser with one strike and one expiry."""

    strike: torch.Tensor
    choose: torch.Tensor
    expiry: torch.Tensor
    strike_put: torch.Tensor | None = None
    put_expiry: torch.Tensor | None = None

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del paths, times
        raise NotImplementedError("ChooserOption is priced by the analytic engine")

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry, self.choose)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class CliquetOption:
    """Sum of forward-starting vanillas on ``reset_times``. ``kind`` is ``vanilla`` or ``performance``."""

    moneyness: torch.Tensor
    reset_times: torch.Tensor
    expiry: torch.Tensor
    is_call: torch.Tensor
    kind: str = "vanilla"

    def __post_init__(self) -> None:
        if self.kind not in ("vanilla", "performance"):
            raise ValueError("kind must be 'vanilla' or 'performance'")

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        del paths, times
        raise NotImplementedError("CliquetOption is priced by the analytic engine")

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry, self.reset_times)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)


@dataclass(frozen=True)
class DoubleBarrierOption:
    """European option with a lower and an upper barrier."""

    strike: torch.Tensor
    expiry: torch.Tensor
    barrier_lo: torch.Tensor
    barrier_hi: torch.Tensor
    is_call: torch.Tensor
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("knock_in", "knock_out"):
            raise ValueError(f"unknown double-barrier kind {self.kind!r}")

    def payoff(self, paths: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        spot = paths[..., 0]
        hit = (spot <= self.barrier_lo).any(dim=-1) | (spot >= self.barrier_hi).any(dim=-1)
        vanilla = european(paths, times, strike=self.strike, is_call=self.is_call)
        knocked_out = self.kind == "knock_out"
        alive = torch.where(knocked_out, ~hit, hit)
        return torch.where(alive, vanilla, torch.zeros_like(vanilla))

    @property
    def observation_times(self) -> torch.Tensor:
        return _grid(self.expiry)

    @property
    def exercise_times(self) -> torch.Tensor:
        return self.expiry.reshape(-1)
