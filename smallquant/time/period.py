"""Time units, frequencies and periods (QuantLib `Period`, `Frequency`, `TimeUnit`)."""

from __future__ import annotations

from enum import IntEnum


class TimeUnit(IntEnum):
    Days = 0
    Weeks = 1
    Months = 2
    Years = 3


class Frequency(IntEnum):
    """Number of events per year; `NoFrequency` and `Once` are special values."""

    NoFrequency = -1
    Once = 0
    Annual = 1
    Semiannual = 2
    EveryFourthMonth = 3
    Quarterly = 4
    Bimonthly = 6
    Monthly = 12
    EveryFourthWeek = 13
    Biweekly = 26
    Weekly = 52
    Daily = 365
    OtherFrequency = 999


_SUFFIX = {"D": TimeUnit.Days, "W": TimeUnit.Weeks, "M": TimeUnit.Months, "Y": TimeUnit.Years}
_LETTER = {unit: letter for letter, unit in _SUFFIX.items()}


def _days_min_max(p: Period) -> tuple[int, int]:
    n = p.length
    if p.units == TimeUnit.Days:
        return n, n
    if p.units == TimeUnit.Weeks:
        return 7 * n, 7 * n
    if p.units == TimeUnit.Months:
        return 28 * n, 31 * n
    return 365 * n, 366 * n


class Period:
    """A length of time as a signed integer count of `TimeUnit`s.

    Mirrors QuantLib: `==` and `<` compare periods across units where that is decidable
    (12M == 1Y, 7D == 1W) and `<` raises `ValueError` where it is not (1M vs 30D). `==`
    returns False instead of raising for undecidable pairs, since Python expects `==` not to
    raise. Arithmetic never normalizes; call `normalized()` for that.

    Example:
        >>> Period(6, TimeUnit.Months) + Period(1, TimeUnit.Years)
        Period('18M')
        >>> Period.parse("2W") < Period.parse("1M")
        True
    """

    __slots__ = ("_length", "_units")

    def __init__(self, length: int | Frequency = 0, units: TimeUnit = TimeUnit.Days):
        if isinstance(length, Frequency):
            length, units = self._from_frequency(length)
        self._length = int(length)
        self._units = TimeUnit(units)

    @staticmethod
    def _from_frequency(f: Frequency) -> tuple[int, TimeUnit]:
        if f == Frequency.NoFrequency:
            return 0, TimeUnit.Days
        if f == Frequency.Once:
            return 0, TimeUnit.Years
        if f == Frequency.Annual:
            return 1, TimeUnit.Years
        if f in (
            Frequency.Semiannual,
            Frequency.EveryFourthMonth,
            Frequency.Quarterly,
            Frequency.Bimonthly,
            Frequency.Monthly,
        ):
            return 12 // f, TimeUnit.Months
        if f in (Frequency.EveryFourthWeek, Frequency.Biweekly, Frequency.Weekly):
            return 52 // f, TimeUnit.Weeks
        if f == Frequency.Daily:
            return 1, TimeUnit.Days
        raise ValueError(f"unknown frequency ({int(f)})")

    @classmethod
    def parse(cls, text: str) -> Period:
        """Parse strings such as "3M", "1Y", "-2W", "10D"."""
        s = text.strip().upper()
        if len(s) < 2 or s[-1] not in _SUFFIX:
            raise ValueError(f"unknown period string {text!r}")
        try:
            n = int(s[:-1])
        except ValueError:
            raise ValueError(f"unknown period string {text!r}") from None
        return cls(n, _SUFFIX[s[-1]])

    @property
    def length(self) -> int:
        return self._length

    @property
    def units(self) -> TimeUnit:
        return self._units

    def frequency(self) -> Frequency:
        n = abs(self._length)
        if n == 0:
            return Frequency.Once if self._units == TimeUnit.Years else Frequency.NoFrequency
        if self._units == TimeUnit.Years:
            return Frequency.Annual if n == 1 else Frequency.OtherFrequency
        if self._units == TimeUnit.Months:
            return Frequency(12 // n) if 12 % n == 0 and n <= 12 else Frequency.OtherFrequency
        if self._units == TimeUnit.Weeks:
            return {1: Frequency.Weekly, 2: Frequency.Biweekly, 4: Frequency.EveryFourthWeek}.get(
                n, Frequency.OtherFrequency
            )
        return Frequency.Daily if n == 1 else Frequency.OtherFrequency

    def normalized(self) -> Period:
        """12M -> 1Y, 14D -> 2W, 0 of any unit -> 0D."""
        n, u = self._length, self._units
        if n == 0:
            return Period(0, TimeUnit.Days)
        if u == TimeUnit.Months and n % 12 == 0:
            return Period(n // 12, TimeUnit.Years)
        if u == TimeUnit.Days and n % 7 == 0:
            return Period(n // 7, TimeUnit.Weeks)
        return Period(n, u)

    def years(self) -> float:
        return self._convert("Years", {TimeUnit.Months: 1 / 12, TimeUnit.Years: 1})

    def months(self) -> float:
        return self._convert("Months", {TimeUnit.Months: 1, TimeUnit.Years: 12})

    def weeks(self) -> float:
        return self._convert("Weeks", {TimeUnit.Days: 1 / 7, TimeUnit.Weeks: 1})

    def days(self) -> float:
        return self._convert("Days", {TimeUnit.Days: 1, TimeUnit.Weeks: 7})

    def _convert(self, target: str, factors: dict[TimeUnit, float]) -> float:
        if self._length == 0:
            return 0.0
        if self._units not in factors:
            raise ValueError(f"cannot convert {self._units.name} into {target}")
        return self._length * factors[self._units]

    def __neg__(self) -> Period:
        return Period(-self._length, self._units)

    def __add__(self, other: Period) -> Period:
        if not isinstance(other, Period):
            return NotImplemented
        if self._length == 0:
            return Period(other._length, other._units)
        if self._units == other._units:
            return Period(self._length + other._length, self._units)
        pair = (self._units, other._units)
        U = TimeUnit
        if pair == (U.Years, U.Months):
            return Period(self._length * 12 + other._length, U.Months)
        if pair == (U.Months, U.Years):
            return Period(self._length + other._length * 12, U.Months)
        if pair == (U.Weeks, U.Days):
            return Period(self._length * 7 + other._length, U.Days)
        if pair == (U.Days, U.Weeks):
            return Period(self._length + other._length * 7, U.Days)
        if other._length == 0:
            return Period(self._length, self._units)
        raise ValueError(f"impossible addition between {self} and {other}")

    def __sub__(self, other: Period) -> Period:
        if not isinstance(other, Period):
            return NotImplemented
        return self + (-other)

    def __mul__(self, n: int) -> Period:
        if not isinstance(n, int):
            return NotImplemented
        return Period(self._length * n, self._units)

    __rmul__ = __mul__

    def __truediv__(self, n: int) -> Period:
        if not isinstance(n, int):
            return NotImplemented
        if n == 0:
            raise ValueError("cannot be divided by zero")
        if self._length % n == 0:
            return Period(self._length // n, self._units)
        length, units = self._length, self._units
        if units == TimeUnit.Years:
            length, units = length * 12, TimeUnit.Months
        elif units == TimeUnit.Weeks:
            length, units = length * 7, TimeUnit.Days
        if length % n != 0:
            raise ValueError(f"{self} cannot be divided by {n}")
        return Period(length // n, units)

    def __lt__(self, other: Period) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        n1, u1, n2, u2 = self._length, self._units, other._length, other._units
        if n1 == 0:
            return n2 > 0
        if n2 == 0:
            return n1 < 0
        U = TimeUnit
        if u1 == u2:
            return n1 < n2
        if (u1, u2) == (U.Months, U.Years):
            return n1 < 12 * n2
        if (u1, u2) == (U.Years, U.Months):
            return 12 * n1 < n2
        if (u1, u2) == (U.Days, U.Weeks):
            return n1 < 7 * n2
        if (u1, u2) == (U.Weeks, U.Days):
            return 7 * n1 < n2
        lo1, hi1 = _days_min_max(self)
        lo2, hi2 = _days_min_max(other)
        if hi1 < lo2:
            return True
        if lo1 > hi2:
            return False
        raise ValueError(f"undecidable comparison between {self} and {other}")

    def __gt__(self, other: Period) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        return other < self

    def __le__(self, other: Period) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        return not other < self

    def __ge__(self, other: Period) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Period):
            return NotImplemented
        try:
            return not self < other and not other < self
        except ValueError:
            return False

    def __hash__(self) -> int:
        p = self.normalized()
        return hash((p._length, p._units))

    def __str__(self) -> str:
        return f"{self._length}{_LETTER[self._units]}"

    def __repr__(self) -> str:
        return f"Period('{self}')"
