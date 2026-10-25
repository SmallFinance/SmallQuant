"""Day count conventions (QuantLib `DayCounter` and its concrete conventions).

Each counter provides `day_count(d1, d2)` (integer days under the convention) and
`year_fraction(d1, d2, ref_start=None, ref_end=None)`. The reference period is only used by
conventions that need a coupon period (Actual/Actual ISMA, Actual/365 Canadian).

Not ported: Business/252 (needs calendars) and Actual/Actual ISMA driven by a `Schedule`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

from .date import Date, Month
from .period import Period, TimeUnit


def _lround(x: float) -> int:
    """C++ `std::lround`: round half away from zero."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


class DayCounter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    def day_count(self, d1: Date, d2: Date) -> int:
        return d2 - d1

    @abstractmethod
    def year_fraction(
        self, d1: Date, d2: Date, ref_start: Date | None = None, ref_end: Date | None = None
    ) -> float: ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DayCounter):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class _ActualFixed(DayCounter):
    """Actual days over a fixed denominator, optionally counting the last day."""

    _label = ""
    _denominator = 0.0

    def __init__(self, include_last_day: bool = False):
        self._inc = include_last_day

    @property
    def name(self) -> str:
        return f"{self._label} (inc)" if self._inc else self._label

    def day_count(self, d1: Date, d2: Date) -> int:
        return (d2 - d1) + (1 if self._inc else 0)

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        return ((d2 - d1) + (1.0 if self._inc else 0.0)) / self._denominator


class Actual360(_ActualFixed):
    _label, _denominator = "Actual/360", 360.0


class Actual366(_ActualFixed):
    _label, _denominator = "Actual/366", 366.0


class Actual36525(_ActualFixed):
    _label, _denominator = "Actual/365.25", 365.25


class Actual365Fixed(DayCounter):
    """Actual/365 (Fixed), with the Canadian bond and no-leap variants."""

    class Convention(Enum):
        Standard = 0
        Canadian = 1
        NoLeap = 2

    _MONTH_OFFSET = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

    def __init__(self, convention: Convention = Convention.Standard):
        self._c = convention

    @property
    def name(self) -> str:
        return {
            self.Convention.Standard: "Actual/365 (Fixed)",
            self.Convention.Canadian: "Actual/365 (Fixed) Canadian Bond",
            self.Convention.NoLeap: "Actual/365 (No Leap)",
        }[self._c]

    def day_count(self, d1: Date, d2: Date) -> int:
        if self._c != self.Convention.NoLeap:
            return d2 - d1
        off = self._MONTH_OFFSET
        s1 = d1.day + off[d1.month - 1] + d1.year * 365
        s2 = d2.day + off[d2.month - 1] + d2.year * 365
        if d1.month == Month.February and d1.day == 29:
            s1 -= 1
        if d2.month == Month.February and d2.day == 29:
            s2 -= 1
        return s2 - s1

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        if self._c == self.Convention.Canadian:
            if d1 == d2:
                return 0.0
            if ref_start is None:
                raise ValueError("invalid refPeriodStart")
            if ref_end is None:
                raise ValueError("invalid refPeriodEnd")
            dcs = float(d2 - d1)
            dcc = float(ref_end - ref_start)
            months = _lround(12 * dcc / 365)
            if months == 0:
                raise ValueError(
                    "invalid reference period for Act/365 Canadian; must be longer than a month"
                )
            frequency = int(12 / months)
            if dcs < int(365 / frequency):
                return dcs / 365.0
            return 1.0 / frequency - (dcc - dcs) / 365.0
        return self.day_count(d1, d2) / 365.0


class Thirty360(DayCounter):
    """30/360 family. `ISDA`/`German` need the deal's termination date for the February rule."""

    class Convention(Enum):
        USA = 0
        BondBasis = 1
        European = 2
        EurobondBasis = 3
        Italian = 4
        German = 5
        ISMA = 6
        ISDA = 7
        NASD = 8

    def __init__(self, convention: Convention, termination_date: Date | None = None):
        C = self.Convention
        self._c = convention
        self._term = termination_date
        self._kind = {
            C.USA: "US",
            C.European: "EU",
            C.EurobondBasis: "EU",
            C.Italian: "IT",
            C.ISMA: "ISMA",
            C.BondBasis: "ISMA",
            C.ISDA: "ISDA",
            C.German: "ISDA",
            C.NASD: "NASD",
        }[convention]

    @property
    def name(self) -> str:
        return {
            "US": "30/360 (US)",
            "ISMA": "30/360 (Bond Basis)",
            "EU": "30E/360 (Eurobond Basis)",
            "IT": "30/360 (Italian)",
            "ISDA": "30E/360 (ISDA)",
            "NASD": "30/360 (NASD)",
        }[self._kind]

    @staticmethod
    def _last_of_feb(d: int, m: int, y: int) -> bool:
        return m == 2 and d == 28 + (1 if Date.is_leap(y) else 0)

    def day_count(self, d1: Date, d2: Date) -> int:
        dd1, dd2 = d1.day, d2.day
        mm1, mm2 = int(d1.month), int(d2.month)
        yy1, yy2 = d1.year, d2.year
        kind = self._kind
        last = self._last_of_feb

        if kind == "US":
            # the order of checks matters
            if last(dd1, mm1, yy1):
                if last(dd2, mm2, yy2):
                    dd2 = 30
                dd1 = 30
            if dd2 == 31 and dd1 >= 30:
                dd2 = 30
            if dd1 == 31:
                dd1 = 30
        elif kind == "ISMA":
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31 and dd1 == 30:
                dd2 = 30
        elif kind == "EU":
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31:
                dd2 = 30
        elif kind == "IT":
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31:
                dd2 = 30
            if mm1 == 2 and dd1 > 27:
                dd1 = 30
            if mm2 == 2 and dd2 > 27:
                dd2 = 30
        elif kind == "ISDA":
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31:
                dd2 = 30
            if last(dd1, mm1, yy1):
                dd1 = 30
            if d2 != self._term and last(dd2, mm2, yy2):
                dd2 = 30
        else:  # NASD
            if dd1 == 31:
                dd1 = 30
            if dd2 == 31 and dd1 >= 30:
                dd2 = 30
            if dd2 == 31 and dd1 < 30:
                dd2 = 1
                mm2 += 1

        return 360 * (yy2 - yy1) + 30 * (mm2 - mm1) + (dd2 - dd1)

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        return self.day_count(d1, d2) / 360.0


class ActualActual(DayCounter):
    """Actual/Actual. ISMA/Bond here is the reference-period form (no `Schedule` support)."""

    class Convention(Enum):
        ISMA = 0
        Bond = 1
        ISDA = 2
        Historical = 3
        Actual365 = 4
        AFB = 5
        Euro = 6

    def __init__(self, convention: Convention = Convention.ISDA):
        C = self.Convention
        self._c = convention
        self._kind = {
            C.ISMA: "ISMA",
            C.Bond: "ISMA",
            C.ISDA: "ISDA",
            C.Historical: "ISDA",
            C.Actual365: "ISDA",
            C.AFB: "AFB",
            C.Euro: "AFB",
        }[convention]

    @property
    def name(self) -> str:
        return f"Actual/Actual ({self._kind})"

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        if self._kind == "ISDA":
            return self._isda(d1, d2)
        if self._kind == "AFB":
            return self._afb(d1, d2)
        return self._isma(d1, d2, ref_start, ref_end)

    @staticmethod
    def _isda(d1: Date, d2: Date) -> float:
        if d1 == d2:
            return 0.0
        if d1 > d2:
            return -ActualActual._isda(d2, d1)
        y1, y2 = d1.year, d2.year
        dib1 = 366.0 if Date.is_leap(y1) else 365.0
        dib2 = 366.0 if Date.is_leap(y2) else 365.0
        total = float(y2 - y1 - 1)
        total += (Date(1, Month.January, y1 + 1) - d1) / dib1
        total += (d2 - Date(1, Month.January, y2)) / dib2
        return total

    @staticmethod
    def _afb(d1: Date, d2: Date) -> float:
        if d1 == d2:
            return 0.0
        if d1 > d2:
            return -ActualActual._afb(d2, d1)
        new_d2 = temp = d2
        total = 0.0
        while temp > d1:
            temp = new_d2 - Period(1, TimeUnit.Years)
            if temp.day == 28 and temp.month == Month.February and Date.is_leap(temp.year):
                temp = temp + 1
            if temp >= d1:
                total += 1.0
                new_d2 = temp
        den = 365.0
        if Date.is_leap(new_d2.year):
            temp = Date(29, Month.February, new_d2.year)
            if new_d2 > temp and d1 <= temp:
                den += 1.0
        elif Date.is_leap(d1.year):
            temp = Date(29, Month.February, d1.year)
            if new_d2 > temp and d1 <= temp:
                den += 1.0
        return total + (new_d2 - d1) / den

    def _isma(self, d1: Date, d2: Date, d3: Date | None, d4: Date | None) -> float:
        if d1 == d2:
            return 0.0
        if d1 > d2:
            return -self._isma(d2, d1, d3, d4)

        # when the reference period is not specified, take it equal to (d1, d2)
        ref_start = d3 if d3 is not None else d1
        ref_end = d4 if d4 is not None else d2
        if not (ref_end > ref_start and ref_end > d1):
            raise ValueError(
                f"invalid reference period: date 1: {d1}, date 2: {d2}, "
                f"reference period start: {ref_start}, reference period end: {ref_end}"
            )

        months = _lround(12 * (ref_end - ref_start) / 365)
        if months == 0:  # short period: take the reference period as 1 year from d1
            ref_start = d1
            ref_end = d1 + Period(1, TimeUnit.Years)
            months = 12
        period = months / 12.0

        if d2 <= ref_end:
            if d1 >= ref_start:
                return period * (d2 - d1) / (ref_end - ref_start)
            # long first coupon: d1 < ref_start < ref_end and d2 <= ref_end
            previous_ref = ref_start - Period(months, TimeUnit.Months)
            if d2 > ref_start:
                return self._isma(d1, ref_start, previous_ref, ref_start) + self._isma(
                    ref_start, d2, ref_start, ref_end
                )
            return self._isma(d1, d2, previous_ref, ref_start)

        # ref_start <= d1 < ref_end < d2
        if not ref_start <= d1:
            raise ValueError("invalid dates: d1 < refPeriodStart < refPeriodEnd < d2")
        total = self._isma(d1, ref_end, ref_start, ref_end)
        i = 0
        while True:
            new_start = ref_end + Period(months * i, TimeUnit.Months)
            new_end = ref_end + Period(months * (i + 1), TimeUnit.Months)
            if d2 < new_end:
                break
            total += period
            i += 1
        return total + self._isma(new_start, d2, new_start, new_end)


class SimpleDayCounter(DayCounter):
    """Whole-month fractions where the days line up, otherwise 30/360 Bond Basis."""

    _fallback = Thirty360(Thirty360.Convention.BondBasis)

    @property
    def name(self) -> str:
        return "Simple"

    def day_count(self, d1: Date, d2: Date) -> int:
        return self._fallback.day_count(d1, d2)

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        dm1, dm2 = d1.day, d2.day
        if (
            dm1 == dm2
            or (dm1 > dm2 and Date.is_end_of_month(d2))
            or (dm1 < dm2 and Date.is_end_of_month(d1))
        ):
            return (d2.year - d1.year) + (int(d2.month) - int(d1.month)) / 12.0
        return self._fallback.year_fraction(d1, d2)


class OneDayCounter(DayCounter):
    """1/1: year fraction is +1 or -1 by the order of the dates."""

    @property
    def name(self) -> str:
        return "1/1"

    def day_count(self, d1: Date, d2: Date) -> int:
        return 1 if d2 >= d1 else -1

    def year_fraction(self, d1, d2, ref_start=None, ref_end=None) -> float:
        return float(self.day_count(d1, d2))
