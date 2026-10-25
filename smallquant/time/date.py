"""Calendar dates with QuantLib-compatible serial numbers.

A `Date` is an immutable day count. Serial numbers follow QuantLib (and Excel): serial 367 is
1901-01-01 and 109574 is 2199-12-31, which is the supported range. Weekday numbering follows
QuantLib too: Sunday = 1 ... Saturday = 7.
"""

from __future__ import annotations

import datetime as _dt
from enum import IntEnum

from .period import Period, TimeUnit

MIN_SERIAL = 367  # 1901-01-01
MAX_SERIAL = 109574  # 2199-12-31

# Serial number of 1970-01-01 (Excel/QuantLib epoch is 1899-12-30).
_UNIX_EPOCH_SERIAL = 25569


class Month(IntEnum):
    January = 1
    February = 2
    March = 3
    April = 4
    May = 5
    June = 6
    July = 7
    August = 8
    September = 9
    October = 10
    November = 11
    December = 12


class Weekday(IntEnum):
    Sunday = 1
    Monday = 2
    Tuesday = 3
    Wednesday = 4
    Thursday = 5
    Friday = 6
    Saturday = 7


_MONTH_LENGTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Days since 1970-01-01 in the proleptic Gregorian calendar (H. Hinnant's algorithm)."""
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _civil_from_days(z: int) -> tuple[int, int, int]:
    z += 719468
    era = z // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return yoe + era * 400 + (m <= 2), m, d


class Date:
    """A calendar date in [1901-01-01, 2199-12-31].

    Arithmetic matches QuantLib: `date + n` and `date - n` move by days, `date - date` is the
    number of days between, and `date +/- Period` advances by the period, clamping the day to
    the end of the month when needed (31 Jan + 1M = 28/29 Feb).

    Example:
        >>> d = Date(31, Month.January, 2024)
        >>> d + Period(1, TimeUnit.Months)
        Date(2024-02-29)
        >>> Date(1, 3, 2024) - d
        30
    """

    __slots__ = ("_serial", "_year", "_month", "_day")

    def __init__(self, day: int, month: int, year: int):
        if not 1900 < year < 2200:
            raise ValueError(f"year {year} out of bound. It must be in [1901,2199]")
        if not 1 <= month <= 12:
            raise ValueError(f"month {int(month)} outside January-December range [1,12]")
        length = Date.month_length(month, Date.is_leap(year))
        if not 1 <= day <= length:
            raise ValueError(f"day outside month ({int(month)}) day-range [1,{length}]")
        self._set(_days_from_civil(year, month, day) + _UNIX_EPOCH_SERIAL, year, month, day)

    def _set(self, serial: int, year: int, month: int, day: int) -> None:
        object.__setattr__(self, "_serial", serial)
        object.__setattr__(self, "_year", year)
        object.__setattr__(self, "_month", Month(month))
        object.__setattr__(self, "_day", day)

    def __setattr__(self, name, value):
        raise AttributeError("Date is immutable")

    @classmethod
    def from_serial(cls, serial: int) -> Date:
        serial = int(serial)
        if not MIN_SERIAL <= serial <= MAX_SERIAL:
            raise ValueError(
                f"Date's serial number ({serial}) outside allowed range "
                f"[{MIN_SERIAL}-{MAX_SERIAL}], i.e. [{cls.min_date()}-{cls.max_date()}]"
            )
        y, m, d = _civil_from_days(serial - _UNIX_EPOCH_SERIAL)
        obj = object.__new__(cls)
        obj._set(serial, y, m, d)
        return obj

    @classmethod
    def from_date(cls, d: _dt.date) -> Date:
        """Convert a `datetime.date` (or the date part of a `datetime`)."""
        return cls(d.day, d.month, d.year)

    @classmethod
    def from_iso(cls, text: str) -> Date:
        return cls.from_date(_dt.date.fromisoformat(text))

    @classmethod
    def today(cls) -> Date:
        return cls.from_date(_dt.date.today())

    @classmethod
    def min_date(cls) -> Date:
        return cls.from_serial(MIN_SERIAL)

    @classmethod
    def max_date(cls) -> Date:
        return cls.from_serial(MAX_SERIAL)

    @property
    def serial(self) -> int:
        return self._serial

    @property
    def year(self) -> int:
        return self._year

    @property
    def month(self) -> Month:
        return self._month

    @property
    def day(self) -> int:
        return self._day

    @property
    def weekday(self) -> Weekday:
        w = self._serial % 7
        return Weekday.Saturday if w == 0 else Weekday(w)

    @property
    def day_of_year(self) -> int:
        return self._serial - (_days_from_civil(self._year - 1, 12, 31) + _UNIX_EPOCH_SERIAL)

    def to_date(self) -> _dt.date:
        return _dt.date(self._year, int(self._month), self._day)

    def isoformat(self) -> str:
        return f"{self._year:04d}-{int(self._month):02d}-{self._day:02d}"

    @staticmethod
    def is_leap(year: int) -> bool:
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    @staticmethod
    def month_length(month: int, leap_year: bool) -> int:
        if month == 2 and leap_year:
            return 29
        return _MONTH_LENGTH[int(month) - 1]

    @staticmethod
    def end_of_month(d: Date) -> Date:
        return Date(Date.month_length(d.month, Date.is_leap(d.year)), d.month, d.year)

    @staticmethod
    def is_end_of_month(d: Date) -> bool:
        return d.day == Date.month_length(d.month, Date.is_leap(d.year))

    @staticmethod
    def next_weekday(d: Date, weekday: Weekday) -> Date:
        """First date on or after `d` that falls on `weekday`."""
        wd = d.weekday
        return d + ((7 if wd > weekday else 0) - wd + weekday)

    @staticmethod
    def nth_weekday(nth: int, weekday: Weekday, month: int, year: int) -> Date:
        """The `nth` (1-5) given weekday of a month, e.g. the 3rd Wednesday of March 2025."""
        if nth <= 0:
            raise ValueError("zeroth day of week in a given (month, year) is undefined")
        if nth >= 6:
            raise ValueError("no more than 5 weekday in a given (month, year)")
        first = Date(1, month, year).weekday
        skip = nth - (1 if weekday >= first else 0)
        return Date((1 + weekday + skip * 7) - first, month, year)

    @staticmethod
    def advance(date: Date, n: int, units: TimeUnit) -> Date:
        """`date` moved by `n` units; month/year steps clamp to the end of the month."""
        if units == TimeUnit.Days:
            return date + n
        if units == TimeUnit.Weeks:
            return date + 7 * n
        if units == TimeUnit.Months:
            m = int(date.month) + n
            y = date.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            if not 1900 <= y <= 2199:
                raise ValueError(f"year {y} out of bounds. It must be in [1901,2199]")
            return Date(min(date.day, Date.month_length(m, Date.is_leap(y))), m, y)
        if units == TimeUnit.Years:
            y = date.year + n
            if not 1900 <= y <= 2199:
                raise ValueError(f"year {y} out of bounds. It must be in [1901,2199]")
            d = date.day
            if d == 29 and date.month == Month.February and not Date.is_leap(y):
                d = 28
            return Date(d, date.month, y)
        raise ValueError("undefined time units")

    def __add__(self, other):
        if isinstance(other, Period):
            return Date.advance(self, other.length, other.units)
        if isinstance(other, int) and not isinstance(other, bool):
            return Date.from_serial(self._serial + other)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Date):
            return self._serial - other._serial
        if isinstance(other, Period):
            return Date.advance(self, -other.length, other.units)
        if isinstance(other, int) and not isinstance(other, bool):
            return Date.from_serial(self._serial - other)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Date):
            return NotImplemented
        return self._serial == other._serial

    def __lt__(self, other: Date) -> bool:
        if not isinstance(other, Date):
            return NotImplemented
        return self._serial < other._serial

    def __le__(self, other: Date) -> bool:
        if not isinstance(other, Date):
            return NotImplemented
        return self._serial <= other._serial

    def __gt__(self, other: Date) -> bool:
        if not isinstance(other, Date):
            return NotImplemented
        return self._serial > other._serial

    def __ge__(self, other: Date) -> bool:
        if not isinstance(other, Date):
            return NotImplemented
        return self._serial >= other._serial

    def __hash__(self) -> int:
        return hash(self._serial)

    def __str__(self) -> str:
        return self.isoformat()

    def __repr__(self) -> str:
        return f"Date({self.isoformat()})"
