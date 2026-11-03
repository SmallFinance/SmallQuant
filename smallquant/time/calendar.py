"""Holiday calendars and business-day conventions (QuantLib `Calendar`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum

from .date import Date, Weekday
from .period import Period, TimeUnit


class BusinessDayConvention(IntEnum):
    """How to adjust a non-business day onto a nearby business day."""

    Following = 0
    ModifiedFollowing = 1
    Preceding = 2
    ModifiedPreceding = 3
    Unadjusted = 4
    HalfMonthModifiedFollowing = 5
    Nearest = 6


class JointCalendarRule(IntEnum):
    """How `JointCalendar` combines holidays from several calendars."""

    JoinHolidays = 0
    JoinBusinessDays = 1


# Western Easter Monday as day-of-year, QuantLib table for 1901-2199.
_WESTERN_EASTER_MONDAY = (
    98, 90, 103, 95, 114, 106, 91, 111, 102,  # 1901-1909
    87, 107, 99, 83, 103, 95, 115, 99, 91, 111,  # 1910-1919
    96, 87, 107, 92, 112, 103, 95, 108, 100, 91,  # 1920-1929
    111, 96, 88, 107, 92, 112, 104, 88, 108, 100,  # 1930-1939
    85, 104, 96, 116, 101, 92, 112, 97, 89, 108,  # 1940-1949
    100, 85, 105, 96, 109, 101, 93, 112, 97, 89,  # 1950-1959
    109, 93, 113, 105, 90, 109, 101, 86, 106, 97,  # 1960-1969
    89, 102, 94, 113, 105, 90, 110, 101, 86, 106,  # 1970-1979
    98, 110, 102, 94, 114, 98, 90, 110, 95, 86,  # 1980-1989
    106, 91, 111, 102, 94, 107, 99, 90, 103, 95,  # 1990-1999
    115, 106, 91, 111, 103, 87, 107, 99, 84, 103,  # 2000-2009
    95, 115, 100, 91, 111, 96, 88, 107, 92, 112,  # 2010-2019
    104, 95, 108, 100, 92, 111, 96, 88, 108, 92,  # 2020-2029
    112, 104, 89, 108, 100, 85, 105, 96, 116, 101,  # 2030-2039
    93, 112, 97, 89, 109, 100, 85, 105, 97, 109,  # 2040-2049
    101, 93, 113, 97, 89, 109, 94, 113, 105, 90,  # 2050-2059
    110, 101, 86, 106, 98, 89, 102, 94, 114, 105,  # 2060-2069
    90, 110, 102, 86, 106, 98, 111, 102, 94, 114,  # 2070-2079
    99, 90, 110, 95, 87, 106, 91, 111, 103, 94,  # 2080-2089
    107, 99, 91, 103, 95, 115, 107, 91, 111, 103,  # 2090-2099
    88, 108, 100, 85, 105, 96, 109, 101, 93, 112,  # 2100-2109
    97, 89, 109, 93, 113, 105, 90, 109, 101, 86,  # 2110-2119
    106, 97, 89, 102, 94, 113, 105, 90, 110, 101,  # 2120-2129
    86, 106, 98, 110, 102, 94, 114, 98, 90, 110,  # 2130-2139
    95, 86, 106, 91, 111, 102, 94, 107, 99, 90,  # 2140-2149
    103, 95, 115, 106, 91, 111, 103, 87, 107, 99,  # 2150-2159
    84, 103, 95, 115, 100, 91, 111, 96, 88, 107,  # 2160-2169
    92, 112, 104, 95, 108, 100, 92, 111, 96, 88,  # 2170-2179
    108, 92, 112, 104, 89, 108, 100, 85, 105, 96,  # 2180-2189
    116, 101, 93, 112, 97, 89, 109, 100, 85, 105,  # 2190-2199
)


def western_easter_monday(year: int) -> int:
    """Day-of-year of Western Easter Monday for `year` in [1901, 2199]."""
    if not 1901 <= year <= 2199:
        raise ValueError(f"year {year} out of Easter Monday table range [1901, 2199]")
    return _WESTERN_EASTER_MONDAY[year - 1901]


class Calendar(ABC):
    """Market holiday calendar.

    Unlike QuantLib, added/removed holidays are per instance (not shared across all
    calendars of the same type). Weekend and fixed-holiday rules still match QuantLib.
    """

    def __init__(self) -> None:
        self._added: set[Date] = set()
        self._removed: set[Date] = set()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _is_weekend(self, weekday: Weekday) -> bool: ...

    @abstractmethod
    def _is_business_day(self, date: Date) -> bool:
        """Underlying rule, ignoring `add_holiday` / `remove_holiday` overrides."""

    def is_weekend(self, weekday: Weekday) -> bool:
        return self._is_weekend(weekday)

    def is_business_day(self, date: Date) -> bool:
        if date in self._added:
            return False
        if date in self._removed:
            return True
        return self._is_business_day(date)

    def is_holiday(self, date: Date) -> bool:
        return not self.is_business_day(date)

    def add_holiday(self, date: Date) -> None:
        self._removed.discard(date)
        if self._is_business_day(date):
            self._added.add(date)

    def remove_holiday(self, date: Date) -> None:
        self._added.discard(date)
        if not self._is_business_day(date):
            self._removed.add(date)

    def reset_added_and_removed_holidays(self) -> None:
        self._added.clear()
        self._removed.clear()

    @property
    def added_holidays(self) -> frozenset[Date]:
        return frozenset(self._added)

    @property
    def removed_holidays(self) -> frozenset[Date]:
        return frozenset(self._removed)

    def start_of_month(self, date: Date) -> Date:
        return self.adjust(Date.start_of_month(date), BusinessDayConvention.Following)

    def end_of_month(self, date: Date) -> Date:
        return self.adjust(Date.end_of_month(date), BusinessDayConvention.Preceding)

    def is_start_of_month(self, date: Date) -> bool:
        return date <= self.start_of_month(date)

    def is_end_of_month(self, date: Date) -> bool:
        return date >= self.end_of_month(date)

    def holiday_list(
        self, from_date: Date, to_date: Date, include_weekends: bool = False
    ) -> list[Date]:
        if from_date > to_date:
            return []
        out: list[Date] = []
        d = from_date
        while d <= to_date:
            if self.is_holiday(d) and (include_weekends or not self.is_weekend(d.weekday)):
                out.append(d)
            d += 1
        return out

    def business_day_list(self, from_date: Date, to_date: Date) -> list[Date]:
        if from_date > to_date:
            return []
        out: list[Date] = []
        d = from_date
        while d <= to_date:
            if self.is_business_day(d):
                out.append(d)
            d += 1
        return out

    def adjust(
        self,
        date: Date,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
    ) -> Date:
        c = BusinessDayConvention(convention)
        if c == BusinessDayConvention.Unadjusted:
            return date

        d1 = date
        if c in (
            BusinessDayConvention.Following,
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.HalfMonthModifiedFollowing,
        ):
            while self.is_holiday(d1):
                d1 += 1
            if c in (
                BusinessDayConvention.ModifiedFollowing,
                BusinessDayConvention.HalfMonthModifiedFollowing,
            ):
                if d1.month != date.month:
                    return self.adjust(date, BusinessDayConvention.Preceding)
                if (
                    c == BusinessDayConvention.HalfMonthModifiedFollowing
                    and date.day <= 15
                    and d1.day > 15
                ):
                    return self.adjust(date, BusinessDayConvention.Preceding)
            return d1

        if c in (BusinessDayConvention.Preceding, BusinessDayConvention.ModifiedPreceding):
            while self.is_holiday(d1):
                d1 -= 1
            if c == BusinessDayConvention.ModifiedPreceding and d1.month != date.month:
                return self.adjust(date, BusinessDayConvention.Following)
            return d1

        if c == BusinessDayConvention.Nearest:
            d2 = date
            while self.is_holiday(d1) and self.is_holiday(d2):
                d1 += 1
                d2 -= 1
            return d2 if self.is_holiday(d1) else d1

        raise ValueError(f"unknown business-day convention ({c})")

    def advance(
        self,
        date: Date,
        n: int | Period,
        unit: TimeUnit | None = None,
        *,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
        end_of_month: bool = False,
    ) -> Date:
        """Advance `date` by a period or by `n` `unit`s, then adjust.

        Pass either a `Period` (`advance(date, Period(1, Months), convention=...)`) or an
        integer count with a unit (`advance(date, 2, TimeUnit.Days, convention=...)`).
        """
        if isinstance(n, Period):
            if unit is not None:
                raise TypeError("unit must not be passed when n is a Period")
            length, units = n.length, n.units
        else:
            if unit is None:
                raise TypeError("unit is required when advancing by an integer")
            length, units = int(n), TimeUnit(unit)

        c = BusinessDayConvention(convention)
        if length == 0:
            return self.adjust(date, c)

        if units == TimeUnit.Days:
            d1 = date
            steps = length
            if steps > 0:
                while steps > 0:
                    d1 += 1
                    while self.is_holiday(d1):
                        d1 += 1
                    steps -= 1
            else:
                while steps < 0:
                    d1 -= 1
                    while self.is_holiday(d1):
                        d1 -= 1
                    steps += 1
            return d1

        if units == TimeUnit.Weeks:
            return self.adjust(date + Period(length, TimeUnit.Weeks), c)

        d1 = date + Period(length, units)
        if end_of_month:
            if c == BusinessDayConvention.Unadjusted:
                if Date.is_end_of_month(date):
                    return Date.end_of_month(d1)
            elif self.is_end_of_month(date):
                return self.end_of_month(d1)
        return self.adjust(d1, c)

    def business_days_between(
        self,
        from_date: Date,
        to_date: Date,
        include_first: bool = True,
        include_last: bool = False,
    ) -> int:
        if from_date < to_date:
            return self._days_between(from_date, to_date, include_first, include_last)
        if from_date > to_date:
            return -self._days_between(to_date, from_date, include_last, include_first)
        return int(include_first and include_last and self.is_business_day(from_date))

    def _days_between(
        self, from_date: Date, to_date: Date, include_first: bool, include_last: bool
    ) -> int:
        res = int(include_last and self.is_business_day(to_date))
        d = from_date if include_first else from_date + 1
        while d < to_date:
            res += int(self.is_business_day(d))
            d += 1
        return res

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Calendar):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class WesternCalendar(Calendar):
    """Saturday/Sunday weekends and Western Easter Monday helper."""

    def _is_weekend(self, weekday: Weekday) -> bool:
        return weekday in (Weekday.Saturday, Weekday.Sunday)

    @staticmethod
    def easter_monday(year: int) -> int:
        return western_easter_monday(year)
