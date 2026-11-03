"""Concrete holiday calendars ported from QuantLib."""

from __future__ import annotations

from enum import Enum

from .calendar import Calendar, JointCalendarRule, WesternCalendar, western_easter_monday
from .date import Date, Month, Weekday


class NullCalendar(Calendar):
    """Every day is a business day; no weekends."""

    @property
    def name(self) -> str:
        return "Null"

    def _is_weekend(self, weekday: Weekday) -> bool:
        return False

    def _is_business_day(self, date: Date) -> bool:
        return True


class WeekendsOnly(WesternCalendar):
    """Only Saturdays and Sundays are holidays."""

    @property
    def name(self) -> str:
        return "weekends only"

    def _is_business_day(self, date: Date) -> bool:
        return not self.is_weekend(date.weekday)


class TARGET(WesternCalendar):
    """TARGET calendar used by the ECB for euro deposits."""

    @property
    def name(self) -> str:
        return "TARGET"

    def _is_business_day(self, date: Date) -> bool:
        w = date.weekday
        d = date.day
        dd = date.day_of_year
        m = date.month
        y = date.year
        em = western_easter_monday(y)
        if (
            self.is_weekend(w)
            or (d == 1 and m == Month.January)
            or (dd == em - 3 and y >= 2000)
            or (dd == em and y >= 2000)
            or (d == 1 and m == Month.May and y >= 2000)
            or (d == 25 and m == Month.December)
            or (d == 26 and m == Month.December and y >= 2000)
            or (d == 31 and m == Month.December and y in (1998, 1999, 2001))
        ):
            return False
        return True


class BespokeCalendar(Calendar):
    """Calendar with a custom name and weekend set; holidays via `add_holiday`."""

    def __init__(self, name: str = ""):
        super().__init__()
        self._name = name
        self._weekend_mask = 0

    @property
    def name(self) -> str:
        return self._name

    def add_weekend(self, weekday: Weekday) -> None:
        self._weekend_mask |= 1 << int(weekday)

    def _is_weekend(self, weekday: Weekday) -> bool:
        return (self._weekend_mask & (1 << int(weekday))) != 0

    def _is_business_day(self, date: Date) -> bool:
        return not self.is_weekend(date.weekday)


class JointCalendar(Calendar):
    """Combine two or more calendars with a join rule."""

    def __init__(
        self,
        *calendars: Calendar,
        rule: JointCalendarRule = JointCalendarRule.JoinHolidays,
    ):
        super().__init__()
        if len(calendars) < 2:
            raise ValueError("JointCalendar needs at least two calendars")
        self._calendars = list(calendars)
        self._rule = JointCalendarRule(rule)

    @property
    def name(self) -> str:
        label = (
            "JoinHolidays"
            if self._rule == JointCalendarRule.JoinHolidays
            else "JoinBusinessDays"
        )
        inner = ", ".join(c.name for c in self._calendars)
        return f"{label}({inner})"

    def _is_weekend(self, weekday: Weekday) -> bool:
        if self._rule == JointCalendarRule.JoinHolidays:
            return any(c.is_weekend(weekday) for c in self._calendars)
        return all(c.is_weekend(weekday) for c in self._calendars)

    def _is_business_day(self, date: Date) -> bool:
        if self._rule == JointCalendarRule.JoinHolidays:
            return all(c.is_business_day(date) for c in self._calendars)
        return any(c.is_business_day(date) for c in self._calendars)


# --- United States -----------------------------------------------------------------


def _is_washington_birthday(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y >= 1971:
        return (15 <= d <= 21) and w == Weekday.Monday and m == Month.February
    return (
        (d == 22 or (d == 23 and w == Weekday.Monday) or (d == 21 and w == Weekday.Friday))
        and m == Month.February
    )


def _is_memorial_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y >= 1971:
        return d >= 25 and w == Weekday.Monday and m == Month.May
    return (
        (d == 30 or (d == 31 and w == Weekday.Monday) or (d == 29 and w == Weekday.Friday))
        and m == Month.May
    )


def _is_labor_day(d: int, m: Month, w: Weekday) -> bool:
    return d <= 7 and w == Weekday.Monday and m == Month.September


def _is_columbus_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    return (8 <= d <= 14) and w == Weekday.Monday and m == Month.October and y >= 1971


def _is_veterans_day(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y <= 1970 or y >= 1978:
        return (
            (d == 11 or (d == 12 and w == Weekday.Monday) or (d == 10 and w == Weekday.Friday))
            and m == Month.November
        )
    return (22 <= d <= 28) and w == Weekday.Monday and m == Month.October


def _is_veterans_day_no_saturday(d: int, m: Month, y: int, w: Weekday) -> bool:
    if y <= 1970 or y >= 1978:
        return (d == 11 or (d == 12 and w == Weekday.Monday)) and m == Month.November
    return (22 <= d <= 28) and w == Weekday.Monday and m == Month.October


def _is_juneteenth(d: int, m: Month, y: int, w: Weekday, move_to_friday: bool = True) -> bool:
    return (
        (
            d == 19
            or (d == 20 and w == Weekday.Monday)
            or ((d == 18 and w == Weekday.Friday) and move_to_friday)
        )
        and m == Month.June
        and y >= 2022
    )


class UnitedStates(WesternCalendar):
    """United States calendars (settlement, NYSE, government bond, SOFR, …)."""

    class Market(Enum):
        Settlement = "settlement"
        LiborImpact = "libor impact"
        NYSE = "nyse"
        GovernmentBond = "government bond"
        NERC = "nerc"
        FederalReserve = "federal reserve"
        SOFR = "sofr"

    def __init__(self, market: Market = Market.Settlement):
        super().__init__()
        self._market = UnitedStates.Market(market)

    @property
    def name(self) -> str:
        return {
            self.Market.Settlement: "US settlement",
            self.Market.LiborImpact: "US with Libor impact",
            self.Market.NYSE: "New York stock exchange",
            self.Market.GovernmentBond: "US government bond market",
            self.Market.NERC: "North American Energy Reliability Council",
            self.Market.FederalReserve: "Federal Reserve Bankwire System",
            self.Market.SOFR: "SOFR fixing calendar",
        }[self._market]

    def _is_business_day(self, date: Date) -> bool:
        m = self._market
        if m == self.Market.Settlement:
            return self._settlement(date)
        if m == self.Market.LiborImpact:
            return self._libor_impact(date)
        if m == self.Market.NYSE:
            return self._nyse(date)
        if m == self.Market.GovernmentBond:
            return self._government_bond(date)
        if m == self.Market.SOFR:
            return self._sofr(date)
        if m == self.Market.NERC:
            return self._nerc(date)
        return self._federal_reserve(date)

    def _settlement(self, date: Date) -> bool:
        w, d, m, y = date.weekday, date.day, date.month, date.year
        if (
            self.is_weekend(w)
            or ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
            or (d == 31 and w == Weekday.Friday and m == Month.December)
            or ((15 <= d <= 21) and w == Weekday.Monday and m == Month.January and y >= 1983)
            or _is_washington_birthday(d, m, y, w)
            or _is_memorial_day(d, m, y, w)
            or _is_juneteenth(d, m, y, w)
            or (
                (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
                and m == Month.July
            )
            or _is_labor_day(d, m, w)
            or _is_columbus_day(d, m, y, w)
            or _is_veterans_day(d, m, y, w)
            or ((22 <= d <= 28) and w == Weekday.Thursday and m == Month.November)
            or (
                (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
                and m == Month.December
            )
        ):
            return False
        return True

    def _libor_impact(self, date: Date) -> bool:
        w, d, m, y = date.weekday, date.day, date.month, date.year
        if (
            ((d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
            and m == Month.July
            and y >= 2015
        ):
            return True
        return self._settlement(date)

    def _nyse(self, date: Date) -> bool:
        w, d, dd, m, y = date.weekday, date.day, date.day_of_year, date.month, date.year
        em = western_easter_monday(y)
        if (
            self.is_weekend(w)
            or ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
            or _is_washington_birthday(d, m, y, w)
            or (dd == em - 3)
            or _is_memorial_day(d, m, y, w)
            or _is_juneteenth(d, m, y, w)
            or (
                (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
                and m == Month.July
            )
            or _is_labor_day(d, m, w)
            or ((22 <= d <= 28) and w == Weekday.Thursday and m == Month.November)
            or (
                (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
                and m == Month.December
            )
        ):
            return False
        if y >= 1998 and (15 <= d <= 21) and w == Weekday.Monday and m == Month.January:
            return False
        if (y <= 1968 or (y <= 1980 and y % 4 == 0)) and m == Month.November and d <= 7 and w == Weekday.Tuesday:
            return False
        if (
            (y == 2025 and m == Month.January and d == 9)
            or (y == 2018 and m == Month.December and d == 5)
            or (y == 2012 and m == Month.October and d in (29, 30))
            or (y == 2007 and m == Month.January and d == 2)
            or (y == 2004 and m == Month.June and d == 11)
            or (y == 2001 and m == Month.September and 11 <= d <= 14)
            or (y == 1994 and m == Month.April and d == 27)
            or (y == 1985 and m == Month.September and d == 27)
            or (y == 1977 and m == Month.July and d == 14)
            or (y == 1973 and m == Month.January and d == 25)
            or (y == 1972 and m == Month.December and d == 28)
            or (y == 1969 and m == Month.July and d == 21)
            or (y == 1969 and m == Month.March and d == 31)
            or (y == 1969 and m == Month.February and d == 10)
            or (y == 1968 and m == Month.July and d == 5)
            or (y == 1968 and dd >= 163 and w == Weekday.Wednesday)
            or (y == 1968 and m == Month.April and d == 9)
            or (y == 1963 and m == Month.November and d == 25)
            or (y == 1961 and m == Month.May and d == 29)
            or (y == 1958 and m == Month.December and d == 26)
            or (y in (1954, 1956, 1965) and m == Month.December and d == 24)
        ):
            return False
        return True

    def _government_bond(self, date: Date) -> bool:
        w, d, dd, m, y = date.weekday, date.day, date.day_of_year, date.month, date.year
        em = western_easter_monday(y)
        if (
            self.is_weekend(w)
            or ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
            or ((15 <= d <= 21) and w == Weekday.Monday and m == Month.January and y >= 1983)
            or _is_washington_birthday(d, m, y, w)
            or (dd == em - 3 and (y < 1996 or d > 7))
            or _is_memorial_day(d, m, y, w)
            or _is_juneteenth(d, m, y, w)
            or (
                (d == 4 or (d == 5 and w == Weekday.Monday) or (d == 3 and w == Weekday.Friday))
                and m == Month.July
            )
            or _is_labor_day(d, m, w)
            or _is_columbus_day(d, m, y, w)
            or _is_veterans_day_no_saturday(d, m, y, w)
            or ((22 <= d <= 28) and w == Weekday.Thursday and m == Month.November)
            or (
                (d == 25 or (d == 26 and w == Weekday.Monday) or (d == 24 and w == Weekday.Friday))
                and m == Month.December
            )
        ):
            return False
        if (
            (y == 2018 and m == Month.December and d == 5)
            or (y == 2012 and m == Month.October and d == 30)
            or (y == 2004 and m == Month.June and d == 11)
        ):
            return False
        return True

    def _sofr(self, date: Date) -> bool:
        if date.day_of_year == western_easter_monday(date.year) - 3:
            return False
        return self._government_bond(date)

    def _nerc(self, date: Date) -> bool:
        w, d, m, y = date.weekday, date.day, date.month, date.year
        if (
            self.is_weekend(w)
            or ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
            or _is_memorial_day(d, m, y, w)
            or ((d == 4 or (d == 5 and w == Weekday.Monday)) and m == Month.July)
            or _is_labor_day(d, m, w)
            or ((22 <= d <= 28) and w == Weekday.Thursday and m == Month.November)
            or ((d == 25 or (d == 26 and w == Weekday.Monday)) and m == Month.December)
        ):
            return False
        return True

    def _federal_reserve(self, date: Date) -> bool:
        w, d, m, y = date.weekday, date.day, date.month, date.year
        if (
            self.is_weekend(w)
            or ((d == 1 or (d == 2 and w == Weekday.Monday)) and m == Month.January)
            or ((15 <= d <= 21) and w == Weekday.Monday and m == Month.January and y >= 1983)
            or _is_washington_birthday(d, m, y, w)
            or _is_memorial_day(d, m, y, w)
            or _is_juneteenth(d, m, y, w, move_to_friday=False)
            or ((d == 4 or (d == 5 and w == Weekday.Monday)) and m == Month.July)
            or _is_labor_day(d, m, w)
            or _is_columbus_day(d, m, y, w)
            or _is_veterans_day_no_saturday(d, m, y, w)
            or ((22 <= d <= 28) and w == Weekday.Thursday and m == Month.November)
            or ((d == 25 or (d == 26 and w == Weekday.Monday)) and m == Month.December)
        ):
            return False
        return True


# --- United Kingdom -----------------------------------------------------------------


def _is_uk_bank_holiday(d: int, w: Weekday, m: Month, y: int) -> bool:
    return (
        (d <= 7 and w == Weekday.Monday and m == Month.May and y not in (1995, 2020))
        or (d == 8 and m == Month.May and y in (1995, 2020))
        or (
            d >= 25
            and w == Weekday.Monday
            and m == Month.May
            and y not in (2002, 2012, 2022)
        )
        or ((d in (3, 4)) and m == Month.June and y == 2002)
        or ((d in (4, 5)) and m == Month.June and y == 2012)
        or ((d in (2, 3)) and m == Month.June and y == 2022)
        or (d >= 25 and w == Weekday.Monday and m == Month.August)
        or (d == 29 and m == Month.April and y == 2011)
        or (d == 19 and m == Month.September and y == 2022)
        or (d == 8 and m == Month.May and y == 2023)
    )


class UnitedKingdom(WesternCalendar):
    """United Kingdom calendars (settlement, LSE, metals)."""

    class Market(Enum):
        Settlement = "settlement"
        Exchange = "exchange"
        Metals = "metals"

    def __init__(self, market: Market = Market.Settlement):
        super().__init__()
        self._market = UnitedKingdom.Market(market)

    @property
    def name(self) -> str:
        return {
            self.Market.Settlement: "UK settlement",
            self.Market.Exchange: "London stock exchange",
            self.Market.Metals: "London metals exchange",
        }[self._market]

    def _is_business_day(self, date: Date) -> bool:
        # Settlement / Exchange / Metals share the same holiday rules in QuantLib 1.43.
        w, d, dd, m, y = date.weekday, date.day, date.day_of_year, date.month, date.year
        em = western_easter_monday(y)
        if (
            self.is_weekend(w)
            or (
                (d == 1 or ((d in (2, 3)) and w == Weekday.Monday))
                and m == Month.January
            )
            or (dd == em - 3)
            or (dd == em)
            or _is_uk_bank_holiday(d, w, m, y)
            or (
                (d == 25 or (d == 27 and w in (Weekday.Monday, Weekday.Tuesday)))
                and m == Month.December
            )
            or (
                (d == 26 or (d == 28 and w in (Weekday.Monday, Weekday.Tuesday)))
                and m == Month.December
            )
            or (d == 31 and m == Month.December and y == 1999)
        ):
            return False
        return True


# --- Japan --------------------------------------------------------------------------


class Japan(WesternCalendar):
    """Japan calendar."""

    @property
    def name(self) -> str:
        return "Japan"

    def _is_business_day(self, date: Date) -> bool:
        w, d, m, y = date.weekday, date.day, date.month, date.year
        # QuantLib uses C++ toward-zero integer division for the leap-year term.
        def _trunc_div(a: int, b: int) -> int:
            return int(a / b)

        exact_vernal = 20.69115
        exact_autumnal = 23.09
        diff_per_year = 0.242194
        moving = (y - 2000) * diff_per_year
        delta = y - 2000
        leaps = _trunc_div(delta, 4) + _trunc_div(delta, 100) - _trunc_div(delta, 400)
        ve = int(exact_vernal + moving - leaps)
        ae = int(exact_autumnal + moving - leaps)

        if (
            self.is_weekend(w)
            or (d == 1 and m == Month.January)
            or (d == 2 and m == Month.January)
            or (d == 3 and m == Month.January)
            or (w == Weekday.Monday and 8 <= d <= 14 and m == Month.January and y >= 2000)
            or ((d == 15 or (d == 16 and w == Weekday.Monday)) and m == Month.January and y < 2000)
            or ((d == 11 or (d == 12 and w == Weekday.Monday)) and m == Month.February)
            or ((d == 23 or (d == 24 and w == Weekday.Monday)) and m == Month.February and y >= 2020)
            or (
                (d == 23 or (d == 24 and w == Weekday.Monday))
                and m == Month.December
                and 1989 <= y < 2019
            )
            or ((d == ve or (d == ve + 1 and w == Weekday.Monday)) and m == Month.March)
            or ((d == 29 or (d == 30 and w == Weekday.Monday)) and m == Month.April)
            or (d == 3 and m == Month.May)
            or (d == 4 and m == Month.May)
            or (d == 5 and m == Month.May)
            or (
                d == 6
                and m == Month.May
                and w in (Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday)
            )
            or (
                w == Weekday.Monday
                and 15 <= d <= 21
                and m == Month.July
                and ((y >= 2003 and y < 2020) or y >= 2022)
            )
            or ((d == 20 or (d == 21 and w == Weekday.Monday)) and m == Month.July and 1996 <= y < 2003)
            or (d == 23 and m == Month.July and y == 2020)
            or (d == 22 and m == Month.July and y == 2021)
            or (
                (d == 11 or (d == 12 and w == Weekday.Monday))
                and m == Month.August
                and ((y >= 2016 and y < 2020) or y >= 2022)
            )
            or (d == 10 and m == Month.August and y == 2020)
            or (d == 9 and m == Month.August and y == 2021)
            or (w == Weekday.Monday and 15 <= d <= 21 and m == Month.September and y >= 2003)
            or ((d == 15 or (d == 16 and w == Weekday.Monday)) and m == Month.September and y < 2003)
            or (
                w == Weekday.Tuesday
                and d + 1 == ae
                and 16 <= d <= 22
                and m == Month.September
                and y >= 2003
            )
            or ((d == ae or (d == ae + 1 and w == Weekday.Monday)) and m == Month.September)
            or (
                w == Weekday.Monday
                and 8 <= d <= 14
                and m == Month.October
                and ((y >= 2000 and y < 2020) or y >= 2022)
            )
            or ((d == 10 or (d == 11 and w == Weekday.Monday)) and m == Month.October and y < 2000)
            or (d == 24 and m == Month.July and y == 2020)
            or (d == 23 and m == Month.July and y == 2021)
            or ((d == 3 or (d == 4 and w == Weekday.Monday)) and m == Month.November)
            or ((d == 23 or (d == 24 and w == Weekday.Monday)) and m == Month.November)
            or (d == 31 and m == Month.December)
            or (d == 10 and m == Month.April and y == 1959)
            or (d == 24 and m == Month.February and y == 1989)
            or (d == 12 and m == Month.November and y == 1990)
            or (d == 9 and m == Month.June and y == 1993)
            or (d == 30 and m == Month.April and y == 2019)
            or (d == 1 and m == Month.May and y == 2019)
            or (d == 2 and m == Month.May and y == 2019)
            or (d == 22 and m == Month.October and y == 2019)
        ):
            return False
        return True
