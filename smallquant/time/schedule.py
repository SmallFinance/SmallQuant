"""Date schedules. Regular forward and backward generation, plus a few IMM-style rules."""

from __future__ import annotations

from enum import Enum

from .calendar import BusinessDayConvention, Calendar
from .date import Date, Month, Weekday
from .period import Period, TimeUnit


class DateGeneration(Enum):
    Backward = "backward"
    Forward = "forward"
    Zero = "zero"
    ThirdWednesday = "third_wednesday"
    Twentieth = "twentieth"
    TwentiethIMM = "twentieth_imm"
    CDS = "cds"


class Schedule:
    """Payment schedule.

    ``end_of_month`` keeps dates on month-ends when the effective date is a
    month-end. Regular periods are marked ``is_regular``. The dates are the
    adjusted schedule, including the effective date and the termination date.
    """

    def __init__(
        self,
        effective: Date,
        termination: Date,
        tenor: Period,
        calendar: Calendar,
        convention: BusinessDayConvention = BusinessDayConvention.Following,
        termination_convention: BusinessDayConvention = BusinessDayConvention.Following,
        rule: DateGeneration = DateGeneration.Backward,
        end_of_month: bool = False,
    ):
        if termination <= effective:
            raise ValueError("termination must be after the effective date")
        self.effective = effective
        self.termination = termination
        self.tenor = tenor
        self.calendar = calendar
        self.convention = BusinessDayConvention(convention)
        self.termination_convention = BusinessDayConvention(termination_convention)
        self.rule = DateGeneration(rule)
        self.end_of_month = end_of_month and _is_end(effective, calendar, self.convention)
        self._dates = _generate(self)
        self._regular = [True] * max(0, len(self._dates) - 1)

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, index: int) -> Date:
        return self._dates[index]

    @property
    def dates(self) -> tuple[Date, ...]:
        return tuple(self._dates)

    def is_regular(self, index: int) -> bool:
        return self._regular[index]

    def reference_period(self, date: Date) -> tuple[Date, Date]:
        """Coupon period of ``date`` for Actual/Actual ISMA."""
        dates = self.quasi_payment_dates()
        for start, end in zip(dates, dates[1:]):
            if start <= date < end or (date == dates[-1] and end == dates[-1]):
                return start, end
        raise ValueError(f"{date} is outside the schedule")

    def quasi_payment_dates(self) -> list[Date]:
        """Schedule dates, extended by a notional coupon when a stub is irregular."""
        dates = list(self._dates)
        if len(dates) < 2:
            return dates
        if not self.is_regular(1) if len(self._regular) > 1 else False:
            notional = self.calendar.advance(
                dates[1], -self.tenor, convention=self.convention, end_of_month=self.end_of_month
            )
            dates[0] = notional
        return dates


def _is_end(date: Date, calendar: Calendar, convention: BusinessDayConvention) -> bool:
    return calendar.is_end_of_month(date) or Date.is_end_of_month(date)


def _generate(schedule: Schedule) -> list[Date]:
    rule = schedule.rule
    if rule == DateGeneration.Zero:
        return [
            schedule.calendar.adjust(schedule.effective, schedule.convention),
            schedule.calendar.adjust(schedule.termination, schedule.termination_convention),
        ]
    if rule == DateGeneration.Backward:
        return _backward(schedule)
    if rule == DateGeneration.Forward:
        return _forward(schedule)
    if rule in (DateGeneration.ThirdWednesday, DateGeneration.Twentieth, DateGeneration.TwentiethIMM, DateGeneration.CDS):
        return _anchored(schedule)
    raise ValueError(f"unsupported date-generation rule {rule}")


def _backward(schedule: Schedule) -> list[Date]:
    raw = [schedule.termination]
    cursor = schedule.termination
    while True:
        cursor = cursor - schedule.tenor
        if schedule.end_of_month:
            cursor = Date.end_of_month(cursor)
        if cursor <= schedule.effective:
            break
        raw.append(cursor)
    raw.append(schedule.effective)
    raw.reverse()
    adjusted = [_adjust(schedule, date, last=index == len(raw) - 1) for index, date in enumerate(raw)]
    return _unique(adjusted)


def _forward(schedule: Schedule) -> list[Date]:
    raw = [schedule.effective]
    cursor = schedule.effective
    while True:
        cursor = cursor + schedule.tenor
        if schedule.end_of_month:
            cursor = Date.end_of_month(cursor)
        if cursor >= schedule.termination:
            break
        raw.append(cursor)
    raw.append(schedule.termination)
    adjusted = [_adjust(schedule, date, last=index == len(raw) - 1) for index, date in enumerate(raw)]
    return _unique(adjusted)


def _anchored(schedule: Schedule) -> list[Date]:
    dates = [schedule.calendar.adjust(schedule.effective, schedule.convention)]
    cursor = schedule.effective
    while cursor < schedule.termination:
        cursor = _next_anchor(cursor, schedule.rule)
        if cursor >= schedule.termination:
            break
        dates.append(schedule.calendar.adjust(cursor, schedule.convention))
    dates.append(schedule.calendar.adjust(schedule.termination, schedule.termination_convention))
    return _unique(dates)


def _next_anchor(date: Date, rule: DateGeneration) -> Date:
    if rule == DateGeneration.ThirdWednesday:
        month = date.month
        year = date.year
        cursor = Date(1, month, year)
        while cursor.weekday != Weekday.Wednesday or cursor.day <= 14:
            cursor = cursor + 1
            if cursor.month != month:
                month = cursor.month
                year = cursor.year
        if cursor <= date:
            # next month
            nxt = Date(1, month, year) + Period(1, TimeUnit.Months)
            return _next_anchor(nxt - 1, rule)
        return cursor
    if rule == DateGeneration.Twentieth:
        twentieth = Date(20, date.month, date.year)
        if twentieth <= date:
            twentieth = Date(20, date.month, date.year) + Period(1, TimeUnit.Months)
        return twentieth
    if rule in (DateGeneration.TwentiethIMM, DateGeneration.CDS):
        month = int(date.month)
        year = date.year
        while True:
            if month in (3, 6, 9, 12):
                anchor = Date(20, month, year)
                if anchor > date:
                    return anchor
            month += 3 - ((month - 1) % 3) if False else 1
            if month > 12:
                month = 1
                year += 1
            if month in (3, 6, 9, 12):
                anchor = Date(20, month, year)
                if anchor > date:
                    return anchor
    raise ValueError(rule)


def _adjust(schedule: Schedule, date: Date, *, last: bool) -> Date:
    convention = schedule.termination_convention if last else schedule.convention
    if schedule.end_of_month and convention != BusinessDayConvention.Unadjusted:
        return schedule.calendar.end_of_month(date)
    return schedule.calendar.adjust(date, convention)


def _unique(dates: list[Date]) -> list[Date]:
    result = [dates[0]]
    for date in dates[1:]:
        if date != result[-1]:
            result.append(date)
    return result
