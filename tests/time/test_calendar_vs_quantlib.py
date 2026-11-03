"""Calendars and business-day conventions against QuantLib."""

import itertools
import random

import pytest

from smallquant.time import (
    TARGET,
    BespokeCalendar,
    BusinessDayConvention,
    Date,
    Japan,
    JointCalendar,
    JointCalendarRule,
    NullCalendar,
    Period,
    TimeUnit,
    UnitedKingdom,
    UnitedStates,
    Weekday,
    WeekendsOnly,
)

ql = pytest.importorskip("QuantLib")

pytestmark = pytest.mark.quantlib

BDC = {
    BusinessDayConvention.Following: ql.Following,
    BusinessDayConvention.ModifiedFollowing: ql.ModifiedFollowing,
    BusinessDayConvention.Preceding: ql.Preceding,
    BusinessDayConvention.ModifiedPreceding: ql.ModifiedPreceding,
    BusinessDayConvention.Unadjusted: ql.Unadjusted,
    BusinessDayConvention.HalfMonthModifiedFollowing: ql.HalfMonthModifiedFollowing,
    BusinessDayConvention.Nearest: ql.Nearest,
}

QL_UNITS = {
    TimeUnit.Days: ql.Days,
    TimeUnit.Weeks: ql.Weeks,
    TimeUnit.Months: ql.Months,
    TimeUnit.Years: ql.Years,
}


def to_ql(d: Date) -> "ql.Date":
    return ql.Date(d.day, int(d.month), d.year)


def from_ql(q: "ql.Date") -> Date:
    return Date(q.dayOfMonth(), q.month(), q.year())


def same(d: Date, q: "ql.Date") -> bool:
    return (d.day, int(d.month), d.year) == (q.dayOfMonth(), q.month(), q.year())


CALENDARS = {
    "null": (NullCalendar(), ql.NullCalendar()),
    "weekends": (WeekendsOnly(), ql.WeekendsOnly()),
    "target": (TARGET(), ql.TARGET()),
    "us_settlement": (UnitedStates(), ql.UnitedStates(ql.UnitedStates.Settlement)),
    "us_nyse": (UnitedStates(UnitedStates.Market.NYSE), ql.UnitedStates(ql.UnitedStates.NYSE)),
    "us_gov": (
        UnitedStates(UnitedStates.Market.GovernmentBond),
        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    ),
    "us_sofr": (UnitedStates(UnitedStates.Market.SOFR), ql.UnitedStates(ql.UnitedStates.SOFR)),
    "us_libor": (
        UnitedStates(UnitedStates.Market.LiborImpact),
        ql.UnitedStates(ql.UnitedStates.LiborImpact),
    ),
    "us_nerc": (UnitedStates(UnitedStates.Market.NERC), ql.UnitedStates(ql.UnitedStates.NERC)),
    "us_fed": (
        UnitedStates(UnitedStates.Market.FederalReserve),
        ql.UnitedStates(ql.UnitedStates.FederalReserve),
    ),
    "uk": (UnitedKingdom(), ql.UnitedKingdom()),
    "uk_exchange": (
        UnitedKingdom(UnitedKingdom.Market.Exchange),
        ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
    ),
    "uk_metals": (
        UnitedKingdom(UnitedKingdom.Market.Metals),
        ql.UnitedKingdom(ql.UnitedKingdom.Metals),
    ),
    "japan": (Japan(), ql.Japan()),
    "joint": (
        JointCalendar(TARGET(), UnitedKingdom()),
        ql.JointCalendar(ql.TARGET(), ql.UnitedKingdom()),
    ),
    "joint_bd": (
        JointCalendar(TARGET(), UnitedStates(), rule=JointCalendarRule.JoinBusinessDays),
        ql.JointCalendar(
            ql.TARGET(),
            ql.UnitedStates(ql.UnitedStates.Settlement),
            ql.JoinBusinessDays,
        ),
    ),
}


@pytest.mark.parametrize("key", CALENDARS.keys())
def test_name_matches(key):
    ours, ref = CALENDARS[key]
    assert ours.name == ref.name()


@pytest.mark.parametrize("key", CALENDARS.keys())
def test_is_business_day_grid(key):
    ours, ref = CALENDARS[key]
    start = Date(1, 1, 1950).serial
    end = Date(31, 12, 2035).serial
    for serial in range(start, end + 1):
        d = Date.from_serial(serial)
        assert ours.is_business_day(d) == ref.isBusinessDay(to_ql(d)), (key, d)
        assert ours.is_holiday(d) == ref.isHoliday(to_ql(d)), (key, d)
        assert ours.is_weekend(d.weekday) == ref.isWeekend(int(d.weekday)), (key, d)


@pytest.mark.parametrize("key", ["target", "us_settlement", "uk", "japan", "weekends"])
def test_adjust_all_conventions(key):
    ours, ref = CALENDARS[key]
    rng = random.Random(key)
    for _ in range(400):
        d = Date.from_serial(rng.randint(Date(1, 1, 1990).serial, Date(31, 12, 2030).serial))
        for bdc, qbdc in BDC.items():
            assert same(ours.adjust(d, bdc), ref.adjust(to_ql(d), qbdc)), (key, d, bdc)


@pytest.mark.parametrize("key", ["target", "us_nyse", "uk", "japan"])
def test_advance_and_business_days_between(key):
    ours, ref = CALENDARS[key]
    rng = random.Random(f"advance-{key}")
    periods = [
        Period(0, TimeUnit.Days),
        Period(1, TimeUnit.Days),
        Period(-3, TimeUnit.Days),
        Period(2, TimeUnit.Weeks),
        Period(1, TimeUnit.Months),
        Period(-1, TimeUnit.Months),
        Period(1, TimeUnit.Years),
        Period(6, TimeUnit.Months),
    ]
    for _ in range(200):
        d = Date.from_serial(rng.randint(Date(1, 1, 2000).serial, Date(31, 12, 2028).serial))
        bdc = rng.choice(list(BDC))
        eom = rng.choice((False, True))
        for p in periods:
            got = ours.advance(d, p, convention=bdc, end_of_month=eom)
            want = ref.advance(to_ql(d), ql.Period(p.length, QL_UNITS[p.units]), BDC[bdc], eom)
            assert same(got, want), (key, d, p, bdc, eom)

        n = rng.randint(-10, 10)
        got = ours.advance(d, n, TimeUnit.Days, convention=bdc)
        want = ref.advance(to_ql(d), n, ql.Days, BDC[bdc])
        assert same(got, want), (key, d, n, bdc)

        d2 = Date.from_serial(rng.randint(Date(1, 1, 2000).serial, Date(31, 12, 2028).serial))
        for include_first, include_last in itertools.product((False, True), repeat=2):
            assert ours.business_days_between(
                d, d2, include_first, include_last
            ) == ref.businessDaysBetween(to_ql(d), to_ql(d2), include_first, include_last)


@pytest.mark.parametrize("key", ["target", "us_settlement", "japan"])
def test_month_boundaries_and_lists(key):
    ours, ref = CALENDARS[key]
    for serial in range(Date(1, 1, 2020).serial, Date(31, 12, 2026).serial + 1):
        d = Date.from_serial(serial)
        q = to_ql(d)
        assert ours.is_start_of_month(d) == ref.isStartOfMonth(q)
        assert ours.is_end_of_month(d) == ref.isEndOfMonth(q)
        assert same(ours.start_of_month(d), ref.startOfMonth(q))
        assert same(ours.end_of_month(d), ref.endOfMonth(q))

    start, end = Date(1, 1, 2024), Date(31, 12, 2024)
    assert [d.isoformat() for d in ours.holiday_list(start, end)] == [
        from_ql(q).isoformat() for q in ref.holidayList(to_ql(start), to_ql(end))
    ]
    assert [d.isoformat() for d in ours.holiday_list(start, end, include_weekends=True)] == [
        from_ql(q).isoformat()
        for q in ref.holidayList(to_ql(start), to_ql(end), True)
    ]
    assert [d.isoformat() for d in ours.business_day_list(start, end)] == [
        from_ql(q).isoformat() for q in ref.businessDayList(to_ql(start), to_ql(end))
    ]


def test_add_remove_holiday_is_per_instance():
    a, b = TARGET(), TARGET()
    day = Date(3, 1, 2024)  # Wednesday business day on TARGET
    assert a.is_business_day(day)
    a.add_holiday(day)
    assert a.is_holiday(day)
    assert b.is_business_day(day)
    a.remove_holiday(Date(1, 1, 2024))
    assert a.is_business_day(Date(1, 1, 2024))
    assert b.is_holiday(Date(1, 1, 2024))
    a.reset_added_and_removed_holidays()
    assert a.is_holiday(Date(1, 1, 2024))
    assert a.is_business_day(day)


def test_bespoke_matches_quantlib():
    ours = BespokeCalendar("Custom")
    ref = ql.BespokeCalendar("Custom")
    for w in (Weekday.Saturday, Weekday.Sunday, Weekday.Friday):
        ours.add_weekend(w)
        ref.addWeekend(int(w))
    for serial in range(Date(1, 1, 2024).serial, Date(31, 12, 2024).serial + 1):
        d = Date.from_serial(serial)
        assert ours.is_business_day(d) == ref.isBusinessDay(to_ql(d)), d
    day = Date(10, 1, 2024)  # Wednesday
    ours.add_holiday(day)
    ref.addHoliday(to_ql(day))
    assert ours.is_holiday(day) and ref.isHoliday(to_ql(day))
