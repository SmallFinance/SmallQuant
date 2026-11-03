"""Date and Period against QuantLib."""

import itertools
import random

import pytest

from smallquant.time import Date, Frequency, Month, Period, TimeUnit, Weekday

ql = pytest.importorskip("QuantLib")

pytestmark = pytest.mark.quantlib

QL_UNITS = {
    TimeUnit.Days: ql.Days,
    TimeUnit.Weeks: ql.Weeks,
    TimeUnit.Months: ql.Months,
    TimeUnit.Years: ql.Years,
}


def to_ql(d: Date) -> "ql.Date":
    return ql.Date(d.day, int(d.month), d.year)


def to_ql_period(p: Period) -> "ql.Period":
    return ql.Period(p.length, QL_UNITS[p.units])


def same(d: Date, q: "ql.Date") -> bool:
    return (d.day, int(d.month), d.year) == (q.dayOfMonth(), q.month(), q.year())


def outcome(fn):
    """Value, or the marker "error" when the call raises (QuantLib and ours differ in type)."""
    try:
        return fn()
    except (RuntimeError, ValueError):
        return "error"


class TestDate:
    def test_range_matches_quantlib(self):
        assert Date.min_date().serial == ql.Date.minDate().serialNumber()
        assert Date.max_date().serial == ql.Date.maxDate().serialNumber()

    def test_every_serial_number(self):
        for serial in range(ql.Date.minDate().serialNumber(), ql.Date.maxDate().serialNumber() + 1):
            d, q = Date.from_serial(serial), ql.Date(serial)
            assert (d.day, int(d.month), d.year) == (q.dayOfMonth(), q.month(), q.year())
            assert d.weekday == q.weekday()
            assert d.day_of_year == q.dayOfYear()
            assert Date(d.day, d.month, d.year).serial == serial

    def test_out_of_range(self):
        for bad in (366, 109575):
            with pytest.raises(ValueError):
                Date.from_serial(bad)
        for args in ((1, 1, 1900), (1, 1, 2200), (29, 2, 2023), (0, 1, 2020), (1, 13, 2020)):
            with pytest.raises(ValueError):
                Date(*args)

    def test_leap_and_month_length(self):
        for y in range(1901, 2200):
            assert Date.is_leap(y) == ql.Date.isLeap(y)
        for y, m in itertools.product((1999, 2000, 2024, 2100), range(1, 13)):
            assert Date.month_length(m, Date.is_leap(y)) == ql.Date.endOfMonth(ql.Date(1, m, y)).dayOfMonth()

    def test_end_of_month(self):
        for serial in range(Date(1, 1, 2023).serial, Date(31, 12, 2025).serial + 1):
            d = Date.from_serial(serial)
            q = to_ql(d)
            assert Date.is_end_of_month(d) == ql.Date.isEndOfMonth(q)
            assert same(Date.end_of_month(d), ql.Date.endOfMonth(q))

    def test_next_and_nth_weekday(self):
        for serial in range(Date(1, 1, 2024).serial, Date(31, 12, 2024).serial + 1):
            d = Date.from_serial(serial)
            for w in Weekday:
                assert same(Date.next_weekday(d, w), ql.Date.nextWeekday(to_ql(d), int(w)))
        for y, m, w, n in itertools.product((2023, 2024), range(1, 13), Weekday, range(1, 6)):
            ours = outcome(lambda: Date.nth_weekday(n, w, m, y))
            theirs = outcome(lambda: ql.Date.nthWeekday(n, int(w), m, y))
            if theirs == "error":
                assert ours == "error"
            else:
                assert same(ours, theirs)

    def test_day_arithmetic(self):
        d = Date(31, Month.December, 2024)
        q = to_ql(d)
        for n in (-1000, -1, 0, 1, 59, 366, 10000):
            assert same(d + n, q + n)
            assert same(d - n, q - n)
            assert same(n + d, q + n)
        assert Date(1, 3, 2024) - Date(1, 2, 2024) == 29
        with pytest.raises(ValueError):
            Date.max_date() + 1

    def test_ordering_and_hash(self):
        a, b = Date(1, 1, 2020), Date(2, 1, 2020)
        assert a < b <= b and b > a >= a and a != b and a == Date(1, 1, 2020)
        assert len({a, Date(1, 1, 2020), b}) == 2

    def test_period_arithmetic_every_day(self):
        periods = [
            Period(n, u) for u in TimeUnit for n in (-25, -13, -12, -3, -1, 0, 1, 2, 6, 12, 13, 25)
        ]
        for serial in range(Date(25, 1, 2023).serial, Date(10, 3, 2025).serial + 1):
            d = Date.from_serial(serial)
            q = to_ql(d)
            for p in periods:
                qp = to_ql_period(p)
                assert same(d + p, q + qp), (d, p)
                assert same(d - p, q - qp), (d, p)

    def test_period_arithmetic_random(self):
        rng = random.Random(1)
        for _ in range(5000):
            d = Date.from_serial(rng.randint(Date.min_date().serial, Date.max_date().serial))
            p = Period(rng.randint(-3000, 3000) // rng.choice((1, 7, 30, 365)), rng.choice(list(TimeUnit)))
            q, qp = to_ql(d), to_ql_period(p)
            ours, theirs = outcome(lambda: d + p), outcome(lambda: q + qp)
            if theirs == "error":
                assert ours == "error", (d, p)
            else:
                assert same(ours, theirs), (d, p)

    def test_period_out_of_range(self):
        with pytest.raises(ValueError):
            Date(1, 1, 2199) + Period(1, TimeUnit.Years)
        with pytest.raises(ValueError):
            Date(1, 1, 1901) - Period(2, TimeUnit.Years)

    def test_datetime_roundtrip(self):
        d = Date(29, 2, 2024)
        assert Date.from_date(d.to_date()) == d
        assert Date.from_iso(d.isoformat()) == d == Date.from_iso("2024-02-29")
        assert str(d) == "2024-02-29"


PERIODS = [Period(n, u) for u in TimeUnit for n in range(-3, 30)] + [
    Period(n, u) for u in TimeUnit for n in (36, 48, 52, 60, 84, 365, 366)
]


class TestPeriod:
    def test_construct_from_frequency(self):
        for f in Frequency:
            ours = outcome(lambda: Period(f))
            theirs = outcome(lambda: ql.Period(int(f)))
            if theirs == "error":
                assert ours == "error", f
            else:
                assert (ours.length, int(ours.units)) == (theirs.length(), theirs.units()), f

    def test_frequency(self):
        for p in PERIODS:
            assert int(p.frequency()) == int(to_ql_period(p).frequency()), p

    def test_normalized(self):
        for p in PERIODS:
            n, qn = p.normalized(), to_ql_period(p).normalized()
            assert (n.length, int(n.units)) == (qn.length(), qn.units()), p

    def test_str(self):
        for p in PERIODS:
            assert str(p) == str(to_ql_period(p)), p

    def test_parse(self):
        for p in PERIODS:
            q = ql.Period(str(to_ql_period(p)))
            assert Period.parse(str(p)) == p
            assert (Period.parse(str(p)).length, int(Period.parse(str(p)).units)) == (q.length(), q.units())
        for bad in ("", "M", "3", "3X", "aM"):
            with pytest.raises(ValueError):
                Period.parse(bad)

    def test_comparison(self):
        for p1, p2 in itertools.product(PERIODS[::3], PERIODS[1::3]):
            q1, q2 = to_ql_period(p1), to_ql_period(p2)
            assert outcome(lambda: p1 < p2) == outcome(lambda: q1 < q2), (p1, p2)
            assert outcome(lambda: p1 > p2) == outcome(lambda: q1 > q2), (p1, p2)
            assert outcome(lambda: p1 <= p2) == outcome(lambda: q1 <= q2), (p1, p2)
            assert outcome(lambda: p1 >= p2) == outcome(lambda: q1 >= q2), (p1, p2)
            theirs_eq = outcome(lambda: q1 == q2)
            assert (p1 == p2) == (theirs_eq is True), (p1, p2)

    def test_equal_periods_hash_equal(self):
        assert Period(12, TimeUnit.Months) == Period(1, TimeUnit.Years)
        assert hash(Period(12, TimeUnit.Months)) == hash(Period(1, TimeUnit.Years))
        assert hash(Period(0, TimeUnit.Years)) == hash(Period(0, TimeUnit.Days))
        assert Period(1, TimeUnit.Months) != Period(30, TimeUnit.Days)

    def test_add_sub(self):
        for p1, p2 in itertools.product(PERIODS[::4], PERIODS[2::4]):
            q1, q2 = to_ql_period(p1), to_ql_period(p2)
            for ours, theirs in ((lambda: p1 + p2, lambda: q1 + q2), (lambda: p1 - p2, lambda: q1 - q2)):
                a, b = outcome(ours), outcome(theirs)
                if b == "error":
                    assert a == "error", (p1, p2)
                else:
                    assert (a.length, int(a.units)) == (b.length(), b.units()), (p1, p2)

    def test_mul_neg_div(self):
        for p in PERIODS:
            q = to_ql_period(p)
            assert (-p).length == (-q).length()
            for n in (-2, 0, 3):
                m = p * n
                assert (m.length, int(m.units)) == ((q * n).length(), (q * n).units())
                assert (n * p) == m

    def test_division(self):
        # the Python bindings do not expose QuantLib's Period / int, so these are the C++ rules
        P = Period
        U = TimeUnit
        assert (P(24, U.Months) / 2) == P(12, U.Months)
        assert (P(24, U.Months) / 2).units == U.Months
        assert (P(1, U.Years) / 2) == P(6, U.Months) and (P(1, U.Years) / 2).units == U.Months
        assert (P(1, U.Weeks) / 7) == P(1, U.Days) and (P(1, U.Weeks) / 7).units == U.Days
        assert (P(3, U.Months) / -3) == P(-1, U.Months)
        for p, n in ((P(1, U.Months), 2), (P(1, U.Years), 5), (P(3, U.Days), 2)):
            with pytest.raises(ValueError):
                p / n
        with pytest.raises(ValueError):
            Period(1, TimeUnit.Days) / 0

    def test_unit_conversions(self):
        P = Period
        U = TimeUnit
        assert P(18, U.Months).years() == 1.5 and P(2, U.Years).months() == 24
        assert P(14, U.Days).weeks() == 2 and P(2, U.Weeks).days() == 14
        assert P(0, U.Months).days() == 0 and P(0, U.Years).weeks() == 0
        for p, fn in ((P(1, U.Days), "years"), (P(1, U.Weeks), "months"), (P(1, U.Months), "weeks"), (P(1, U.Years), "days")):
            with pytest.raises(ValueError):
                getattr(p, fn)()
