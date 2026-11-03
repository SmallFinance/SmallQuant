"""Day counters against QuantLib, on edge-case grids and random date pairs."""

import itertools
import random

import pytest

from smallquant.time import (
    Actual36525,
    Actual360,
    Actual365Fixed,
    Actual366,
    ActualActual,
    Date,
    OneDayCounter,
    Period,
    SimpleDayCounter,
    Thirty360,
    TimeUnit,
)

ql = pytest.importorskip("QuantLib")

pytestmark = pytest.mark.quantlib

A365 = Actual365Fixed.Convention
AA = ActualActual.Convention
T360 = Thirty360.Convention


def to_ql(d):
    return None if d is None else ql.Date(d.day, int(d.month), d.year)


def outcome(fn):
    try:
        return fn()
    except (RuntimeError, ValueError):
        return "error"


# (id, ours, QuantLib). ISDA-style 30/360 with a termination date is added separately.
COUNTERS = [
    ("act360", Actual360(), ql.Actual360()),
    ("act360_inc", Actual360(True), ql.Actual360(True)),
    ("act366", Actual366(), ql.Actual366()),
    ("act366_inc", Actual366(True), ql.Actual366(True)),
    ("act36525", Actual36525(), ql.Actual36525()),
    ("act365f", Actual365Fixed(), ql.Actual365Fixed()),
    ("act365f_noleap", Actual365Fixed(A365.NoLeap), ql.Actual365Fixed(ql.Actual365Fixed.NoLeap)),
    ("30_360_usa", Thirty360(T360.USA), ql.Thirty360(ql.Thirty360.USA)),
    ("30_360_bond", Thirty360(T360.BondBasis), ql.Thirty360(ql.Thirty360.BondBasis)),
    ("30_360_isma", Thirty360(T360.ISMA), ql.Thirty360(ql.Thirty360.ISMA)),
    ("30e_360", Thirty360(T360.European), ql.Thirty360(ql.Thirty360.European)),
    ("30e_360_euro", Thirty360(T360.EurobondBasis), ql.Thirty360(ql.Thirty360.EurobondBasis)),
    ("30_360_italian", Thirty360(T360.Italian), ql.Thirty360(ql.Thirty360.Italian)),
    ("30_360_german", Thirty360(T360.German), ql.Thirty360(ql.Thirty360.German)),
    ("30_360_isda", Thirty360(T360.ISDA), ql.Thirty360(ql.Thirty360.ISDA)),
    ("30_360_nasd", Thirty360(T360.NASD), ql.Thirty360(ql.Thirty360.NASD)),
    ("actact_isda", ActualActual(AA.ISDA), ql.ActualActual(ql.ActualActual.ISDA)),
    ("actact_historical", ActualActual(AA.Historical), ql.ActualActual(ql.ActualActual.Historical)),
    ("actact_365", ActualActual(AA.Actual365), ql.ActualActual(ql.ActualActual.Actual365)),
    ("actact_afb", ActualActual(AA.AFB), ql.ActualActual(ql.ActualActual.AFB)),
    ("actact_euro", ActualActual(AA.Euro), ql.ActualActual(ql.ActualActual.Euro)),
    ("simple", SimpleDayCounter(), ql.SimpleDayCounter()),
    ("one", OneDayCounter(), ql.OneDayCounter()),
]
REF_COUNTERS = [
    ("actact_isma", ActualActual(AA.ISMA), ql.ActualActual(ql.ActualActual.ISMA)),
    ("actact_bond", ActualActual(AA.Bond), ql.ActualActual(ql.ActualActual.Bond)),
    ("act365f_canadian", Actual365Fixed(A365.Canadian), ql.Actual365Fixed(ql.Actual365Fixed.Canadian)),
]
IDS = [c[0] for c in COUNTERS]


def edge_dates():
    """Month ends, month starts and the days around them, incl. leap and non-leap Februaries."""
    out = []
    for y, m in itertools.product(range(2023, 2026), range(1, 13)):
        length = Date.month_length(m, Date.is_leap(y))
        for day in sorted({1, 2, 15, 27, 28, 29, 30, 31, length - 1, length}):
            if 1 <= day <= length:
                out.append(Date(day, m, y))
    return out


def random_pairs(n, seed, lo=Date(1, 1, 1901), hi=Date(31, 12, 2199)):
    rng = random.Random(seed)
    return [
        (Date.from_serial(rng.randint(lo.serial, hi.serial)), Date.from_serial(rng.randint(lo.serial, hi.serial)))
        for _ in range(n)
    ]


def check(ours, theirs, d1, d2, r1=None, r2=None):
    q1, q2, qr1, qr2 = to_ql(d1), to_ql(d2), to_ql(r1), to_ql(r2)
    a = outcome(lambda: ours.year_fraction(d1, d2, r1, r2))
    b = outcome(lambda: theirs.yearFraction(q1, q2, qr1, qr2) if r1 else theirs.yearFraction(q1, q2))
    if b == "error":
        assert a == "error", (ours.name, d1, d2, r1, r2)
    else:
        assert a == pytest.approx(b, rel=1e-14, abs=1e-15), (ours.name, d1, d2, r1, r2)
    if not r1:
        assert ours.day_count(d1, d2) == theirs.dayCount(q1, q2), (ours.name, d1, d2)


@pytest.mark.parametrize("cid,ours,theirs", COUNTERS, ids=IDS)
def test_names(cid, ours, theirs):
    assert ours.name == theirs.name()


@pytest.mark.parametrize("cid,ours,theirs", COUNTERS, ids=IDS)
def test_edge_grid(cid, ours, theirs):
    for d1, d2 in itertools.product(edge_dates(), repeat=2):
        check(ours, theirs, d1, d2)


@pytest.mark.parametrize("cid,ours,theirs", COUNTERS, ids=IDS)
def test_random_pairs_full_range(cid, ours, theirs):
    for d1, d2 in random_pairs(5000, seed=7):
        check(ours, theirs, d1, d2)


@pytest.mark.parametrize("cid,ours,theirs", COUNTERS, ids=IDS)
def test_same_date_and_short_spans(cid, ours, theirs):
    d = Date(28, 2, 2024)
    for k in range(-10, 11):
        check(ours, theirs, d, d + k)


@pytest.mark.parametrize("term", [Date(28, 2, 2025), Date(29, 2, 2024), Date(31, 8, 2024)])
def test_thirty360_isda_termination_date(term):
    ours = Thirty360(T360.ISDA, term)
    theirs = ql.Thirty360(ql.Thirty360.ISDA, to_ql(term))
    for d1, d2 in itertools.product(edge_dates(), repeat=2):
        check(ours, theirs, d1, d2)
    for d1 in edge_dates():
        check(ours, theirs, d1, term)


@pytest.mark.parametrize("cid,ours,theirs", REF_COUNTERS, ids=[c[0] for c in REF_COUNTERS])
def test_reference_period_counters(cid, ours, theirs):
    assert ours.name == theirs.name()
    rng = random.Random(3)
    months_choices = (1, 2, 3, 4, 6, 12)
    for _ in range(4000):
        start = Date.from_serial(rng.randint(Date(1, 1, 1990).serial, Date(31, 12, 2060).serial))
        months = rng.choice(months_choices)
        end = start + Period(months, TimeUnit.Months)
        # dates around and beyond the reference period, in either order
        d1 = start + rng.randint(-200, 300)
        d2 = start + rng.randint(-200, 500)
        check(ours, theirs, d1, d2, start, end)


@pytest.mark.parametrize("cid,ours,theirs", REF_COUNTERS[:2], ids=["isma", "bond"])
def test_isma_without_reference_period(cid, ours, theirs):
    for d1, d2 in random_pairs(2000, seed=11, lo=Date(1, 1, 1990), hi=Date(31, 12, 2060)):
        check(ours, theirs, d1, d2)


def test_canadian_requires_reference_period():
    c = Actual365Fixed(A365.Canadian)
    with pytest.raises(ValueError):
        c.year_fraction(Date(1, 1, 2024), Date(1, 2, 2024))
    assert c.year_fraction(Date(1, 1, 2024), Date(1, 1, 2024)) == 0.0


def test_equality_by_name():
    assert Actual360() == Actual360()
    assert Actual360() != Actual360(True)
    assert Thirty360(T360.BondBasis) == Thirty360(T360.ISMA)
    assert len({Thirty360(T360.BondBasis), Thirty360(T360.ISMA)}) == 1
