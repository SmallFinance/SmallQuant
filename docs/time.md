# Dates and day counting

`smallquant.time` is a pure-Python port of QuantLib's date machinery. It has no torch
dependency; the pricing core keeps working in year fractions
(see [conventions.md](conventions.md)), and this module is what turns dates into those
year fractions.

```python
from smallquant.time import Date, Month, Period, TimeUnit, Actual365Fixed, Thirty360

start = Date(31, Month.January, 2024)
end = start + Period(6, TimeUnit.Months)        # 2024-07-31
Actual365Fixed().year_fraction(start, end)      # 182 / 365
Thirty360(Thirty360.Convention.USA).day_count(start, end)  # 180
```

## What is included

| Piece | Notes |
| --- | --- |
| `Date` | Immutable; QuantLib serial numbers and range (1901-01-01 to 2199-12-31); weekday Sunday = 1; end-of-month clamping on month/year arithmetic; `next_weekday`, `nth_weekday`, `end_of_month` |
| `Period`, `TimeUnit`, `Frequency` | Normalisation, cross-unit comparison and arithmetic with the same rules as QuantLib; `Period.parse("3M")` |
| `Actual360`, `Actual366`, `Actual36525` | Optional `include_last_day` |
| `Actual365Fixed` | Standard, Canadian bond and no-leap conventions |
| `Thirty360` | USA, BondBasis/ISMA, European/EurobondBasis, Italian, German/ISDA (with termination date), NASD |
| `ActualActual` | ISDA/Historical/Actual365, AFB/Euro, ISMA/Bond with an explicit reference period |
| `SimpleDayCounter`, `OneDayCounter` | |

Not ported yet: calendars and business-day conventions, `Schedule`, `Business252`, and
`ActualActual` ISMA driven by a `Schedule` (the reference-period form is supported).

## Differences from QuantLib

- Names are snake_case (`year_fraction`, `day_count`) and `Date` fields are properties.
- A missing reference period is `None`, not a null `Date`.
- `Period.__eq__` returns `False` for undecidable comparisons (1M vs 30D) where QuantLib
  throws; `<` still raises `ValueError`.
- Errors are `ValueError`.

## Testing

`tests/time/` compares against QuantLib (the `quantlib` marker): every serial number in the
supported range, `Date ± Period` on every day over two years plus random draws, all `Period`
operations on a grid, and every day counter on a month-end/February edge grid, random pairs
over the whole range, and random reference periods. Year fractions are compared to a relative
1e-14.

Reference behaviour is QuantLib 1.43. QuantLib changes edge cases between releases (30/360 US
reordered its February checks), so a failure after a QuantLib upgrade means a port needs
updating.
