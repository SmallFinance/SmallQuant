# Changelog

All notable changes to SmallQuant are recorded here. The project follows semantic
versioning; breaking changes before 1.0 are allowed and listed under their release.

## Unreleased

### Added

- MIT licence (`LICENSE`) and the `py.typed` marker.
- `ruff check` in CI with pyflakes, syntax-level pycodestyle and import-order rules.
- Build scope version 1.1 (`docs/specs.md`): current state re-baselined, JAX removed from
  scope, owner decisions recorded, market risk and credit risk added.

### Removed

- Unused local variables and imports reported by ruff. No behaviour changes.

### Fixed

- Monte Carlo honours `steps`: the time grid merges the instrument's dates with `steps`
  equal steps (spec MC-02). Before, every run simulated one step and a discrete barrier
  was checked only at 0 and expiry.
- Monte Carlo prices discrete single and double barriers; their payoffs raised `TypeError`.
- An Asian option with `fixing_times` averages on its fixing dates only, and an American
  option with `exercise_grid` exercises on those dates only, when the simulation grid is
  finer. An American option without `exercise_grid` now exercises on every step; before,
  its grid was `[0, T]`, so it priced as a European.
