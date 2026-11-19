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

### Known issues

- Monte Carlo ignores `steps` for instruments with an observation grid, so European,
  digital and barrier runs simulate one step and discrete barriers are monitored only at
  0 and T (spec MC-02). This blocks the 0.1.0 tag.
