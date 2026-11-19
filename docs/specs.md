# SmallQuant Production Build Scope

**Implementation specification**

Prepared for Ted Wong and SmallQuant contributors | 19 November 2026 | Version 1.1

**Purpose:** turn the code that exists today into a complete, tested, PyTorch-native
derivatives pricing library, and extend it to market risk and credit risk (sections 17A and
17B). This document records the final product decisions, current behaviour, production
gaps, interface contracts, numerical requirements and acceptance criteria. It is a build
baseline for estimation and implementation; Appendix B records the decisions taken and the
few that still need an owner's answer.

**Code baseline:** `main` at commit `fae8063` (19 November 2026), inspected on 19 November
2026. Source: `smallquant/` and `tests/` in the repository. Test run: 541 passed in 180 s on
CPU (Python 3.13.12, PyTorch 2.14.1, QuantLib 1.44, NumPy 2.5.4, TF Quant Finance 0.0.1.dev34, macOS arm64).

**Status:** pricing phases 0–4 are implemented on PyTorch, with the gaps listed in section 4.
Release tooling is partly done. Market risk and credit risk are not started. JAX is out of
scope (PRD-08).

**Changes in version 1.1:** current state re-baselined from `35e7e4b` to `fae8063`; JAX
removed from scope; licence decided (MIT); decisions the code has already taken recorded in
Appendix B; market risk and credit risk added.

## Contents

1. [How to use this specification](#1-how-to-use-this-specification)
2. [Product outcome and boundaries](#2-product-outcome-and-boundaries)
3. [Final decisions that must survive the build](#3-final-decisions-that-must-survive-the-build)
4. [Current library and remaining gap](#4-current-library-and-remaining-gap)
5. [Users and supported environments](#5-users-and-supported-environments)
6. [Tensor, dtype and device contract](#6-tensor-dtype-and-device-contract)
7. [Mathematical primitives](#7-mathematical-primitives)
8. [Closed-form models](#8-closed-form-models)
9. [Greeks and differentiation](#9-greeks-and-differentiation)
10. [Implied volatility and surfaces](#10-implied-volatility-and-surfaces)
11. [Instruments](#11-instruments)
12. [Engines and the price entry point](#12-engines-and-the-price-entry-point)
13. [Monte Carlo](#13-monte-carlo)
14. [PDE, Fourier and lattice engines](#14-pde-fourier-and-lattice-engines)
15. [Calibration](#15-calibration)
16. [Rates](#16-rates)
17. [Dates, day counts and calendars](#17-dates-day-counts-and-calendars)
    - [17A Market risk](#17a-market-risk)
    - [17B Credit risk](#17b-credit-risk)
18. [Public API surface and packaging](#18-public-api-surface-and-packaging)
19. [Errors and validation](#19-errors-and-validation)
20. [Performance and compilation](#20-performance-and-compilation)
21. [Documentation](#21-documentation)
22. [Verification and acceptance](#22-verification-and-acceptance)
23. [Delivery plan and definition of done](#23-delivery-plan-and-definition-of-done)
- [Appendix A Argument naming reference](#appendix-a-argument-naming-reference)
- [Appendix B Decisions requiring an owner](#appendix-b-decisions-requiring-an-owner)
- [Appendix C Code and evidence map](#appendix-c-code-and-evidence-map)
- [Appendix D Reference sources](#appendix-d-reference-sources)
- [Appendix E Related libraries](#appendix-e-related-libraries)

## 1 How to use this specification

Read sections 2–6 first. They fix the outcome, the decisions that must not drift and the
tensor contract every module obeys. Sections 7–17 describe each pricing area; sections 17A
and 17B add market risk and credit risk. Sections 18–23 cover packaging, validation,
performance, documentation, testing and delivery. Each numbered requirement is suitable for
conversion into an issue; keep its identifier in the issue title, the test name or
docstring, and the pull request description.

Labels distinguish the authority of a statement:

- **Confirmed requirement** means it is agreed in this specification or is behaviour
  already shipped and tested.
- **Current implementation** describes observed code behaviour at the baseline commit,
  including limitations. It is not automatically a permanent policy.
- **Build requirement** is an engineering requirement proposed here to make the design
  reliable.
- **Decided** records an owner decision, or a default this specification takes, with its
  date. Appendix B lists them.
- **Decision required** means an owner must resolve it before the affected behaviour is
  released.

The earlier design documents (`docs/overview.md`, `api.md`, `roadmap.md` and the others)
were removed from the repository; this specification is the only design document. Where it
and the code disagree on a public name or signature, the section text records which one
wins. Argument lists may grow, but names and semantics may not change without a recorded
decision.

Version 1.1 was written from an audit of the full source tree against every requirement and
a fresh test run at the baseline commit. It is not a claim that every planned numerical
method has been prototyped.

## 2 Product outcome and boundaries

**PRD-01 Confirmed requirement.** SmallQuant is the maintained, PyTorch-native equivalent of
Google's archived TF Quant Finance. Every price is a tensor computation, so Greeks of any
order are exact derivatives from autograd, and the same code runs batched on CPU, CUDA and
Apple MPS.

**PRD-02 Confirmed requirement.** Accuracy comes first, speed second. `float64` is the
default. Speed comes from batching, GPUs and `torch.compile`, never from silently dropping
precision.

**PRD-03 Confirmed requirement.** Models, payoffs and numerical engines are separate,
composable pieces. A payoff can be priced by an analytic formula, Monte Carlo, a PDE solver,
Fourier inversion or a lattice wherever each applies.

**PRD-04 Confirmed requirement.** The only runtime dependency is PyTorch. NumPy, QuantLib,
pytest, TF Quant Finance and TensorFlow are test-only.

**PRD-05 Scope boundary.** Not in scope: trade booking, trade stores, market-data
connectors, regulatory reporting templates, a full QuantLib replacement, and
machine-learning price models. Deep hedging and neural pricers may be built on top of
SmallQuant but are not part of it.

**PRD-06 Confirmed requirement.** The pricing core works in year fractions and never
imports `smallquant.time`. The time module is an optional helper that turns dates into year
fractions.

**PRD-07 Confirmed requirement.** Target users are quants wanting exact Greeks and fast
calibration, risk teams running large GPU Monte Carlo jobs (XVA, VaR, scenario sweeps), and
researchers combining classical pricing with ML in one PyTorch graph.

**PRD-08 Decided (19 November 2026).** No JAX backend and no other array framework.
SmallQuant is PyTorch only. Requirements that existed only for JAX (former TEN-09, TEN-10,
ENV-05, PRF-05, AT-24 and decisions D12–D15) are withdrawn.

**PRD-09 Confirmed requirement.** SmallQuant covers market risk and credit risk on top of
its pricing: portfolio sensitivities, scenario and stress revaluation, value at risk and
expected shortfall (section 17A); default probability curves, credit derivatives, exposure
simulation and credit valuation adjustment (section 17B). Both reuse the pricing engines and
autograd, so risk numbers and their sensitivities come from the same tensor graph as prices.
Risk inputs (positions, scenarios, market data) are tensors the caller supplies; PRD-05
still excludes trade stores and market-data connectors.

## 3 Final decisions that must survive the build

| Area | Final instruction |
| --- | --- |
| Framework | PyTorch only (PRD-08). No JAX, TensorFlow or NumPy code paths in `smallquant/`. |
| Tensor inputs | Public numeric arguments are `torch.Tensor`. No Python `float`/`int`, NumPy scalars/arrays, or other raw types on the public surface. |
| Licence | MIT (decision D01, 19 November 2026). Code copied from another library must carry a compatible licence; copyleft and non-commercial sources are test references only (Appendix E). |
| Risk | Market and credit risk numbers are computed by revaluing through the pricing engines and differentiating with autograd, never by a separate approximate pricer (sections 17A, 17B). |
| Default dtype | `float64`. MPS falls back to `float32` with a one-time warning. |
| Batching | Every public function broadcasts its tensor inputs. No Python loops over batch elements. |
| Device | Outputs live on the inputs' device. Mixed devices raise; nothing is copied silently. |
| Branching | No Python `if` on tensor values. Use `torch.where`, and mask inputs before unsafe operations. |
| Greeks | Autograd, never bump-and-reprice. `greeks.bump` exists for checking only. |
| Solvers | Implied vol, roots, bootstrapping and calibration differentiate through the implicit function theorem, not by unrolling iterations. |
| Discontinuous payoffs | Smoothing, likelihood ratio or one-step survival, chosen per instrument and documented. Autograd's zero delta on a digital is a bug, not an answer. |
| Early exercise | Longstaff-Schwartz boundary held fixed under `no_grad` for Greeks; projected Crank-Nicolson in PDE. |
| Path layout | `[num_paths, num_steps + 1, dim]`; paths first, index 0 is the initial state, trailing `dim` kept for one-factor models. |
| Time | Year fractions in the core. Rates continuously compounded unless a function says otherwise. |
| Argument names | `spot`, `forward`, `strike`, `expiry`, `rate`, `div`, `vol`, `normal_vol`, `is_call`, `discount` (Appendix A). Keyword-only after the first few. |
| API levels | Functional (plain functions on tensors) is primary; object level (instrument + params + engine) is a thin layer over it. |
| Layering | Lower layers never import upper ones. `math/` knows no finance; `black_scholes/` knows no models, exotics or engines; `exotics/` and `models/` may use `black_scholes/` but know no engines. |
| Package layout | Follows TF Quant Finance. `black_scholes/` has exactly TF's files: `vanilla_prices.py` (`option_price` for Black-Scholes, Black-76 and Bachelier; `binary_price`, `asset_or_nothing_price`, `barrier_price`), `asian_prices.py`, `brownian_bridge.py`, `variance_swaps.py`, `implied_vol_lib.py` / `implied_vol_newton_root.py` / `implied_vol_approximation.py` / `implied_vol_utils.py`, `crr_binomial_tree.py`, `approximations/american_option.py`. `exotics/`: one `*_prices.py` per exotic family (barriers, lookbacks, choosers, compounds, cliquets, quantos, two-asset and others). `models/`: dynamics only (`geometric_brownian_motion`, `arithmetic_brownian_motion`, `merton`, `cev`, `local_vol`, `hull_white`, `params`), with `heston/` and `sabr/` packages whose prices live in `approximations/`. |
| Validation | Value checks (`vol > 0`) are off by default because they sync the GPU; shape/dtype/device checks are always on. |
| Monte Carlo results | Always reported with a standard error in tests, examples and docs. |
| Benchmarks | No performance number is quoted before it is measured and labelled with hardware and versions. |
| Time module | Pure Python port of QuantLib 1.4x date machinery, snake_case names, per-instance calendar holiday edits. |

## 4 Current library and remaining gap

The repository holds about 10,500 lines of Python in `smallquant/` and 541 tests, all
passing on CPU in `float64`. The tests compare against QuantLib 1.44 and TF Quant Finance.
The table is the state at `fae8063`; the requirement sections carry the detail.

| Capability | At `fae8063` | Remaining gap |
| --- | --- | --- |
| Package entry point | `import smallquant as sq` exposes `black_scholes`, `exotics`, `models`, `engines`, `instruments`, `payoffs`, `greeks`, `calibration` (and `sq.calibrate`), `rates`, `vol`, `math`, `sq.price`, `sq.Market`, `set_default_dtype` / `default_dtype()` and `set_validation` / `validation()`. It never imports `smallquant.time`. | A test for every API-01 name and for the time-import boundary (AT-01, AT-22) |
| Tensor contract | `_tensor.align` rejects non-tensors with `TypeError`, promotes to the widest float, honours `dtype=`, falls back to `float32` on MPS with one warning, and rejects mixed devices | Bare `torch.tensor` constants remain (TEN-05); `barrier_price` is not masked (TEN-06); device and dtype tests (AT-02, AT-03) |
| Closed forms | `option_price` for Black-Scholes-Merton, Black-76 and Bachelier; binary, asset-or-nothing and single-barrier prices; about 20 exotic families in `exotics/`; Merton series; SABR (Hagan); Heston (Attari); Barone-Adesi-Whaley and Bjerksund-Stensland; variance swaps | Greeks tests for Black-76, Bachelier and binaries (AT-06); no-arbitrage property tests (AT-07); citations in barrier, CEV, SSVI and Hull-White code (MOD-09) |
| Greeks | `first_order`, `second_order`, `jacobian`, `bump` | Named-Greek helpers (GRK-05); a `jacobian` test; the zero-delta warning on barriers (GRK-07) |
| Implied vol | Newton (Brenner-Subrahmanyam start, implicit-function backward) and the Stefanica-Radoicic one-shot formula, lognormal and normal | `tol` is accepted but unused; no convergence mask is exposed (MTH-04); a grid round trip to 1e-10 (AT-09) |
| Vol surfaces | SVI and SSVI as functions, `dupire_local_vol`, `svi_butterfly_ok`, `ssvi_calendar_ok` | SSVI butterfly and SVI calendar checks (VOL-06); Dupire's second strike derivative is not differentiable |
| Math | Normal and bivariate normal distributions, masked `newton` and `brent` with implicit-function backward, interpolation, quadrature, tridiagonal and Cholesky helpers, pseudo-random and Sobol normals with a Brownian bridge | `newton` clamps the slope positive, which fails on decreasing functions (MTH-02); NaN for non-converged elements under validation (MTH-04); tests beyond the tridiagonal solver |
| Engines | `Analytic`, `MonteCarlo`, `PDE`, `Fourier` and `Lattice`, dispatched through a registry keyed by (instrument type, params type). The Monte Carlo grid merges instrument dates with `steps` equal steps (MC-02, fixed after `fae8063`) | The engine does not expose Sobol, control variates or standard errors (MC-05, MC-06, MC-09); no maximum step size option |
| American exercise | Longstaff-Schwartz in Monte Carlo, Crank-Nicolson with `max(value, payoff)` after each step, CRR tree, analytic approximations | Longstaff-Schwartz tests and the 2,000-step tree comparison (AT-16); a true projected step (PDE-02) |
| Calibration | `calibrate` with L-BFGS, softplus, `tanh` and `(lo, hi)` bounds, and implicit-function sensitivities to quotes | `loss="vol_rmse"` computes the price RMSE; `iterations` always reports `max_iter`; no stop reason (CAL-01, CAL-04); Heston and SABR calibration tests (AT-17) |
| Rates | `DiscountCurve`, `ZeroCurve`, closed-form `bootstrap`, `par_swap_rate`; Hull-White bond, zero-bond option, caplet, Jamshidian swaption and exact step | Caps and floors as sums, floorlets, a batched swaption, `HullWhiteParams` in the engines (RTS-03); Hull-White tests (AT-19); bucketed deltas against bumps (AT-18) |
| Dates | Calendars for TARGET, United States, United Kingdom, Japan, Australia and the generic ones; ten day-counter families including `Business252` and schedule-driven `ActualActual` ISMA; `Schedule` with seven date-generation rules | Stub detection (`is_regular` is always true), first and next-to-last dates, more `Schedule` tests (TIM-04) |
| Tests | 541 tests on CPU in `float64` | `conftest.py` device and dtype parametrisation, wide `gradcheck` / `gradgradcheck`, compile tests, documentation example tests (QA-03) |
| Tooling | Hatchling wheel; CI on macOS CPU for Python 3.10 and 3.13 running `ruff check`, `pytest` and `uv build`; MIT `LICENSE`; `py.typed`; `CHANGELOG.md` (the ruff, licence, marker and changelog items land with this version of the specification) | Version is `0.0.1`; tag `0.1.0` when the Phase 0 exit tests pass; PyPI name and publishing (D01) |
| Benchmarks | `scripts/european_mc.py` with results in `docs/european-monte-carlo.md` | `benchmarks/` with chain pricing, implied vol over a surface, and Heston Monte Carlo with Greeks (PRF-02) |
| Market risk | None | Section 17A |
| Credit risk | None | Section 17B |

Stale material to clean up, not to preserve:

- The `ActualActual` class docstring says it has no `Schedule` support; it now accepts
  `schedule=` for ISMA.
- A comment in `smallquant/time/calendars.py` cites QuantLib 1.43; the reference is 1.44.

## 5 Users and supported environments

**ENV-01 Confirmed requirement.** Python `>=3.10,<3.15`. Development uses Python 3.13
(`.python-version`) and `uv` with the committed `uv.lock`.

**ENV-02 Confirmed requirement.** Devices: CPU, CUDA and MPS. Tests are parametrised over
available devices. MPS has no `float64`; see TEN-04.

**ENV-03 Decided (default, 19 November 2026).** The minimum supported PyTorch is the version
in `uv.lock` (2.14) and `pyproject.toml` states it. It is lowered only after CI passes on the
lower version. `torch.func`, `torch.compile` and `torch.quasirandom.SobolEngine` behaviour
must be available at that minimum.

**ENV-04 Decided (default, 19 November 2026).** CI runs on macOS CPU (Linux runners pull
CUDA wheels). CUDA and MPS are checked by hand before each release, and the release notes
record the hardware and the result.

## 6 Tensor, dtype and device contract

**TEN-01 Confirmed requirement.** Public numeric arguments are `torch.Tensor` only. The
library does not accept raw Python numbers (`float`, `int`), NumPy scalars/arrays, or other
array-likes on the public surface; callers pass tensors. Non-numeric flags such as
`is_call` may remain `bool` or a bool tensor. All tensor arguments broadcast under PyTorch
rules and the output shape is the broadcast shape.

**TEN-02 Confirmed requirement.** dtype resolution order: the `dtype=` argument if given,
otherwise the highest floating dtype among tensor inputs, otherwise `float64`.

**TEN-03 Current implementation.** `_tensor.align` meets TEN-01 and TEN-02: it raises
`TypeError` for non-tensor numeric arguments, promotes mixed float dtypes to the widest,
honours `dtype=`, casts integer tensors and keeps bool tensors. `is_call` may also be a
Python `bool`.

**TEN-04 Build requirement.** On MPS, when resolution would give `float64`, use `float32`
and emit one `UserWarning` per process. Never raise because of the MPS fallback. Explicit
`dtype=torch.float64` on MPS raises a clear error.

**TEN-05 Confirmed requirement.** All inputs share one device; mixing raises. Constants are
Python scalars in arithmetic or tensors created with the inputs' `dtype` and `device`. A
bare `torch.tensor(...)` constant in library code is a defect.

**TEN-06 Confirmed requirement.** Every price is differentiable with respect to every
floating-point input. No in-place operations on tensors that may require grad. No Python
`if` on tensor values. Mask inputs before unsafe operations so unselected branches cannot
produce NaN gradients.

**TEN-07 Build requirement.** `_tensor.py` provides: helpers that validate and align
`torch.Tensor` inputs on a common dtype/device (rejecting non-tensors per TEN-01),
`common_device`, `broadcast_shapes`, `zeros_like_inputs` style constant helpers, and the MPS
fallback. It is private and may change, but every public function uses it. At `fae8063`
`align`, `check`, the dtype and validation state and the MPS fallback exist; the device,
shape and constant helpers do not, and bare `torch.tensor` constants remain in
`math/distributions.py`, `black_scholes/vanilla_prices.py`, `models/hull_white.py` and
`engines/fourier.py` (TEN-05).

**TEN-08 Confirmed requirement.** Extra shape-defining dimensions are documented per
function in a fixed position: paths, then time, then state dimension.

## 7 Mathematical primitives

**MTH-01 Confirmed requirement.** `sq.math.distributions.normal_cdf`, `normal_pdf`,
`normal_icdf` and `bivariate_normal_cdf(x, y, rho)`, differentiable and accurate in the
tails. `normal_cdf` uses `erfc` for negative arguments so deep out-of-the-money prices keep
relative accuracy. The functions live in `sq.math.distributions`; they are not re-exported
from `sq.math`.

**MTH-02 Confirmed requirement.** `roots.newton(residual, x0, params, *, tol=1e-12,
max_iter=50)` and `roots.brent(residual, lo, hi, params, *, tol=1e-12, max_iter=100)` are
batched, run a fixed maximum iteration count with a convergence mask, and never sync to the
host inside the loop. `params` carries the tensors the implicit-function backward
differentiates. `brent` is bisection with a Newton trial step, not Brent's method; the name
stays. Defect at `fae8063`: `newton` clamps the slope to at least 1e-14, so it fails on
decreasing residuals.

**MTH-03 Confirmed requirement.** Root results are differentiable through the implicit
function theorem: one backward step at the solution, `dx/dθ = -(∂g/∂x)⁻¹ · ∂g/∂θ`.

**MTH-04 Build requirement.** Roots return `(root, converged)` so callers can detect
non-converged elements without a host sync (done). Non-converged elements return NaN only
when validation is on; otherwise the last iterate, documented as such (not done).
`implied_vol` computes a convergence mask and discards it; it should honour `tol` and expose
the mask.

**MTH-05 Confirmed requirement.** Interpolation: `linear`, `log_linear`, `monotone_cubic`
and natural cubic spline; differentiable with respect to both `x` and the node values `fp`.

**MTH-06 Confirmed requirement.** Quadrature: Gauss-Legendre, Gauss-Laguerre and Simpson.
Batched tridiagonal (Thomas) solver and Cholesky helpers in `linalg.py`.

**MTH-07 Confirmed requirement.** `random.standard_normal(*, num_paths, num_steps, dim, seed,
dtype, device, antithetic=False, method="pseudo" | "sobol")` and the `NormalStream` class that
replays fixed blocks for chunked runs; see section 13 for Sobol scrambling and Brownian
bridge.

## 8 Closed-form models

**MOD-01 Confirmed requirement and current implementation.** Black-Scholes-Merton is
`sq.black_scholes.option_price` with `spot` and `vol` (MOD-04). It prices calls and puts
with continuous dividend yield, accepts a bool or bool tensor `is_call`, defaults to
`float64` and matches QuantLib's `AnalyticEuropeanEngine` and TF Quant Finance to relative
1e-9 for price, delta, gamma, vega, rho and the expiry derivative. The old path
`sq.models.black_scholes.price` is removed.

**MOD-02 Current implementation.** The zero-expiry and zero-vol limits of MOD-03 are
implemented by masking and tested (AT-08). At `expiry=0` the discounted-forward intrinsic
equals the undiscounted intrinsic because the discount factor is 1.

**MOD-03 Build requirement.** Define zero-expiry and zero-vol limits: price equals intrinsic
(undiscounted at `expiry=0`, discounted forward intrinsic at `vol=0`) with finite gradients.
Implement by masking `vol * sqrt(expiry)` to a safe value before the division and selecting
the limit with `torch.where`, per TEN-06.

**MOD-04 Confirmed requirement.** One entry point, as in TF Quant Finance:
`sq.black_scholes.option_price(*, strike, expiry, is_call, vol=None, normal_vol=None,
spot=None, forward=None, rate=None, div=None, discount=None, dtype=None)`. Pass exactly one
of `vol` (lognormal) and `normal_vol` (Bachelier), and exactly one of `spot` (with `rate`,
optional `div`) and `forward` (with `discount` or `rate`, no `div`). `spot` with `vol` is
Black-Scholes-Merton (MOD-01 to MOD-03), `forward` with `vol` is Black-76, and `normal_vol`
is Bachelier. Invalid combinations raise `ValueError`. These choices are Python `None`
checks, never tests on tensor values.

**MOD-07 Confirmed requirement.** `sq.black_scholes.binary_price` (cash-or-nothing, same
inputs as `option_price`, lognormal or normal), `asset_or_nothing_price` (lognormal) and
`barrier_price` (Merton and Reiner-Rubinstein single barriers; `spot`, `div` and `rebate`
required, `kind` a string) live in `vanilla_prices.py`. Exotic closed forms live in
`sq.exotics.<family>_prices`.

**MOD-08 Confirmed requirement.** `sq.black_scholes.approximations.adesi_whaley`
(Barone-Adesi and Whaley 1987) and `bjerksund_stensland(version=1993 | 2002)` give fast,
differentiable American prices. They match QuantLib's approximation engines and serve as a
cross-check on the tree, PDE and Longstaff-Schwartz prices (AT-25). The critical-price
solve differentiates by the implicit function theorem (GRK-08).

**MOD-05 Confirmed requirement.** `sq.models.sabr.approximations.implied_vol(*, forward,
strike, expiry, alpha, beta, rho, nu)` (Hagan 2002) and `european_option_price` (Black-76 at
the SABR vol); `sq.models.heston.characteristic_function(u, *, spot, rate, div, expiry, v0,
kappa, theta, sigma, rho)` and `sq.models.heston.approximations.european_option_price`
(Attari 2004 single integral, `num_points=1024`); and `sq.models.merton.price(*, spot,
strike, expiry, rate, vol, div, is_call, jump_intensity, jump_mean, jump_vol, num_terms=64,
dtype=None)`. All accept `dtype=`.

**MOD-06 Confirmed requirement.** Parameter dataclasses `BlackScholesParams`,
`Black76Params`, `BachelierParams`, `MertonParams`, `HestonParams`, `SABRParams`,
`LocalVolParams` and `HullWhiteParams` hold tensors and are registered with
`torch.utils._pytree` so `torch.func` and `torch.compile` see through them.

**MOD-10 Confirmed requirement.** Each model module also exposes simulation dynamics as a
`step(...)` function consumed by the Monte Carlo engine. GBM, arithmetic Brownian motion,
Heston (QE), Merton, local vol and Hull-White have one; CEV and SABR do not. (Numbered MOD-07
before version 1.1, which duplicated an identifier.)

**MOD-11 Build requirement.** SABR at-the-money uses the series limit of `z / x(z)` so the
formula and its gradient are finite at `strike == forward`. Heston's characteristic
function uses the formulation that avoids the branch-cut discontinuity (Albrecher et al.
"little Heston trap"). Both are implemented. (Numbered MOD-08 before version 1.1.)

**MOD-09 Confirmed requirement.** Every formula docstring cites its source (book, paper,
equation number).

## 9 Greeks and differentiation

**GRK-01 Confirmed requirement.** `sq.greeks.first_order(price, inputs, *, sum_batch=True)`
returns `dict[str, Tensor]`. With `sum_batch=True` each gradient is of `price.sum()`:
per-option for inputs batched one-to-one with prices, portfolio-aggregate for shared inputs.

**GRK-02 Decided (in code, recorded 19 November 2026).** `sum_batch=False` raises
`ValueError` for any shared input, including a scalar input against a batched price, and
points to `jacobian`. The default is `True`.

**GRK-03 Confirmed requirement.** `second_order(price_fn, inputs, pairs=None)` returns
`dict[tuple[str, str], Tensor]` using forward-over-reverse
(`torch.func.jacfwd(torch.func.grad(f))`). `pairs=None` returns all unique pairs.

**GRK-04 Confirmed requirement.** `jacobian(price_fn, inputs)` returns per-option
sensitivities for shared inputs. `bump(price_fn, inputs, *, rel=1e-4, scheme="central")`
exists for testing and user verification only.

**GRK-05 Confirmed requirement.** Named Greeks: delta = dV/dspot, gamma = d²V/dspot²,
vega = dV/dvol, theta = −dV/dexpiry, rho = dV/drate, vanna = d²V/dspot dvol,
volga = d²V/dvol². Units are per unit of the input (vega per 1.00 of vol, not per 1%). No
named-Greek helper exists at `fae8063`; callers use `first_order` and `second_order`.

**GRK-06 Confirmed requirement.** Monte Carlo pathwise Greeks come from autograd through
the estimator. Discontinuous payoffs support payoff smoothing (`smoothing="sigmoid"`,
`eps`), the likelihood ratio method (`greeks_method="likelihood_ratio"`) for models with
known densities, and one-step survival for discrete barriers (Glasserman 2004, §7.2).

**GRK-07 Build requirement.** Each discontinuous instrument states its default treatment and
the supported alternatives in its docstring. A pathwise delta of exactly zero on a digital or
barrier, with no treatment selected, emits a warning when validation is on. At `fae8063` the
warning exists for cash-or-nothing digitals only.

**GRK-08 Confirmed requirement.** Solvers (implied vol, roots, bootstrap, calibration)
attach implicit-function-theorem backward passes. `d implied_vol / d price = 1 / vega`
exactly, with no unrolled graph.

## 10 Implied volatility and surfaces

**VOL-01 Confirmed requirement.** `sq.black_scholes.implied_vol(price, *, strike, expiry,
is_call=True, spot=None, forward=None, rate=None, div=None, discount=None,
underlying_distribution=UnderlyingDistribution.LOGNORMAL,
method=ImpliedVolMethod.NEWTON, tol=None, max_iter=50, dtype=None)` solves a whole chain in
one call. Inputs follow `option_price` (MOD-04); `underlying_distribution=NORMAL` gives
Bachelier volatility. Defect at `fae8063`: `tol` is passed to the solver and ignored.

**VOL-02 Confirmed requirement.** `method=NEWTON` (default): rational initial guess, then
Newton safeguarded by bisection, under the fixed-iteration masked loop of MTH-02.
`method=FAST_APPROX`: the one-shot Polya-approximation formula of Stefanica and Radoicic
(2017), no iterations, lognormal only, for very large chains or as a starting point. Both
are also callable directly as `implied_vol_newton` and `implied_vol_approx`.

**VOL-03 Build requirement.** Prices outside no-arbitrage bounds (below intrinsic, above the
discounted underlying) return NaN for those elements. Prices at intrinsic return zero vol.
The behaviour is documented and tested, not left to the solver. Newton does this for the
lognormal case; the normal case has only a lower bound, and `FAST_APPROX` returns `inf`
rather than NaN at the upper bound.

**VOL-04 Decided (default, 19 November 2026).** Newton keeps the Brenner-Subrahmanyam initial
guess, clamped to [1e-8, 5]. Let's Be Rational is not built; it is reconsidered only if a
benchmark shows Newton's iteration count matters.

**VOL-05 Confirmed requirement.** Surfaces are functions, not classes:
`svi_total_variance(log_moneyness, a, b, rho, m, sigma)`, `svi_implied_vol`,
`ssvi_total_variance(log_moneyness, theta, rho, phi)`, `ssvi_implied_vol` and
`dupire_local_vol(price_fn, *, spot, times, strikes, rate, div)`, all differentiable in
their parameters. Defect at `fae8063`: Dupire's second strike derivative is taken without
`create_graph`, so local vol is not differentiable through it.

**VOL-06 Build requirement.** Provide no-arbitrage checks for SVI/SSVI (butterfly and
calendar conditions) as functions returning masks, used by calibration penalties and tests.
`svi_butterfly_ok` and `ssvi_calendar_ok` exist; the SSVI butterfly and SVI calendar checks
do not.

## 11 Instruments

**INS-01 Confirmed requirement.** Instruments are frozen dataclasses holding contract terms
with a `payoff(paths, times) -> Tensor` method returning undiscounted cash flows per path.
They contain no pricing logic.

**INS-02 Confirmed requirement.** Instruments in `sq.instruments`:
`EuropeanOption(strike, expiry, is_call)`;
`DigitalOption(strike, expiry, is_call, cash, smoothing=None, eps=None)`;
`BarrierOption(strike, expiry, barrier, is_call, kind, rebate=None, monitoring="discrete",
vol=None, discrete_correction=False)`;
`AsianOption(..., fixing_times=None, average="arithmetic" | "geometric",
sampling="discrete" | "continuous")`;
`LookbackOption(expiry, is_call, strike=None, extremum=None)`, floating when `strike` is
`None`; `AmericanOption(..., exercise_grid=None)`; and, analytic only,
`ForwardStartOption`, `CompoundOption`, `ChooserOption`, `CliquetOption` and
`DoubleBarrierOption` (`knock_in` / `knock_out`, also Monte Carlo). Analytic-only
instruments raise `NotImplementedError` from `payoff`.

**INS-03 Build requirement.** Instruments declare their observation and exercise times so
the Monte Carlo engine can merge them into its grid (MC-02), and declare the running
statistics they need (running max, running sum, hit flag) so full paths need not be stored
(MC-08). Instruments expose `observation_times` / `exercise_times`, which the engine merges
with its equal steps (MC-02). No running statistics are declared.

**INS-04 Decided (in code, recorded 19 November 2026).** Barrier kinds are `down_in`,
`down_out`, `up_in` and `up_out`; the `_and_` spellings (`"down_and_out"`) are accepted and
normalised. Monitoring is `"discrete"` (default), `"survival"` (one-step survival,
knock-out only) or `"continuous"`. The Broadie-Glasserman-Kou shift is opt-in with
`discrete_correction=True` and needs `vol`. The analytic engine prices only
`monitoring="continuous"`, so `sq.price` on a default barrier raises and asks for an engine.

**INS-05 Build requirement.** Contract terms may be tensors so a batch of strikes or
barriers prices in one call, following TEN-01. Analytic, PDE, Fourier and Lattice broadcast;
Monte Carlo payoffs assume scalar terms, `ForwardStartOption` calls `.item()`, and the Monte
Carlo grid reads the first expiry only.

## 12 Engines and the price entry point

**ENG-01 Confirmed requirement.** Every engine implements `engine.price(instrument,
model_params, market) -> Tensor`. `market` is `Market(spot, rate=None, div=None,
discount=None, curve=None)`; a curve is read through `zero_rate(expiry)`.

**ENG-02 Confirmed requirement.** Engines declare the (model, instrument) pairs they support
and raise `NotImplementedError` naming the pair otherwise.

**ENG-03 Confirmed requirement.** `sq.price(instrument, model_params, market, engine=None)`
uses `Analytic` when it supports the pair and otherwise raises, asking for an explicit
engine. It never silently falls back to Monte Carlo.

**ENG-04 Confirmed requirement.** Engines and their defaults at `fae8063`:
`Analytic(num_terms=128, fourier_terms=256)` (Merton series length, Heston COS terms);
`MonteCarlo(num_paths=8192, seed=1, antithetic=True, brownian_bridge=False,
chunk_size=None, moment_match=False, greeks_method="pathwise", degree=2, steps=50)`;
`PDE(num_space=200, num_time=200, theta=0.5, rannacher_steps=2)`;
`Fourier(method="cos", num_terms=256)`; `Lattice(num_steps=200, kind="crr")`. The engine
defaults win over the larger values in earlier versions of this specification.

**ENG-05 Confirmed requirement.** Each engine holds a class-level `registry` keyed by
(instrument type, params type), so adding a pair does not require editing dispatch code and
ENG-02 errors list what is supported. A public `register` function is still to be added.

## 13 Monte Carlo

**MC-01 Confirmed requirement.** Paths have shape `[num_paths, num_steps + 1, dim]`, are
simulated in log space where the model allows, and keep a trailing `dim` of 1 for one-factor
models.

**MC-02 Confirmed requirement.** The time grid is the sorted merge of instrument
observation/exercise times with an optional maximum step size. `MonteCarlo._times` merges
the instrument's dates with `steps` equal steps to expiry, dropping an equal-step point
within `1e-12 * expiry` of an instrument date. A discrete barrier is monitored on every grid
date; an Asian with `fixing_times` averages on its fixing dates only; an American option
with `exercise_grid` exercises on those dates only, and without one on every grid date. A
maximum step size option is still to be added.

Fixed after `fae8063`: the engine used to return the instrument's `[0, T]` grid and ignore
`steps`, so every run simulated one step and a discrete barrier was checked only at 0 and
T. The discrete barrier and double-barrier payoffs also raised `TypeError` (a Python `bool`
passed to `torch.where`), so Monte Carlo could not price them at all. AT-30 covers both.

**MC-03 Confirmed requirement.** Default schemes: exact lognormal (Black-Scholes), exact
normal (Bachelier), exact step with Poisson jump counts (Merton), Andersen QE with
martingale correction (Heston), log-Euler with surface interpolation (local vol), exact
Gaussian step of rate and integral (Hull-White). Euler is available for every model as a
reference. At `fae8063` the engine wires exact GBM, Heston QE (without the martingale
correction) and local vol; Bachelier, Merton and Hull-White steps exist with no engine pair;
Merton's `torch.poisson` ignores the seed; there is no Euler reference.

**MC-04 Confirmed requirement.** Pseudo-random numbers use a `torch.Generator` on the target
device seeded by `seed=`. Same seed, device and dtype give the same paths.

**MC-05 Confirmed requirement.** Quasi-random numbers use scrambled
`torch.quasirandom.SobolEngine` mapped through the inverse normal CDF with a Brownian bridge.
Error estimates use several independent scramblings (randomised QMC). Sobol and the bridge
exist in `math/random.py`; the engine cannot select Sobol and there is no randomised-QMC
error estimate.

**MC-06 Confirmed requirement.** Variance reduction: antithetic variates (default on),
control variates with the coefficient estimated from the same paths, optional moment
matching. Control variates exist in `engines/monte_carlo.price` but not on the `MonteCarlo`
engine; moment matching draws every path at once and ignores `chunk_size`.

**MC-07 Confirmed requirement.** `chunk_size=` simulates and prices in chunks, accumulating
payoff sums and gradients. Results are identical to the unchunked run with the same seed.
At `fae8063` chunked prices equal unchunked prices, but per-path values are concatenated, so
the autograd graph of every chunk is kept; the American engine ignores `chunk_size`.

**MC-08 Confirmed requirement.** Payoffs needing only running statistics are computed on
the fly. `torch.utils.checkpoint` over time blocks is available for Greeks on very long
paths. Neither exists at `fae8063`.

**MC-09 Confirmed requirement.** `return_stderr=True` returns `(price, stderr)`. It exists on
`engines/monte_carlo.price` but not on the `MonteCarlo` engine or `sq.price`.

**MC-10 Confirmed requirement.** Chunked runs draw the same random numbers as unchunked runs.
Pseudo-random normals come in fixed 4,096-path blocks replayed from the seed; Sobol uses
`fast_forward`.

**MC-11 Decided (default, 19 November 2026).** Reproducibility is same-device only, as in
MC-04. Sobol paths do not depend on the device.

## 14 PDE, Fourier and lattice engines

**PDE-01 Confirmed requirement.** One-dimensional Crank-Nicolson finite differences with
Rannacher start-up steps, using the batched tridiagonal solver.

**PDE-02 Confirmed requirement.** American options use a projected Crank-Nicolson step. At
`fae8063` the engine applies `max(value, payoff)` after each linear solve; a linear
complementarity solve (Brennan-Schwartz or PSOR) is still to be built.

**PDE-03 Build requirement.** Grids are non-uniform and concentrated around the strike and
barriers; Greeks come from autograd through the solver and must be smooth in spot (no grid
staircase). Test against closed forms per AT-12. At `fae8063` the sinh log grid
concentrates around the strike only; barriers are imposed by masking nodes rather than
placed on a node; a spline read-out keeps Greeks smooth; `monitoring` is ignored.

**FOU-01 Confirmed requirement.** Characteristic-function pricing with COS (default) and
Carr-Madan for Heston and other affine models. Both broadcast over batches of strikes,
expiries and parameters, with the integration grid as a trailing dimension.
Characteristic functions exist for Heston and Black-Scholes only.

**FOU-02 Build requirement.** COS prices the put and gets the call by parity, so wide
truncation ranges cannot blow up through ``exp(b)``. The range is asymmetric: the left
edge uses Fang and Oosterlee's fat-tail rule ``c1 - 12 sqrt(c2 + sqrt(c4))``, the right
edge ``c1 + 12 sqrt(c2)``. Error falls with ``num_terms`` for every tested parameter set;
vol of vol near 1.2 at long expiries needs about 1,024 terms. Carr-Madan integrates in
forward moneyness ``log(K/F)`` with a trapezoid rule of step 0.1, whose error falls like
``exp(-2 pi alpha / step)``; short expiries need ``num_terms`` in the thousands to reach
the tail of the integrand.

**LAT-01 Confirmed requirement.** CRR binomial and trinomial trees, European and American,
in `sq.black_scholes.option_price_binomial`; the `Lattice` engine wraps it (default 200
steps). Inputs broadcast, with tree nodes as a trailing dimension.

**LSM-01 Confirmed requirement.** Longstaff-Schwartz in the Monte Carlo engine. The
regression boundary is estimated, then held fixed under `torch.no_grad` when computing
Greeks (envelope theorem). `exercise_index` takes an `exercisable` mask over the grid, so a
Bermudan exercise grid is honoured on a finer simulation grid.

**LSM-02 Decided (in code, recorded 19 November 2026).** Monomials `1, x, ..., x^degree` in
spot with `degree=2`, fitted on in-the-money paths. The same paths fit the boundary and price
the option; an independent fitting set is not used.

## 15 Calibration

**CAL-01 Confirmed requirement.** `calibrate(model_params, market_quotes, pricer, *,
loss="vol_rmse", optimizer="lbfgs", bounds=None, max_iter=200, tol=1e-8)` returns a
`CalibrationResult(params, loss, converged, iterations, grad_norm)`. `bounds=None` leaves
parameters unconstrained. Defect at `fae8063`: `"vol_rmse"` and `"price_rmse"` compute the
same RMSE of the pricer output; `"vol_rmse"` must convert prices to implied vols first.

**CAL-02 Confirmed requirement.** Bounds are handled by reparameterisation: softplus for
positive parameters, `tanh` for correlations. Standard unconstrained `torch.optim`
optimisers apply.

**CAL-03 Confirmed requirement.** Returned parameters carry a gradient path to the market
quotes through the implicit function theorem at the optimum (a dense Hessian of the loss
with a 1e-8 ridge), so parameter sensitivities to each quote are available when the quotes
require grad.

**CAL-04 Build requirement.** Record iteration count, final gradient norm and the reason for
stopping. `converged=False` is a normal return, not an exception. At `fae8063` `iterations`
always reports `max_iter` and there is no stop reason.

## 16 Rates

**RTS-01 Confirmed requirement.** `DiscountCurve(times, discount_factors,
interpolation="log_linear" | "linear")` with `discount(t)`, `zero_rate(t)` and
`forward_rate(t1, t2)`; `ZeroCurve(times, zero_rates, interpolation)` as a subclass.

**RTS-02 Confirmed requirement.** `bootstrap(*, pay_times, rates, kinds, deltas)` is
differentiable with respect to quotes, so bucketed rate deltas come straight from autograd.
A deposit is a continuously compounded rate; a swap pays fixed on every earlier pillar.
`par_swap_rate` gives the inverse.

**RTS-03 Confirmed requirement.** Hull-White one-factor: bond prices, caps/floors,
swaptions by Jamshidian decomposition, and exact simulation. At `fae8063` there are
`bond_price`, `zero_bond_option`, `caplet` (no floorlet, no cap or floor as a sum),
`swaption` (Jamshidian, unbatched) and an exact `step`; `HullWhiteParams` is in no engine and
the module has no tests.

**RTS-04 Decided (in code, recorded 19 November 2026).** Callers pass accrual year fractions
and payment times (PRD-06). An optional adapter in `smallquant.time` that builds them from a
`Schedule` may be added later.

## 17 Dates, day counts and calendars

**TIM-01 Confirmed requirement and current implementation.** `smallquant.time` is a pure
Python port of QuantLib's date machinery with no torch dependency. It provides `Date`,
`Month`, `Weekday`, `Period`, `TimeUnit`, `Frequency`; the day counters `Actual360`,
`Actual366`, `Actual36525`, `Actual365Fixed`, `Thirty360` (nine conventions),
`ActualActual` (seven conventions, ISMA with `schedule=`), `SimpleDayCounter`,
`OneDayCounter` and `Business252(calendar)`; `BusinessDayConvention`, `Calendar`,
`Schedule`; and the calendars `NullCalendar`, `WeekendsOnly`, `TARGET`, `UnitedStates`
(seven markets), `UnitedKingdom` (three), `Japan`, `Australia` (Settlement, ASX),
`JointCalendar` and `BespokeCalendar`.

**TIM-02 Confirmed requirement.** These differences from QuantLib are intentional and must
not be "fixed": snake_case names, `None` for a missing reference period, `Period.__eq__`
returning `False` for undecidable comparisons, `ValueError` for errors, per-instance
`add_holiday` / `remove_holiday`, keyword-only `convention` and `end_of_month` on
`advance`. This list replaces the removed `docs/time.md`.

**TIM-03 Current implementation.** Tests compare against QuantLib over every serial number
in range, two years of daily `Date ± Period`, every day counter on edge grids and random
pairs (relative 1e-14), and calendar holiday grids from 1950 to 2035.

**TIM-04 Build requirement.** `Schedule` (rules Backward, Forward, Zero, ThirdWednesday,
Twentieth, TwentiethIMM and CDS), schedule-driven `ActualActual` ISMA and `Business252` are
done. Still to build: stub detection (`is_regular` is always true), `first_date` and
`next_to_last_date`, and `Schedule` tests beyond the single TARGET case. Additional country
calendars are added on request, each with a QuantLib holiday-grid test.

**TIM-05 Confirmed requirement.** The QuantLib reference version is the one `uv.lock`
resolves (1.44 at `fae8063`). A failure after a QuantLib upgrade is a port to update.

**TIM-06 Decided (19 November 2026).** Australia (Settlement and ASX) is ported and
holiday-grid tested. Further calendars are added on request.

## 17A Market risk

Market risk measures how much a portfolio's value can change when market prices move. In
SmallQuant every risk number is a function of prices that the engines already compute, so
sensitivities come from one backward pass and scenario revaluation is one batched pricing
call. Inputs (positions, scenarios, historical factor changes) are tensors the caller
supplies (PRD-05, PRD-09). The module is `sq.risk.market`.

**MKR-01 Confirmed requirement.** A `Portfolio` is a list of positions, each an
(instrument, quantity, model params, engine) tuple. Positions of the same instrument and
params type price in one batched call. Portfolio value is the quantity-weighted sum of
position prices, as a tensor.

**MKR-02 Confirmed requirement.** Market state is a `RiskFactors` mapping from factor name
to tensor: spots, volatilities (flat or surface nodes), rate-curve pillars, dividend yields,
and credit spreads or hazard pillars (section 17B). Positions read their `Market` and model
parameters from it, so every factor is a leaf of the autograd graph.

**MKR-03 Confirmed requirement.** `sensitivities(portfolio, factors, *, order=1 | 2)` returns
the first-order sensitivity of portfolio value to every factor (delta, vega, bucketed rate
delta per curve pillar, dividend and credit sensitivities) and, with `order=2`, gamma and
cross-gamma, from autograd (GRK-01 to GRK-04). Units follow GRK-05. A bump check is
available through `greeks.bump`.

**MKR-04 Confirmed requirement.** `revalue(portfolio, factors, shocks)` applies a batch of
shocks shaped `[num_scenarios, ...]` per factor, absolute or relative as declared per
factor, and returns the profit and loss per scenario, shaped `[num_scenarios]`, by full
revaluation. The scenario dimension leads and is batched through the engines; there is no
Python loop over scenarios.

**MKR-05 Confirmed requirement.** Historical simulation: from a caller-supplied window of
historical factor changes, `historical_var(...)` and `historical_es(...)` return value at
risk and expected shortfall. Defaults: 99% VaR, 97.5% ES, one-day horizon. VaR is the
negative of the k-th worst P&L with k = ceil(n · (1 − confidence)); ES is the negative mean
of the losses at or beyond that quantile. Horizon scaling by the square root of time is
available and off by default.

**MKR-06 Confirmed requirement.** Parametric VaR from sensitivities and a factor covariance
matrix: delta-normal, and delta-gamma by Cornish-Fisher expansion or by Monte Carlo of the
quadratic form.

**MKR-07 Confirmed requirement.** Monte Carlo VaR and ES: simulate factor changes from a
covariance matrix (Cholesky, `math/linalg.py`) or a caller-supplied sampler, revalue in full
(MKR-04), and report the result with a standard error (section 3, "Monte Carlo results").

**MKR-08 Confirmed requirement.** Stress testing: named deterministic scenarios are shock
tensors; output is the P&L per scenario and per position.

**MKR-09 Confirmed requirement.** Risk attribution: Euler allocation of VaR and ES to
positions (component VaR and component ES) through autograd with respect to position
quantities. Components sum to the total.

**MKR-10 Confirmed requirement.** P&L explain: the Taylor P&L from first- and second-order
sensitivities is compared with full revaluation for each scenario, and the unexplained
residual is reported.

**MKR-11 Build requirement.** ES is differentiable with respect to quantities and market
inputs. VaR's gradient is the gradient of the scenario that sets the quantile; the
docstring states this convention, because the quantile is not smooth.

**MKR-12 Confirmed requirement.** Backtesting: given a VaR series and realised P&L, report
exception counts and Kupiec's proportion-of-failures test.

**MKR-13 Scope boundary.** No regulatory capital templates (FRTB standardised approach,
Basel traffic-light reporting) and no market-data sourcing. Callers can build them on the
sensitivities and scenario P&L above.

## 17B Credit risk

Credit risk covers the chance that a counterparty or issuer defaults: pricing default-linked
contracts, measuring exposure to counterparties, and the loss distribution of a credit
portfolio. The module is `sq.risk.credit`. The core works in year fractions (PRD-06); an
optional adapter builds CDS schedules with `smallquant.time.Schedule` and its CDS rule.

**CRD-01 Confirmed requirement.** `SurvivalCurve(times, hazard_rates)` with piecewise-constant
hazard: `survival(t)`, `default_probability(t1, t2)` and `hazard(t)`, all tensors and
differentiable with respect to the pillars.

**CRD-02 Confirmed requirement.** Credit default swaps: premium leg with accrual on default,
protection leg with recovery, par spread, upfront and risky annuity (RPV01), following the
ISDA CDS Standard Model's piecewise-flat integration. Batched over contracts.

**CRD-03 Confirmed requirement.** `bootstrap_survival(...)` builds a `SurvivalCurve` from CDS
par spreads or upfronts. It is differentiable with respect to the quotes, through closed
forms or the implicit function theorem (GRK-08), so CS01 per pillar comes from autograd.

**CRD-04 Confirmed requirement.** Risky zero-coupon and fixed-coupon bonds priced from a
discount curve, a survival curve and a recovery rate.

**CRD-05 Confirmed requirement.** Exposure simulation: simulate risk factors on a date grid
with the Monte Carlo models (GBM, Heston, Hull-White), revalue each position at each date
with a closed form where one exists and otherwise by regression (Longstaff-Schwartz basis,
LSM-02), and return exposures shaped `[num_paths, num_dates]`. From them: expected exposure
(EE), expected positive and negative exposure (EPE, ENE), effective EPE and potential future
exposure (PFE, default quantile 97.5%).

**CRD-06 Confirmed requirement.** Netting and collateral: positions in a netting set are
summed before the positive part is taken. Collateral supports a threshold, a minimum transfer
amount and a margin period of risk (collateral lagged by that period).

**CRD-07 Confirmed requirement.** Credit valuation adjustment: unilateral CVA =
(1 − recovery) · Σ discounted EE(tᵢ) · PD(tᵢ₋₁, tᵢ), and bilateral CVA with DVA from the
own survival curve. Default is independent of exposure. CVA sensitivities to market factors
and to credit pillars (CVA delta, CS01) come from autograd through the whole simulation.

**CRD-08 Confirmed requirement.** Portfolio credit loss: the one-factor Gaussian copula
(Vasicek) with probability of default, loss given default, exposure at default and asset
correlation per obligor. Expected loss, credit VaR and ES by Monte Carlo, plus the
large-pool closed form and the asymptotic single risk factor formula.

**CRD-09 Build requirement.** Wrong-way risk (correlation between exposure drivers and
default intensity), rating migration and multi-name credit derivatives (CDO tranches, nth to
default) are sequenced after CRD-01 to CRD-08 and specified when that work starts.

## 18 Public API surface and packaging

**API-01 Confirmed requirement.** `import smallquant as sq` exposes `sq.math`, `sq.rates`,
`sq.vol`, `sq.black_scholes`, `sq.exotics`, `sq.models`, `sq.instruments`, `sq.payoffs`,
`sq.engines`, `sq.greeks`, `sq.calibration`, `sq.calibrate`, `sq.price`, `sq.Market`,
`sq.set_default_dtype`, `sq.default_dtype`, `sq.set_validation` and `sq.validation` (all
present at `fae8063`), and `sq.risk` once sections 17A and 17B are built.

**API-02 Current implementation.** `smallquant/__init__.py` and `smallquant/models/__init__.py`
import their submodules, so the API-01 names resolve.

**API-03 Build requirement.** `smallquant.time` is not imported by `smallquant/__init__.py`
(PRD-06); users import it explicitly. Everything not listed in a module's `__all__` is
private.

**API-04 Current implementation.** `pyproject.toml` uses the hatchling build backend and CI
runs `uv build`. Tests still set `pythonpath = ["."]`.

**API-05 Build requirement.** Version the package with semantic versioning starting at
`0.1.0` for the Phase 0 release; ship the `py.typed` marker (added). Breaking changes before
`1.0` are allowed but recorded in `CHANGELOG.md` (added).

**API-06 Decided in part (19 November 2026).** The licence is MIT (`LICENSE`, `pyproject.toml`).
Still required: the PyPI distribution name (`smallquant` availability) and who holds
publishing rights for the SmallFinance organisation (D01).

## 19 Errors and validation

**VAL-01 Confirmed requirement.** Shape, dtype and device mismatches are always checked.

**VAL-02 Confirmed requirement.** Value checks (`vol > 0`, `expiry >= 0`, probabilities in
range) require a host sync and are off by default; enabled with `sq.set_validation(True)` or
`SMALLQUANT_VALIDATE=1`.

**VAL-03 Build requirement.** Value-check failures raise `ValueError` naming the argument and
the count of offending elements. Validation must not change results when inputs are valid.

**VAL-04 Build requirement.** `set_validation` and `set_default_dtype` are process-global.
Also provide a context manager form for tests, and make the state safe to read under
`torch.compile` (resolved at trace time and documented as such).

## 20 Performance and compilation

**PRF-01 Confirmed requirement.** Core functions run under `torch.compile(fullgraph=True)`
without graph breaks and match eager results.

**PRF-02 Confirmed requirement.** `benchmarks/` holds on-demand timing scripts: chain
pricing, implied vol over a surface, Heston Monte Carlo with Greeks, and (once built) a
portfolio VaR revaluation and a CVA run. They do not run in CI. At `fae8063` the only timing
script is `scripts/european_mc.py`.

**PRF-03 Confirmed requirement.** Documentation quotes only measured numbers, labelled with
hardware and library versions. `docs/european-monte-carlo.md` records the hardware but not
the library versions.

**PRF-04 Build requirement.** Each benchmark reports a bump-and-reprice baseline next to the
autograd result, since that comparison is the library's main claim.

## 21 Documentation

**DOC-01 Confirmed requirement.** Public functions have type hints and Google-style
docstrings with `Args`, `Returns` and a short `Example`. Formulas cite sources.

**DOC-02 Build requirement.** Every code example in `docs/` and `README.md` runs as a test
(doctest or an example-runner test). That would have caught API-02.

**DOC-03 Confirmed requirement.** The root `README.md` has installation and a working quick
look, and links this specification and the Monte Carlo timings. Its examples are covered by
DOC-02.

**DOC-04 Confirmed requirement.** A documentation site built from the Markdown files is a
later, unscheduled item.

## 22 Verification and acceptance

**QA-01 Build requirement.** Convert each requirement ID into a traceable test or review
check. Seven test kinds apply to every feature: published
references, independent implementation (QuantLib, SciPy), no-arbitrage identities,
cross-engine agreement, Greeks, batching and devices, compilation.

**QA-02 Confirmed requirement.** Tolerances: closed forms in `float64` relative 1e-10 against
full-precision references and absolute 1e-4 against four-decimal published values; `float32`
relative 1e-5; Monte Carlo within 3 standard errors with a fixed seed.

**QA-03 Current evidence.** On 19 November 2026 at `fae8063` the full suite ran on CPU: 541
passed in 180 s. Black-Scholes tests use relative 1e-9 / absolute 1e-10 against QuantLib
(D11: kept, see Appendix B). There is no `conftest.py` and no device or dtype
parametrisation; `gradcheck` appears in two files and `gradgradcheck` nowhere; there is no
`torch.compile` test; `smallquant/models/hull_white.py` and `calibrate` with a real model
have no tests. CUDA and MPS have not been exercised by the suite.

Status is at `fae8063` except where a row says otherwise: **met**, **partial** (what is missing is in section 4 or the
requirement) or **open**.

| Test ID | Minimum acceptance scenario | Status |
| --- | --- | --- |
| AT-01 | `import smallquant as sq` exposes every API-01 name; every docs and README example runs. | Partial: 12 names tested, no example test |
| AT-02 | Passing a Python `float`/`int` (or other non-tensor) as a public numeric argument raises; `float32` + `float64` tensors give `float64`; `dtype=` overrides. | Partial: tested in seven modules, not across engines, exotics or math |
| AT-03 | Mixed-device inputs raise; outputs land on the input device; MPS without `float64` warns once and returns `float32`. | Partial: CPU and meta mix only |
| AT-04 | A batched call equals the stack of individual calls for every public pricing function. | Partial: not exotics, Monte Carlo, Fourier or implied vol |
| AT-05 | `gradcheck` and `gradgradcheck` pass in `float64` for every differentiable public function. | Partial: two files |
| AT-06 | `option_price` in its Black-Scholes, Black-76 and Bachelier forms, plus `binary_price` and `asset_or_nothing_price`, match QuantLib and published values for price and first- and second-order Greeks, calls and puts. | Partial: Greeks for Black-Scholes only |
| AT-07 | Put-call parity, monotonicity and convexity in strike, and price bounds hold over random inputs for every closed form. | Open |
| AT-08 | Zero expiry and zero vol return intrinsic values with finite gradients (MOD-03). | Met |
| AT-09 | `implied_vol` (Newton) round-trips prices across a strike-expiry grid to 1e-10 relative; its gradient equals `1 / vega`; out-of-bounds prices return NaN. | Partial: one point at 1e-8 |
| AT-10 | `greeks.first_order`, `second_order` and `jacobian` agree with closed-form Greeks and with `bump` central differences. | Partial: `jacobian` untested |
| AT-11 | SABR at `strike == forward` has finite price and gradient; Heston characteristic function matches a reference across long expiries without branch jumps. | Partial: no direct long-expiry characteristic-function reference |
| AT-12 | European options priced by Analytic, PDE, Fourier, Lattice and Monte Carlo agree: deterministic engines to a stated tolerance, Monte Carlo within 3 standard errors. | Met for one contract |
| AT-13 | Monte Carlo with the same seed, device and dtype is bit-identical; chunked and unchunked runs agree exactly. | Met on CPU (chunk test spans one block) |
| AT-14 | Sobol with Brownian bridge reports a randomised-QMC error estimate, and its error is below pseudo-random error at equal path count on a benchmark Asian. | Open |
| AT-15 | Digital and barrier Monte Carlo deltas with smoothing, likelihood ratio and one-step survival agree with closed forms within tolerance; untreated pathwise delta is flagged. | Partial: smoothed digital only |
| AT-16 | American put by Longstaff-Schwartz, projected Crank-Nicolson and a 2,000-step CRR tree agree to a stated tolerance; Greeks are finite. | Partial: no Longstaff-Schwartz test |
| AT-17 | Heston and SABR calibrate to synthetic surfaces generated from known parameters and recover them; parameter sensitivities to quotes match bumps. | Open |
| AT-18 | Bootstrapped curves reprice their input instruments to 1e-12; bucketed deltas match bumps. | Partial: 1e-10, deltas only checked finite |
| AT-19 | Hull-White bond and swaption prices match QuantLib. | Open |
| AT-20 | Core functions compile with `fullgraph=True` and match eager results. | Open |
| AT-21 | Validation on rejects invalid inputs with named errors; validation off never syncs the host (checked with a sync-detecting test hook). | Partial: no sync hook |
| AT-22 | `smallquant.time` suite stays green against the CI QuantLib version; `smallquant` core never imports it. | Partial: no import-boundary test |
| AT-23 | The wheel builds, installs into a clean environment with only PyTorch, and imports. | Partial: CI builds, no clean install |
| AT-25 | Barone-Adesi-Whaley and Bjerksund-Stensland 1993 match QuantLib's approximation engines; Bjerksund-Stensland 2002 matches published values or a 2,000-step tree; prices are at least European and gradients are finite. | Met |
| AT-26 | `implied_vol(method=FAST_APPROX)` stays within its stated error over a strike-expiry-vol grid, agrees across spot and forward inputs, returns NaN out of bounds, and is faster than Newton on a large chain. | Met |
| AT-27 | Heston `european_option_price` matches QuantLib's `AnalyticHestonEngine` and the `Fourier` engine across short and long expiries. | Met |
| AT-28 | `brownian_bridge_single` and `brownian_bridge_double` match fine-grid bridge simulation, converge in `n_cutoff`, stay in [0, 1], and agree with the survival probability in `payoffs.py`. | Met |
| AT-29 | `variance_swap_fair_strike` matches QuantLib's `ReplicatingVarianceSwapEngine` without dividends, reproduces TF's worked example, and converges to sigma squared on a dense flat-vol strip. | Met |
| AT-30 | A Monte Carlo discrete barrier with `steps=n` monitors at n dates: its price converges to the continuous closed form as n grows and matches QuantLib's biased `MCBarrierEngine` within 3 combined standard errors (MC-02). | Met: `tests/engines/test_monte_carlo.py` |
| AT-31 | Portfolio sensitivities (MKR-03) match `bump` central differences for every factor, including bucketed rate deltas. | Open |
| AT-32 | Historical VaR and ES on a fixed P&L vector equal a NumPy reference under the MKR-05 convention; delta-normal VaR of a linear portfolio equals Monte Carlo VaR within 3 standard errors. | Open |
| AT-33 | Euler component VaR and ES sum to the total; Taylor P&L explain leaves zero residual for a linear portfolio. | Open |
| AT-34 | Kupiec's test reproduces a published worked example. | Open |
| AT-35 | CDS par spread, upfront and RPV01 match QuantLib's `IsdaCdsEngine`; the bootstrap reprices its input CDS to 1e-12 and CS01 matches bumps. | Open |
| AT-36 | Expected exposure of a bought European option equals its forward value; its unilateral CVA equals (1 − recovery) × price × PD(0, T) under independence; netting and collateral reduce exposure as specified. | Open |
| AT-37 | The Vasicek large-pool loss distribution matches its closed form; Monte Carlo loss quantiles match within 3 standard errors. | Open |

**QA-04 Release gate.** No open defect in price correctness, Greek correctness, dtype or
device handling, or reproducibility. A passing happy-path example is not enough; each
release attaches the test run and the hardware it ran on.

## 23 Delivery plan and definition of done

Estimate by phase and identify numerical risks before
committing a date. Later phases remain in scope even though they are sequenced last.

| Phase | Concrete deliverables and exit evidence | Status at `fae8063` |
| --- | --- | --- |
| 0 Foundation | Installable package, CI on CPU, `ruff`, `conftest.py` device/dtype parametrisation; `_tensor.py`; `math/distributions` and `math/roots`; Black-Scholes limits on `_tensor.py`; Black76 and Bachelier; `sq.greeks`; `implied_vol`; package re-exports; MIT licence; changelog. Exit: AT-01–AT-10, AT-20–AT-23. Tag `0.1.0`. | Code done. Open: `conftest.py`, AT-07 and AT-20, and the partial exit tests |
| 1 Monte Carlo | `math/random` with Sobol and Brownian bridge; instruments; `engines/monte_carlo` with antithetics, control variates, chunking and standard errors; exact Black-Scholes and Bachelier simulation; Merton jump; discontinuous-payoff Greeks. Exit: AT-12 (MC), AT-13–AT-15, AT-30. | Built; Sobol, control variates and standard errors not on the engine |
| 2 Stochastic vol and calibration | Heston characteristic function, Fourier engine and QE simulation; SABR; SVI/SSVI; `calibrate`. Exit: AT-11, AT-17. | Built; no Heston or SABR calibration test |
| 3 PDE, lattices, early exercise | Tridiagonal solver; PDE engine; lattice engine; American options; Dupire local vol in Monte Carlo and PDE. Exit: AT-12 (all engines), AT-16. | Built; Longstaff-Schwartz untested, projected step approximate |
| 4 Rates | Curves, differentiable bootstrap, Hull-White. Exit: AT-18, AT-19. | Built; Hull-White untested |
| 5 Market risk | `sq.risk.market`: portfolio, risk factors, sensitivities, scenario revaluation, historical, parametric and Monte Carlo VaR and ES, stress tests, Euler attribution, P&L explain, backtesting (section 17A). Exit: AT-31–AT-34. | Not started |
| 6 Credit risk | `sq.risk.credit`: survival curves, CDS and bootstrap, risky bonds, exposure simulation, netting and collateral, CVA and DVA, Vasicek portfolio loss (section 17B). Exit: AT-35–AT-37. | Not started |
| Later | Multi-asset models, wrong-way risk, rating migration, multi-name credit (CRD-09), further calendars, documentation site. | — |

**DONE-01 Build requirement.** Each phase delivers version-controlled source, tests, updated
docs, a changelog entry and a tagged release. CI is green on the tag.

**DONE-02 Build requirement.** Benchmarks for the phase's headline outcome are run once and
recorded with hardware and versions (PRF-03, PRF-04).

**DONE-03 Confirmed outcome.** A user can price vanilla and exotic options, calibrate Heston
or SABR, build rate and credit curves, measure portfolio market risk (sensitivities, VaR,
ES, stress P&L) and counterparty credit risk (exposure, CVA), on CPU or GPU, with exact
Greeks and risk sensitivities from autograd, without writing adjoint code or bump loops.

**DONE-04 Acceptance.** Ted Wong signs off each release against this specification and the
decision log. "No bugs" is not a testable guarantee; verified requirements, no open
release-gate defects and an owner for each known issue are.

## Appendix A Argument naming reference

| Quantity | Argument name | Units |
| --- | --- | --- |
| Spot price | `spot` | currency |
| Forward price | `forward` | currency |
| Strike | `strike` | currency |
| Time to expiry | `expiry` | years (year fraction) |
| Continuously compounded risk-free rate | `rate` | per year |
| Continuous dividend / carry yield | `div` | per year |
| Lognormal volatility | `vol` | per sqrt(year) |
| Normal (Bachelier) volatility | `normal_vol` | currency per sqrt(year) |
| Call/put flag | `is_call` | bool or bool tensor |
| Discount factor | `discount` | unitless |

New arguments follow the same pattern: full lowercase words, units stated in the docstring,
no single-letter names in public signatures.

## Appendix B Decisions requiring an owner

Each decision is either taken by the owner, already embodied in the code, or a default this
specification takes. Only D01's PyPI name and publishing rights still need an answer before
a public release.

| ID and owner | Decision | Status |
| --- | --- | --- |
| D01 Ted | Licence, PyPI name and publishing rights (API-06) | Licence decided: MIT (19 November 2026). PyPI name and publishing rights open |
| D02 Ted | CI hardware for CUDA and MPS (ENV-04) | Default taken: macOS CPU in CI, manual CUDA and MPS check per release |
| D03 Ted | Minimum supported PyTorch version (ENV-03) | Default taken: the locked version (2.14) until CI tests a lower one |
| D04 Ted | Meaning of `sum_batch` in `greeks.first_order` (GRK-02) | Decided in code: default `True`; `False` rejects shared inputs |
| D05 Ted | Black implied-vol initial guess (VOL-04) | Default taken: Brenner-Subrahmanyam; no Let's Be Rational |
| D06 Ted | Barrier kinds and continuous-monitoring correction (INS-04) | Decided in code: four kinds, Broadie-Glasserman-Kou opt-in |
| D07 Ted | Cross-device Monte Carlo reproducibility (MC-11) | Default taken: same-device only |
| D08 Ted | Longstaff-Schwartz basis and independent-path boundary (LSM-02) | Decided in code: monomials to degree 2, same paths |
| D09 Ted | How rate instruments receive schedules (RTS-04) | Decided in code: callers pass year fractions |
| D10 Ted | Next calendar to port (TIM-06) | Done: Australia |
| D11 Ted | Black-Scholes test tolerance 1e-9 or 1e-10 (QA-03) | Default taken: QuantLib comparisons keep relative 1e-9; QA-02's 1e-10 applies to full-precision references |
| D12–D15 | JAX exposure, version, timing and random numbers | Withdrawn: no JAX (PRD-08, 19 November 2026) |

Defaults taken here stand unless the owner overrides them; an override is recorded in this
table with its date.

## Appendix C Code and evidence map

| Area | Source landmarks to inspect |
| --- | --- |
| Package entry | `smallquant/__init__.py`; `_tensor.py` (alignment, dtype and validation state) |
| Black-Scholes | `smallquant/black_scholes/vanilla_prices.py` (`option_price`, `binary_price`, `asset_or_nothing_price`, `barrier_price`), `crr_binomial_tree.py`, `approximations/american_option.py`, `variance_swaps.py`, `brownian_bridge.py` |
| Implied vol | `smallquant/black_scholes/implied_vol_lib.py` (`implied_vol`, `ImpliedVolMethod`), `implied_vol_newton_root.py`, `implied_vol_approximation.py` |
| Exotics | `smallquant/exotics/*_prices.py`, one file per family |
| Model dynamics | `smallquant/models/`: `geometric_brownian_motion.py`, `arithmetic_brownian_motion.py`, `heston/`, `sabr/`, `merton.py`, `cev.py`, `local_vol.py`, `hull_white.py`, `params.py` |
| Instruments and engines | `smallquant/instruments.py`, `payoffs.py`, `market.py`, `price.py`; `engines/pricing.py` (engine classes and registries; `MonteCarlo._times` builds the MC-02 grid), `monte_carlo.py`, `finite_difference.py`, `fourier.py`, `longstaff.py` |
| Math | `smallquant/math/`: `distributions.py`, `roots.py`, `interp.py`, `quadrature.py`, `tridiagonal.py`, `linalg.py`, `random.py` |
| Greeks, surfaces, calibration | `smallquant/greeks.py`, `vol/surface.py`, `calibration.py` |
| Rates | `smallquant/rates/curve.py`, `bootstrap.py`; `models/hull_white.py` |
| Dates | `smallquant/time/`: `date.py`, `period.py`, `daycounter.py`, `calendar.py`, `calendars.py`, `schedule.py` |
| References | `tests/reference/ql.py` (QuantLib), `tests/reference/tff.py` (TF Quant Finance) |
| Tests | `tests/black_scholes/` (180), `tests/time/` (175), `tests/exotics/` (79), `tests/engines/` (56), `tests/models/` (30), `tests/test_pricing_surface.py` (19), `tests/math/` (2) |
| Timing | `scripts/european_mc.py`, `docs/european-monte-carlo.md` |
| Tooling | `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `LICENSE`, `CHANGELOG.md` |

## Appendix D Reference sources

| Ref | Source and relevance |
| --- | --- |
| S1 | Hull, *Options, Futures, and Other Derivatives*: Black-Scholes reference value 10.4506 |
| S2 | Glasserman, *Monte Carlo Methods in Financial Engineering* (2004): pathwise and likelihood-ratio Greeks, §7.2 barrier smoothing |
| S3 | Andersen, "Efficient Simulation of the Heston Stochastic Volatility Model" (2008): QE scheme |
| S4 | Hagan, Kumar, Lesniewski, Woodward, "Managing Smile Risk" (2002): SABR implied vol |
| S5 | Albrecher, Mayer, Schoutens, Tistaert, "The Little Heston Trap" (2007): stable characteristic function |
| S6 | Fang and Oosterlee, "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions" (2008): COS method |
| S7 | Carr and Madan, "Option Valuation Using the Fast Fourier Transform" (1999) |
| S8 | Longstaff and Schwartz, "Valuing American Options by Simulation" (2001) |
| S9 | Gatheral and Jacquier, "Arbitrage-Free SVI Volatility Surfaces" (2014): SVI/SSVI and no-arbitrage conditions |
| S10 | Jäckel, "Let's Be Rational" (2015): Black implied volatility |
| S11 | QuantLib (quantlib.org): reference implementation for dates, calendars, closed forms |
| S12 | TF Quant Finance (github.com/google/tf-quant-finance): prior art on TensorFlow |
| S13 | torchquant (jialuechen.github.io/torchquant): prior PyTorch effort |
| S14 | Jorion, *Value at Risk* (3rd ed., 2006): historical, parametric and Monte Carlo VaR (MKR-05 to MKR-07) |
| S15 | Tasche, "Euler Allocation: Theory and Practice" (2007): risk attribution (MKR-09) |
| S16 | Kupiec, "Techniques for Verifying the Accuracy of Risk Measurement Models" (1995): VaR backtesting (MKR-12) |
| S17 | ISDA CDS Standard Model (cdsmodel.com): CDS pricing conventions (CRD-02) |
| S18 | O'Kane, *Modelling Single-name and Multi-name Credit Derivatives* (2008): survival curves and bootstrap (CRD-01 to CRD-03) |
| S19 | Gregory, *The xVA Challenge* (4th ed., 2020): exposure, collateral and CVA (CRD-05 to CRD-07) |
| S20 | Vasicek, "The Distribution of Loan Portfolio Value" (2002): portfolio credit loss (CRD-08) |

## Appendix E Related libraries

Open-source libraries checked on 1 November 2026 (GitHub repository, licence file and last
push). Licences matter for one reason here: SmallQuant may read, test against and learn from
any of them, but copying code in ties SmallQuant to that licence (D01). Copyleft (GPL, AGPL)
and non-commercial terms are fine for test-only references and incompatible with a permissive
SmallQuant licence if code is copied.

Most production pricing code is proprietary: bank in-house libraries and vendor products such
as Numerix, FINCAD, Murex, Bloomberg DLIB, MATLAB Financial Toolbox and MatLogica AADC. They
are not available as references, but they set the bar users will compare SmallQuant against.

**Commercial products (benchmarks of expectation, not references)**

| Product | Website | Licence | Relevance |
| --- | --- | --- | --- |
| Numerix | numerix.com | Commercial enterprise licence | Broad cross-asset pricing, XVA and risk for banks, insurers and asset managers. Defines the breadth SmallQuant deliberately does not attempt (PRD-05). |
| Vola Dynamics | voladynamics.com | Commercial; pricing on request | Arbitrage-free volatility surface fitting and options pricing for equity and FICC options (stocks, ETFs, indices, futures), plus VIX, variance and volatility swaps. Ships SVI, SSVI and SABR alongside its own parametric curves, with C++, Python, Java and C# libraries. Used by institutional options desks. The bar for VOL-05 and VOL-06: surfaces must stay arbitrage-free and stable through stressed markets without manual tuning. |

**Differentiable or GPU pricing (closest to SmallQuant)**

| Library | Repository | Licence | Relevance |
| --- | --- | --- | --- |
| TF Quant Finance | `google/tf-quant-finance` | Apache-2.0 | The design SmallQuant follows: rates, vol models, PDE and Monte Carlo on TensorFlow. README states it is archived and unmaintained; occasional build fixes only. Reference for API and algorithms. |
| torchquant | `jialuechen/torchquant` | Apache-2.0 | Small PyTorch pricing effort. |
| pfhedge | `pfnet-research/pfhedge` | MIT | PyTorch deep hedging: instruments, Black-Scholes and Heston paths, hedging losses. Last push August 2024. Model for ML built on top of SmallQuant (PRD-05). |
| torchsde | `google-research/torchsde` | Apache-2.0 | Differentiable SDE solvers for neural SDEs. Archived. |
| finmath-lib | `finmath/finmath-lib` | Apache-2.0 | Java, with built-in AAD through Monte Carlo, rates and LIBOR market models. Closest prior art for differentiable Monte Carlo and early exercise (sections 13–14). |
| XAD | `auto-differentiation/xad` | AGPL-3.0 | C++ and Python AAD engine, can make QuantLib differentiable. Benchmark comparison for autograd Greeks (PRF-04). Do not copy code. |

**Python pricing (independent test references, QA-01 kind 2)**

| Library | Repository | Licence | Relevance |
| --- | --- | --- | --- |
| FinancePy | `domokane/FinancePy` | GPL-3.0 | Bonds, swaps, swaptions, credit, FX and equity options. Second reference beside QuantLib. Test-only. |
| rateslib | `attack68/rateslib` | Source-available non-commercial, or commercial subscription | Curve building and fixed income with automatic differentiation. Reference for RTS-02 bucketed deltas. Test-only. |
| PyFENG | `quants-net/PyFENG` | GPL-2.0 | Heston, SABR variants and Bachelier from the research literature. Reference for MOD-05 and SABR edge cases. Test-only. |
| py_vollib | `vollib/py_vollib` | MIT | Black, Black-Scholes and Black76 prices and implied vol via Let's Be Rational. Direct reference for VOL-01 and D05. |

**Other languages**

| Library | Repository | Licence | Relevance |
| --- | --- | --- | --- |
| QuantLib | `lballabio/QuantLib` | BSD-style | Primary test reference (S11). Has reference engines for every instrument in section 11: barrier, double and partial barrier, digital and one-touch, Asian (analytic geometric; approximate or Monte Carlo arithmetic), lookback, American, plus cliquet, chooser, compound, basket, Margrabe and variance swaps. |
| OpenGamma Strata | `OpenGamma/Strata` | Apache-2.0 | Java rates and credit with market conventions. Reference for RTS-04 schedules. |
| Open Source Risk Engine | `OpenSourceRisk/Engine` | BSD-style | XVA and portfolio risk built on QuantLib. Reference for exposure and CVA (CRD-05 to CRD-07). |
| RustQuant | `avhz/RustQuant` | Apache-2.0 | Options, rates, stochastic processes and its own autodiff. |

**Trading and portfolio side (outside scope, PRD-05)**

| Library | Repository | Licence | Area |
| --- | --- | --- | --- |
| vectorbt | `polakowo/vectorbt` | Apache-2.0 with Commons Clause | Vectorised backtesting |
| backtrader | `mementum/backtrader` | GPL-3.0 | Backtesting; last push August 2024 |
| zipline-reloaded | `stefan-jansen/zipline-reloaded` | Apache-2.0 | Backtesting |
| PyPortfolioOpt | `PyPortfolio/PyPortfolioOpt` | MIT | Portfolio optimisation |
| Riskfolio-Lib | `dcajasn/Riskfolio-Lib` | BSD-3-Clause | Portfolio optimisation |
| cvxportfolio | `cvxgrp/cvxportfolio` | GPL-3.0 | Portfolio optimisation |
| OpenBB | OpenBB | Apache-2.0 | Market data and research platform |
