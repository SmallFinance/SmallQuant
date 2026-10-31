# SmallQuant Production Build Scope

**Implementation specification**

Prepared for Ted Wong and SmallQuant contributors | 31 October 2026 | Version 1.0

**Purpose:** turn the SmallQuant design documents and the code that exists today into a
complete, tested, PyTorch-native derivatives pricing library. This document records the
final product decisions, current behaviour, production gaps, interface contracts, numerical
requirements and acceptance criteria. It is a build baseline for estimation and
implementation; the decision register identifies the matters that still need an owner's
answer.

**Code baseline:** `main` at commit `35e7e4b` (27 October 2026), inspected on 31 October 2026.
Source: `smallquant/` and `tests/` in the repository. Test run: 197 passed in 58 s on CPU
(Python 3.13.15, PyTorch 2.14.1, QuantLib 1.44, NumPy 2.5.4, macOS arm64).

**Status:** design stage. `smallquant.time` and `smallquant.models.black_scholes` exist; every
other module in this scope is planned. Nothing here is a claim that a planned feature works.

**Resolved since baseline:** the baseline observations below are kept as recorded; these have
since been fixed.

- MOD-02 / MOD-03: Black-Scholes returns the intrinsic limit at `expiry <= 0` and `vol <= 0`
  with finite gradients (AT-08 tests added).
- API-02: `import smallquant as sq` exposes `sq.models.black_scholes`; the core still does not
  import `smallquant.time` (tested).
- API-04: `pyproject.toml` has a hatchling build backend and the wheel installs with PyTorch
  only. Licence, PyPI name and versioning (API-05, API-06) remain open.
- QA-03 / ENV-04: GitHub Actions CI runs `pytest` and `uv build` on macOS CPU for Python 3.10
  and 3.13. `ruff` is not yet enforced: the default rules report 60 findings and 14 files
  need reformatting, which is a separate clean-up.
- DOC-03 / TIM-05: the root README, the `docs/README.md` quick look, the roadmap's stale
  `smallquant.dates` item and the QuantLib reference version (now 1.44) are updated.

## Contents

1. [How to use this specification](#1-how-to-use-this-specification)
2. [Product outcome and boundaries](#2-product-outcome-and-boundaries)
3. [Final decisions that must survive the build](#3-final-decisions-that-must-survive-the-build)
4. [Current library and production gap](#4-current-library-and-production-gap)
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
tensor contract every module obeys. Sections 7–17 describe each functional area. Sections
18–23 cover packaging, validation, performance, documentation, testing and delivery. Each
numbered requirement is suitable for conversion into an issue; keep its identifier in the
issue title, the test name or docstring, and the pull request description.

Four labels distinguish the authority of a statement:

- **Confirmed requirement** means it is stated in the agreed design documents
  (`docs/overview.md`, `architecture.md`, `conventions.md`, `api.md`, `greeks.md`,
  `monte-carlo.md`, `testing.md`, `time.md`, `roadmap.md`) or is behaviour already shipped
  and tested.
- **Current implementation** describes observed code behaviour at the baseline commit,
  including limitations. It is not automatically a permanent policy.
- **Build requirement** is an engineering requirement proposed here to make the design
  reliable.
- **Decision required** means an owner must resolve it before the affected behaviour is
  released.

Where this specification and an older design document disagree, this specification wins and
the design document is updated in the same pull request. Signatures in `docs/api.md` are the
target contract; argument lists may grow, but names and semantics may not change without a
recorded decision.

This document was written from a reading of every design document, the full source tree and
a fresh test run. It is a specification review, not a claim that every planned numerical
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

**PRD-04 Confirmed requirement.** The runtime dependency is PyTorch only. NumPy, QuantLib,
pytest and SciPy are test-only.

**PRD-05 Scope boundary.** Not in scope: trade booking, trade stores, market-data
connectors, a full QuantLib replacement, and machine-learning price models. Deep hedging and
neural pricers may be built on top of SmallQuant but are not part of it.

**PRD-06 Confirmed requirement.** The pricing core works in year fractions and never
imports `smallquant.time`. The time module is an optional helper that turns dates into year
fractions.

**PRD-07 Confirmed requirement.** Target users are quants wanting exact Greeks and fast
calibration, risk teams running large GPU Monte Carlo jobs (XVA, VaR, scenario sweeps), and
researchers combining classical pricing with ML in one PyTorch graph.

## 3 Final decisions that must survive the build

| Area | Final instruction |
| --- | --- |
| Framework | PyTorch only at runtime. No TensorFlow, JAX or NumPy code paths in `smallquant/`. |
| Default dtype | `float64`. Python numbers never force `float32`. MPS falls back to `float32` with a one-time warning. |
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
| Layering | Lower layers never import upper ones. `math/` knows no finance; `models/` knows no engines. |
| Validation | Value checks (`vol > 0`) are off by default because they sync the GPU; shape/dtype/device checks are always on. |
| Monte Carlo results | Always reported with a standard error in tests, examples and docs. |
| Benchmarks | No performance number is quoted before it is measured and labelled with hardware and versions. |
| Time module | Pure Python port of QuantLib 1.4x date machinery, snake_case names, per-instance calendar holiday edits. |

## 4 Current library and production gap

The repository contains about 2,760 lines of Python. The shipped code is the
Black-Scholes-Merton closed form (59 lines) and the `smallquant.time` package (about 1,880
lines). There are 197 tests, all comparing against QuantLib, all passing on CPU.

| Capability | What exists now | What must be delivered |
| --- | --- | --- |
| Package entry point | `smallquant/__init__.py` is a docstring only; `sq.models` raises `AttributeError` | Re-exports per section 18 so `import smallquant as sq` exposes every public module |
| Tensor coercion | Private `_as_tensors` inside `black_scholes.py` | Shared `_tensor.py` implementing section 6, used by every module |
| Closed forms | Black-Scholes-Merton call/put with dividend yield | Black76, Bachelier, Merton jump, SABR implied vol, Heston characteristic function |
| Greeks | Users call `torch.autograd` themselves | `sq.greeks` helpers for first order, second order, Jacobian and bump checks |
| Implied vol | None | Batched Black and Bachelier implied vol with implicit gradients |
| Math | `torch.erf` inline | Distributions, batched roots, interpolation, quadrature, tridiagonal solver, random numbers |
| Instruments and engines | None | Section 11–14 instruments and Analytic, Monte Carlo, PDE, Fourier, Lattice engines |
| Calibration | None | `calibrate` with reparameterised bounds and L-BFGS |
| Rates | None | Discount curves and differentiable bootstrapping; Hull-White |
| Dates | `smallquant.time` complete for dates, periods, 10 day-counter families and 8 calendar families | `Schedule`, `Business252`, more country calendars (section 17) |
| Tests | QuantLib comparisons on CPU, `float64` | Device and dtype parametrisation, `gradcheck`, no-arbitrage, cross-engine, compile tests |
| Tooling | `uv` project with `package = false`; no CI, no linter config | Installable package, CI, `ruff`, release process (section 18) |
| Benchmarks | None | `benchmarks/` scripts run on demand |

Stale material to clean up, not to preserve:

- `README.md` (repository root) is a two-line summary with external links only.
- `docs/README.md` quick look calls `sq.models.black_scholes.price` and
  `sq.greeks.first_order`; neither resolves from `import smallquant as sq` today.
- `docs/roadmap.md` "Later" lists a `smallquant.dates` module; that work shipped as
  `smallquant.time`.
- `docs/time.md` names QuantLib 1.43 as the reference; the locked environment resolves
  QuantLib 1.44 and the suite still passes.

## 5 Users and supported environments

**ENV-01 Confirmed requirement.** Python `>=3.10,<3.15`. Development uses Python 3.13
(`.python-version`) and `uv` with the committed `uv.lock`.

**ENV-02 Confirmed requirement.** Devices: CPU, CUDA and MPS. Tests are parametrised over
available devices. MPS has no `float64`; see TEN-04.

**ENV-03 Build requirement.** State a minimum supported PyTorch version in `pyproject.toml`
and test it in CI alongside the latest release. `torch.func`, `torch.compile` and
`torch.quasirandom.SobolEngine` behaviour must be available at that minimum.

**ENV-04 Decision required.** Which CI hardware is available. CPU on Linux and macOS is the
proposed default; CUDA and MPS runs need either self-hosted runners or a documented manual
release check.

## 6 Tensor, dtype and device contract

**TEN-01 Confirmed requirement.** Public functions accept `torch.Tensor`, Python numbers or
anything `torch.as_tensor` accepts. All tensor arguments broadcast under PyTorch rules and
the output shape is the broadcast shape.

**TEN-02 Confirmed requirement.** dtype resolution order: the `dtype=` argument if given,
otherwise the highest floating dtype among tensor inputs, otherwise `float64`. Python
numbers never force `float32`.

**TEN-03 Current implementation.** `black_scholes._as_tensors` takes the dtype and device of
the first tensor input and raises `ValueError` if any other tensor differs in dtype or
device. It has no `dtype=` argument. A `float32` tensor input produces a `float32` result,
which is consistent with TEN-02, but a `float32` and a `float64` tensor together raise
rather than promoting. Replace it with the shared `_tensor.py` implementation.

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

**TEN-07 Build requirement.** `_tensor.py` provides: `coerce(*values, dtype=None)` returning
tensors on a common dtype/device, `common_device`, `broadcast_shapes`, `zeros_like_inputs`
style constant helpers, and the MPS fallback. It is private and may change, but every
public function uses it.

**TEN-08 Confirmed requirement.** Extra shape-defining dimensions are documented per
function in a fixed position: paths, then time, then state dimension.

## 7 Mathematical primitives

**MTH-01 Confirmed requirement.** `sq.math.normal_cdf`, `normal_pdf`, `normal_icdf` and
`bivariate_normal_cdf(x, y, rho)`, differentiable and accurate in the tails. `normal_cdf`
uses `erfc` for negative arguments so deep out-of-the-money prices keep relative accuracy.

**MTH-02 Confirmed requirement.** `roots.newton(f, x0, *, fprime=None, tol=1e-12,
max_iter=50)` and `roots.brent(f, lo, hi, *, tol=1e-12, max_iter=100)` are batched, run a
fixed maximum iteration count with a convergence mask, and never sync to the host inside the
loop.

**MTH-03 Confirmed requirement.** Root results are differentiable through the implicit
function theorem: one backward step at the solution, `dx/dθ = -(∂g/∂x)⁻¹ · ∂g/∂θ`.

**MTH-04 Build requirement.** Roots return a convergence mask (or expose one) so callers can
detect non-converged elements without a host sync. Non-converged elements return NaN only
when validation is on; otherwise the last iterate, documented as such.

**MTH-05 Confirmed requirement.** Interpolation: `linear`, `log_linear`, `monotone_cubic`
and natural cubic spline; differentiable with respect to both `x` and the node values `fp`.

**MTH-06 Confirmed requirement.** Quadrature: Gauss-Legendre, Gauss-Laguerre and Simpson.
Batched tridiagonal (Thomas) solver and Cholesky helpers in `linalg.py`.

**MTH-07 Confirmed requirement.** `random.normals(shape, *, method="pseudo" | "sobol",
generator=None)`; see section 13 for Sobol scrambling and Brownian bridge.

## 8 Closed-form models

**MOD-01 Confirmed requirement and current implementation.**
`sq.models.black_scholes.price(*, spot, strike, expiry, rate, vol, div=0.0, is_call=True)`
exists. It prices calls and puts with continuous dividend yield, accepts a bool or bool
tensor `is_call`, defaults to `float64` and matches QuantLib's `AnalyticEuropeanEngine` to
relative 1e-9 for price, delta, gamma, vega, rho and the expiry derivative over eleven cases
including negative rates and five-year expiries.

**MOD-02 Current implementation.** At `expiry=0` the function returns NaN because it
divides by `vol * sqrt(expiry)`. At `vol=0` the price is the correct discounted intrinsic
value but gradients are not defined. Both limits are listed in `testing.md` as required
identities.

**MOD-03 Build requirement.** Define zero-expiry and zero-vol limits: price equals intrinsic
(undiscounted at `expiry=0`, discounted forward intrinsic at `vol=0`) with finite gradients.
Implement by masking `vol * sqrt(expiry)` to a safe value before the division and selecting
the limit with `torch.where`, per TEN-06.

**MOD-04 Confirmed requirement.** Add `black76.price(*, forward, strike, expiry, vol,
discount=1.0, is_call=True)` and `bachelier.price(*, forward, strike, expiry, normal_vol,
discount=1.0, is_call=True)`.

**MOD-05 Confirmed requirement.** Add `sabr.implied_vol(*, forward, strike, expiry, alpha,
beta, rho, nu)` (Hagan 2002), `heston.characteristic_function(u, *, params, expiry, ...)`
and `merton_jump.price(..., jump_intensity, jump_mean, jump_vol, num_terms=64)`.

**MOD-06 Confirmed requirement.** Parameter dataclasses `BlackScholesParams`,
`HestonParams`, `SABRParams`, `LocalVolParams`, `HullWhiteParams` hold tensors and are
registered as pytrees so `torch.func` and `torch.compile` see through them.

**MOD-07 Confirmed requirement.** Each model module also exposes simulation dynamics
(`drift_diffusion(...)` or an exact `step(...)`) consumed by the Monte Carlo engine.

**MOD-08 Build requirement.** SABR at-the-money uses the series limit of `z / x(z)` so the
formula and its gradient are finite at `strike == forward`. Heston's characteristic
function uses the formulation that avoids the branch-cut discontinuity (Albrecher et al.
"little Heston trap").

**MOD-09 Confirmed requirement.** Every formula docstring cites its source (book, paper,
equation number).

## 9 Greeks and differentiation

**GRK-01 Confirmed requirement.** `sq.greeks.first_order(price, inputs, *, sum_batch=False)`
returns `dict[str, Tensor]`. For a batched price each gradient is of `price.sum()`: per-option
for inputs batched one-to-one with prices, portfolio-aggregate for shared inputs.

**GRK-02 Decision required.** `sum_batch=False` is in the signature but its meaning is not
specified; `first_order` already sums by construction. Either define it (for example, reduce
shared-input gradients only when `True`, otherwise raise for shared inputs and point to
`jacobian`) or remove it before the first release.

**GRK-03 Confirmed requirement.** `second_order(price_fn, inputs, pairs=None)` returns
`dict[tuple[str, str], Tensor]` using forward-over-reverse
(`torch.func.jacfwd(torch.func.grad(f))`). `pairs=None` returns all unique pairs.

**GRK-04 Confirmed requirement.** `jacobian(price_fn, inputs)` returns per-option
sensitivities for shared inputs. `bump(price_fn, inputs, *, rel=1e-4, scheme="central")`
exists for testing and user verification only.

**GRK-05 Confirmed requirement.** Named Greeks: delta = dV/dspot, gamma = d²V/dspot²,
vega = dV/dvol, theta = −dV/dexpiry, rho = dV/drate, vanna = d²V/dspot dvol,
volga = d²V/dvol². Units are per unit of the input (vega per 1.00 of vol, not per 1%).

**GRK-06 Confirmed requirement.** Monte Carlo pathwise Greeks come from autograd through
the estimator. Discontinuous payoffs support payoff smoothing (`smoothing="sigmoid"`,
`eps`), the likelihood ratio method (`greeks_method="likelihood_ratio"`) for models with
known densities, and one-step survival for discrete barriers (Glasserman 2004, §7.2).

**GRK-07 Build requirement.** Each discontinuous instrument states its default treatment and
the supported alternatives in its docstring. A pathwise delta of exactly zero on a digital or
barrier, with no treatment selected, emits a warning when validation is on.

**GRK-08 Confirmed requirement.** Solvers (implied vol, roots, bootstrap, calibration)
attach implicit-function-theorem backward passes. `d implied_vol / d price = 1 / vega`
exactly, with no unrolled graph.

## 10 Implied volatility and surfaces

**VOL-01 Confirmed requirement.** `implied_vol(price, *, spot, strike, expiry, rate,
div=0.0, is_call=True)` and `implied_normal_vol(price, *, forward, strike, expiry,
discount=1.0, is_call=True)` solve a whole chain in one call.

**VOL-02 Confirmed requirement.** Method: rational initial guess, then Newton safeguarded by
bisection, under the fixed-iteration masked loop of MTH-02.

**VOL-03 Build requirement.** Prices outside no-arbitrage bounds (below intrinsic, above the
discounted underlying) return NaN for those elements. Prices at intrinsic return zero vol.
The behaviour is documented and tested, not left to the solver.

**VOL-04 Decision required.** Initial guess for Black implied vol: Jäckel's "Let's Be
Rational" gives machine precision in two iterations but is substantially more code; a
Brenner-Subrahmanyam or Corrado-Miller guess is simpler and needs more Newton steps.

**VOL-05 Confirmed requirement.** `SVI(a, b, rho, m, sigma)` slices, `SSVI(theta_curve, rho,
phi)` surfaces and `dupire_local_vol(surface, spot, times, strikes)`, all differentiable in
their parameters.

**VOL-06 Build requirement.** Provide no-arbitrage checks for SVI/SSVI (butterfly and
calendar conditions) as functions returning masks, used by calibration penalties and tests.

## 11 Instruments

**INS-01 Confirmed requirement.** Instruments are frozen dataclasses holding contract terms
with a `payoff(paths, times) -> Tensor` method returning undiscounted cash flows per path.
They contain no pricing logic.

**INS-02 Confirmed requirement.** Instruments: `EuropeanOption`, `DigitalOption` (cash),
`BarrierOption` (kind, rebate, discrete or continuous monitoring, observation times),
`AsianOption` (arithmetic or geometric average), `LookbackOption` (floating or fixed strike)
and `AmericanOption` (exercise times).

**INS-03 Build requirement.** Instruments declare their observation and exercise times so
the Monte Carlo engine can merge them into its grid (MC-02), and declare the running
statistics they need (running max, running sum, hit flag) so full paths need not be stored
(MC-08).

**INS-04 Decision required.** Barrier `kind` values. The API shows `"down_and_out"`; the full
set (up/down × in/out) and whether continuous monitoring uses the Broadie-Glasserman-Kou
shift in Monte Carlo need to be fixed before the barrier engine work.

**INS-05 Build requirement.** Contract terms may be tensors so a batch of strikes or
barriers prices in one call, following TEN-01.

## 12 Engines and the price entry point

**ENG-01 Confirmed requirement.** Every engine implements `engine.price(instrument,
model_params, market) -> Tensor`. `market` is `Market(spot, rate | curve, div=0.0)`.

**ENG-02 Confirmed requirement.** Engines declare the (model, instrument) pairs they support
and raise `NotImplementedError` naming the pair otherwise.

**ENG-03 Confirmed requirement.** `sq.price(instrument, model_params, market, engine=None)`
uses `Analytic` when it supports the pair and otherwise raises, asking for an explicit
engine. It never silently falls back to Monte Carlo.

**ENG-04 Confirmed requirement.** Engines: `Analytic()`, `MonteCarlo(...)` (section 13),
`PDE(num_space=400, num_time=400, *, theta=0.5, rannacher_steps=2)`,
`Fourier(method="cos", num_terms=256)`, `Lattice(num_steps=500, kind="crr" | "trinomial")`.

**ENG-05 Build requirement.** Support registration through a table keyed by (model params
type, instrument type), so adding a pair does not require editing the engine's dispatch
code and `ENG-02` errors list what is supported.

## 13 Monte Carlo

**MC-01 Confirmed requirement.** Paths have shape `[num_paths, num_steps + 1, dim]`, are
simulated in log space where the model allows, and keep a trailing `dim` of 1 for one-factor
models.

**MC-02 Confirmed requirement.** The time grid is the sorted merge of instrument
observation/exercise times with an optional maximum step size.

**MC-03 Confirmed requirement.** Default schemes: exact lognormal (Black-Scholes), exact
normal (Bachelier), exact step with Poisson jump counts (Merton), Andersen QE with
martingale correction (Heston), log-Euler with surface interpolation (local vol), exact
Gaussian step of rate and integral (Hull-White). Euler is available for every model as a
reference.

**MC-04 Confirmed requirement.** Pseudo-random numbers use a `torch.Generator` on the target
device seeded by `seed=`. Same seed, device and dtype give the same paths.

**MC-05 Confirmed requirement.** Quasi-random numbers use scrambled
`torch.quasirandom.SobolEngine` mapped through the inverse normal CDF with a Brownian bridge.
Error estimates use several independent scramblings (randomised QMC).

**MC-06 Confirmed requirement.** Variance reduction: antithetic variates (default on),
control variates with the coefficient estimated from the same paths, optional moment
matching.

**MC-07 Confirmed requirement.** `chunk_size=` simulates and prices in chunks, accumulating
payoff sums and gradients. Results are identical to the unchunked run with the same seed.

**MC-08 Confirmed requirement.** Payoffs needing only running statistics are computed on
the fly. `torch.utils.checkpoint` over time blocks is available for Greeks on very long
paths.

**MC-09 Confirmed requirement.** `return_stderr=True` returns `(price, stderr)`.

**MC-10 Build requirement.** Chunked runs must draw the same random numbers as unchunked
runs. Generate per-chunk streams from a counter-based offset (or Sobol `fast_forward`)
rather than from a shared generator consumed in a different order.

**MC-11 Decision required.** CPU and GPU random streams differ in PyTorch. Decide whether
cross-device reproducibility is a goal (requires generating on CPU and moving) or only
same-device reproducibility (the current wording of MC-04).

## 14 PDE, Fourier and lattice engines

**PDE-01 Confirmed requirement.** One-dimensional Crank-Nicolson finite differences with
Rannacher start-up steps, using the batched tridiagonal solver.

**PDE-02 Confirmed requirement.** American options use a projected Crank-Nicolson step.

**PDE-03 Build requirement.** Grids are non-uniform and concentrated around the strike and
barriers; Greeks come from autograd through the solver and must be smooth in spot (no grid
staircase). Test against closed forms per AT-12.

**FOU-01 Confirmed requirement.** Characteristic-function pricing with COS (default) and
Carr-Madan for Heston and other affine models.

**LAT-01 Confirmed requirement.** CRR binomial and trinomial trees, European and American.

**LSM-01 Confirmed requirement.** Longstaff-Schwartz in the Monte Carlo engine. The
regression boundary is estimated, then held fixed under `torch.no_grad` when computing
Greeks (envelope theorem).

**LSM-02 Decision required.** Regression basis (polynomial degree, Laguerre versus monomial)
and whether the boundary is fitted on an independent path set to remove upward bias.

## 15 Calibration

**CAL-01 Confirmed requirement.** `calibrate(model_params, market_quotes, pricer, *,
loss="vol_rmse", optimizer="lbfgs", bounds=..., max_iter=200)` returns an object with
`params`, `loss` and `converged`.

**CAL-02 Confirmed requirement.** Bounds are handled by reparameterisation: softplus for
positive parameters, `tanh` for correlations. Standard unconstrained `torch.optim`
optimisers apply.

**CAL-03 Build requirement.** Returned parameters carry a gradient path to the market quotes
through the implicit function theorem at the optimum, so parameter sensitivities to each
quote are available (the roadmap Phase 2 outcome).

**CAL-04 Build requirement.** Record iteration count, final gradient norm and the reason for
stopping. `converged=False` is a normal return, not an exception.

## 16 Rates

**RTS-01 Confirmed requirement.** `DiscountCurve(times, discount_factors,
interpolation="log_linear")` with `discount(t)`, `zero_rate(t)` and `forward_rate(t1, t2)`;
`ZeroCurve` as an alternative constructor.

**RTS-02 Confirmed requirement.** `bootstrap(deposits=..., swaps=...)` is differentiable
with respect to quotes, so bucketed rate deltas come straight from autograd.

**RTS-03 Confirmed requirement.** Hull-White one-factor: bond prices, caps/floors,
swaptions by Jamshidian decomposition, and exact simulation.

**RTS-04 Decision required.** How swap and deposit quotes specify their schedules. The core
uses year fractions only (PRD-06), so either callers pass accrual year fractions and payment
times, or an optional adapter in `smallquant.time` builds them once `Schedule` exists.

## 17 Dates, day counts and calendars

**TIM-01 Confirmed requirement and current implementation.** `smallquant.time` is a pure
Python port of QuantLib's date machinery with no torch dependency. It provides `Date`,
`Month`, `Weekday`, `Period`, `TimeUnit`, `Frequency`, the day counters `Actual360`,
`Actual366`, `Actual36525`, `Actual365Fixed`, `Thirty360`, `ActualActual`,
`SimpleDayCounter`, `OneDayCounter`, `BusinessDayConvention`, `Calendar`, and the calendars
`NullCalendar`, `WeekendsOnly`, `TARGET`, `UnitedStates`, `UnitedKingdom`, `Japan`,
`JointCalendar`, `BespokeCalendar`.

**TIM-02 Confirmed requirement.** The recorded differences from QuantLib in `docs/time.md`
are intentional and must not be "fixed": snake_case names, `None` for a missing reference
period, `Period.__eq__` returning `False` for undecidable comparisons, `ValueError` for
errors, per-instance `add_holiday` / `remove_holiday`, keyword-only `convention` and
`end_of_month` on `advance`.

**TIM-03 Current implementation.** Tests compare against QuantLib over every serial number
in range, two years of daily `Date ± Period`, every day counter on edge grids and random
pairs (relative 1e-14), and calendar holiday grids from 1950 to 2035.

**TIM-04 Build requirement.** Port `Schedule` (with date-generation rules), then
`ActualActual` ISMA driven by a schedule, then `Business252` with its calendar. Additional
country calendars are added on request, each with a QuantLib holiday-grid test.

**TIM-05 Build requirement.** Pin or record the QuantLib reference version. The docs say
1.43; the lock resolves 1.44. Update `docs/time.md` to the version CI uses, and treat a
failure after a QuantLib upgrade as a port to update, as the doc already says.

**TIM-06 Decision required.** Whether `Australia` (ASX, Settlement) is the next calendar,
given the maintainers' location.

## 18 Public API surface and packaging

**API-01 Confirmed requirement.** `import smallquant as sq` exposes `sq.math`, `sq.rates`,
`sq.vol`, `sq.models`, `sq.instruments`, `sq.engines`, `sq.greeks`, `sq.calibration`,
`sq.price`, `sq.Market`, `sq.set_default_dtype` and `sq.set_validation`.

**API-02 Current implementation.** `smallquant/__init__.py` and
`smallquant/models/__init__.py` contain only docstrings, so `sq.models` raises
`AttributeError` and the documentation quick look fails. Users must import
`smallquant.models.black_scholes` directly.

**API-03 Build requirement.** `smallquant.time` is not imported by `smallquant/__init__.py`
(PRD-06); users import it explicitly. Everything not listed in a module's `__all__` is
private.

**API-04 Current implementation.** `pyproject.toml` sets `[tool.uv] package = false` and
tests rely on `pythonpath = ["."]`. The project cannot be built or installed as a wheel.

**API-05 Build requirement.** Add a build backend, remove `package = false`, add a
`py.typed` marker and version the package with semantic versioning starting at `0.1.0` for
the Phase 0 release. Breaking changes before `1.0` are allowed but recorded in a changelog.

**API-06 Decision required.** Licence (the repository has none), PyPI distribution name
(`smallquant` availability) and who holds publishing rights for the SmallFinance
organisation.

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
pricing, implied vol over a surface, Heston Monte Carlo with Greeks. They do not run in CI.

**PRF-03 Confirmed requirement.** Documentation quotes only measured numbers, labelled with
hardware and library versions.

**PRF-04 Build requirement.** Each benchmark reports a bump-and-reprice baseline next to the
autograd result, since that comparison is the library's main claim.

## 21 Documentation

**DOC-01 Confirmed requirement.** Public functions have type hints and Google-style
docstrings with `Args`, `Returns` and a short `Example`. Formulas cite sources.

**DOC-02 Build requirement.** Every code example in `docs/` and `README.md` runs as a test
(doctest or an example-runner test). That would have caught API-02.

**DOC-03 Build requirement.** Replace the root `README.md` with installation, a working quick
look and a link to `docs/README.md`. Remove the stale `smallquant.dates` item from
`docs/roadmap.md`. Link this specification from `docs/README.md`.

**DOC-04 Confirmed requirement.** A documentation site built from the Markdown files is a
later, unscheduled item.

## 22 Verification and acceptance

**QA-01 Build requirement.** Convert each requirement ID into a traceable test or review
check. The seven test kinds in `docs/testing.md` apply to every feature: published
references, independent implementation (QuantLib, SciPy), no-arbitrage identities,
cross-engine agreement, Greeks, batching and devices, compilation.

**QA-02 Confirmed requirement.** Tolerances: closed forms in `float64` relative 1e-10 against
full-precision references and absolute 1e-4 against four-decimal published values; `float32`
relative 1e-5; Monte Carlo within 3 standard errors with a fixed seed.

**QA-03 Current evidence.** On 31 October 2026 the full suite ran on CPU: 197 passed in
58 s. Black-Scholes tests use relative 1e-9 / absolute 1e-10 against QuantLib (looser than
QA-02's 1e-10; record whether that is deliberate). There is no `conftest.py`, no device or
dtype parametrisation, no `gradcheck`, no `torch.compile` test and no CI. CUDA and MPS have
not been exercised.

| Test ID | Minimum acceptance scenario |
| --- | --- |
| AT-01 | `import smallquant as sq` exposes every API-01 name; every docs and README example runs. |
| AT-02 | Mixed Python numbers and `float32` tensors give `float32`; Python numbers alone give `float64`; `float32` + `float64` tensors give `float64`; `dtype=` overrides. |
| AT-03 | Mixed-device inputs raise; outputs land on the input device; MPS without `float64` warns once and returns `float32`. |
| AT-04 | A batched call equals the stack of individual calls for every public pricing function. |
| AT-05 | `gradcheck` and `gradgradcheck` pass in `float64` for every differentiable public function. |
| AT-06 | Black-Scholes, Black76 and Bachelier match QuantLib and published values for price and first- and second-order Greeks, calls and puts. |
| AT-07 | Put-call parity, monotonicity and convexity in strike, and price bounds hold over random inputs for every closed form. |
| AT-08 | Zero expiry and zero vol return intrinsic values with finite gradients (MOD-03). |
| AT-09 | `implied_vol` round-trips prices across a strike-expiry grid to 1e-10 relative; its gradient equals `1 / vega`; out-of-bounds prices return NaN. |
| AT-10 | `greeks.first_order`, `second_order` and `jacobian` agree with closed-form Greeks and with `bump` central differences. |
| AT-11 | SABR at `strike == forward` has finite price and gradient; Heston characteristic function matches a reference across long expiries without branch jumps. |
| AT-12 | European options priced by Analytic, PDE, Fourier, Lattice and Monte Carlo agree: deterministic engines to a stated tolerance, Monte Carlo within 3 standard errors. |
| AT-13 | Monte Carlo with the same seed, device and dtype is bit-identical; chunked and unchunked runs agree exactly. |
| AT-14 | Sobol with Brownian bridge reports a randomised-QMC error estimate, and its error is below pseudo-random error at equal path count on a benchmark Asian. |
| AT-15 | Digital and barrier Monte Carlo deltas with smoothing, likelihood ratio and one-step survival agree with closed forms within tolerance; untreated pathwise delta is flagged. |
| AT-16 | American put by Longstaff-Schwartz, projected Crank-Nicolson and a 2,000-step CRR tree agree to a stated tolerance; Greeks are finite. |
| AT-17 | Heston and SABR calibrate to synthetic surfaces generated from known parameters and recover them; parameter sensitivities to quotes match bumps. |
| AT-18 | Bootstrapped curves reprice their input instruments to 1e-12; bucketed deltas match bumps. |
| AT-19 | Hull-White bond and swaption prices match QuantLib. |
| AT-20 | Core functions compile with `fullgraph=True` and match eager results. |
| AT-21 | Validation on rejects invalid inputs with named errors; validation off never syncs the host (checked with a sync-detecting test hook). |
| AT-22 | `smallquant.time` suite stays green against the CI QuantLib version; `smallquant` core never imports it. |
| AT-23 | The wheel builds, installs into a clean environment with only PyTorch, and imports. |

**QA-04 Release gate.** No open defect in price correctness, Greek correctness, dtype or
device handling, or reproducibility. A passing happy-path example is not enough; each
release attaches the test run and the hardware it ran on.

## 23 Delivery plan and definition of done

The phases follow `docs/roadmap.md`. Estimate by phase and identify numerical risks before
committing a date. Later phases remain in scope even though they are sequenced last.

| Phase | Concrete deliverables and exit evidence |
| --- | --- |
| 0 Foundation | Installable package, CI on CPU, `ruff`, `conftest.py` device/dtype parametrisation; `_tensor.py`; `math/distributions` and `math/roots`; Black-Scholes limits fixed and moved onto `_tensor.py`; Black76 and Bachelier; `sq.greeks`; `implied_vol` and `implied_normal_vol`; package re-exports. Exit: AT-01–AT-10, AT-20–AT-23. Tag `0.1.0`. |
| 1 Monte Carlo | `math/random` with Sobol and Brownian bridge; instruments; `engines/monte_carlo` with antithetics, control variates, chunking and standard errors; exact Black-Scholes and Bachelier simulation; Merton jump; discontinuous-payoff Greeks. Exit: AT-12 (MC), AT-13–AT-15. |
| 2 Stochastic vol and calibration | Heston characteristic function, Fourier engine and QE simulation; SABR; SVI/SSVI; `calibrate`. Exit: AT-11, AT-17. |
| 3 PDE, lattices, early exercise | Tridiagonal solver; PDE engine; lattice engine; American options; Dupire local vol in Monte Carlo and PDE. Exit: AT-12 (all engines), AT-16. |
| 4 Rates | Curves, differentiable bootstrap, Hull-White. Exit: AT-18, AT-19. |
| Later | Multi-asset models, XVA building blocks, `Schedule` and further calendars, documentation site. |

**DONE-01 Build requirement.** Each phase delivers version-controlled source, tests, updated
docs, a changelog entry and a tagged release. CI is green on the tag.

**DONE-02 Build requirement.** Benchmarks for the phase's headline outcome are run once and
recorded with hardware and versions (PRF-03, PRF-04).

**DONE-03 Confirmed outcome.** A user can price vanilla and exotic options, calibrate Heston
or SABR, and build rate curves on CPU or GPU, with exact Greeks from autograd, without
writing adjoint code or bump loops.

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

These decisions do not prevent estimating the known work. They must be resolved before the
affected behaviour is released. Until then, the default is to follow the existing design
documents and leave the behaviour unimplemented rather than guess.

| ID and owner | Decision and implementation consequence |
| --- | --- |
| D01 Ted | Licence, PyPI name and publishing rights (API-06) |
| D02 Ted | CI hardware for CUDA and MPS, or a manual release check (ENV-04) |
| D03 Ted | Minimum supported PyTorch version (ENV-03) |
| D04 Ted | Meaning of `sum_batch` in `greeks.first_order`, or its removal (GRK-02) |
| D05 Ted | Black implied-vol initial guess: Let's Be Rational versus simpler guess (VOL-04) |
| D06 Ted | Barrier kinds and continuous-monitoring correction (INS-04) |
| D07 Ted | Cross-device Monte Carlo reproducibility (MC-11) |
| D08 Ted | Longstaff-Schwartz basis and independent-path boundary (LSM-02) |
| D09 Ted | How rate instruments receive schedules without the core importing dates (RTS-04) |
| D10 Ted | Next calendar to port, e.g. Australia (TIM-06) |
| D11 Ted | Whether Black-Scholes test tolerance should tighten from 1e-9 to QA-02's 1e-10 (QA-03) |

## Appendix C Code and evidence map

| Area | Source landmarks to inspect |
| --- | --- |
| Package entry | `smallquant/__init__.py`, `smallquant/models/__init__.py` (docstrings only) |
| Black-Scholes | `smallquant/models/black_scholes.py`: `_as_tensors`, `price` |
| Dates | `smallquant/time/date.py`: `Date`, `Month`, `Weekday`, serial conversion |
| Periods | `smallquant/time/period.py`: `Period`, `TimeUnit`, `Frequency`, `Period.parse` |
| Day counters | `smallquant/time/daycounter.py`: `DayCounter` and subclasses |
| Calendars | `smallquant/time/calendar.py` (`Calendar`, `BusinessDayConvention`, `WesternCalendar`); `calendars.py` (concrete calendars) |
| QuantLib reference | `tests/reference/ql.py`: `european` returns price and Greeks in SmallQuant conventions |
| Model tests | `tests/models/test_black_scholes_vs_quantlib.py` |
| Time tests | `tests/time/test_date_period_vs_quantlib.py`, `test_daycounter_vs_quantlib.py`, `test_calendar_vs_quantlib.py` |
| Example | `examples/black_scholes.py` |
| Design docs | `docs/overview.md`, `architecture.md`, `conventions.md`, `api.md`, `greeks.md`, `monte-carlo.md`, `time.md`, `testing.md`, `roadmap.md` |

Recent history, in order: design docs (`a6b18d0`), QuantLib reference tests and a `float32`
`sqrt(2)` fix (`48fbcab`), `smallquant.time` dates, periods and day counters (`6f14146`),
Black-Scholes moved into `smallquant.models` with puts and dividends (`3f94264`), holiday
calendars and business-day conventions (`6c7e4e5`).

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

## Appendix E Related libraries

Open-source libraries checked on 1 November 2026 (GitHub repository, licence file and last
push). Licences matter for one reason here: SmallQuant may read, test against and learn from
any of them, but copying code in ties SmallQuant to that licence (D01). Copyleft (GPL, AGPL)
and non-commercial terms are fine for test-only references and incompatible with a permissive
SmallQuant licence if code is copied.

Most production pricing code is proprietary: bank in-house libraries and vendor products such
as Numerix, FINCAD, Murex, Bloomberg DLIB, MATLAB Financial Toolbox and MatLogica AADC. They
are not available as references.

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
| Open Source Risk Engine | `OpenSourceRisk/Engine` | BSD-style | XVA and portfolio risk built on QuantLib. Reference for the later XVA item. |
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
