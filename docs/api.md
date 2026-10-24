# API

The planned public API. Signatures are the target; argument lists may grow, but names and
semantics in this document are the contract. See [conventions.md](conventions.md) for shared
rules (broadcasting, dtype, device, argument names).

## Top level

```python
import smallquant as sq

sq.price(instrument, model_params, market, engine=None) -> Tensor
sq.set_default_dtype(torch.float64)
sq.set_validation(False)
```

`sq.price` picks the analytic engine when one exists for the pair, otherwise raises and asks
for an explicit engine.

## `sq.math`

```python
sq.math.normal_cdf(x)
sq.math.normal_pdf(x)
sq.math.normal_icdf(p)
sq.math.bivariate_normal_cdf(x, y, rho)

sq.math.roots.newton(f, x0, *, fprime=None, tol=1e-12, max_iter=50)   # batched
sq.math.roots.brent(f, lo, hi, *, tol=1e-12, max_iter=100)            # batched, bracketing
sq.math.interpolate.linear(x, xp, fp)
sq.math.interpolate.monotone_cubic(x, xp, fp)
sq.math.linalg.tridiagonal_solve(lower, diag, upper, rhs)             # batched Thomas
sq.math.random.normals(shape, *, method="pseudo" | "sobol", generator=None)
```

Root finders run a fixed maximum number of iterations over the whole batch with a
convergence mask, so they work under `torch.compile` and on GPU without host syncs. Their
results are differentiable through the implicit function theorem (one backward step at the
solution) rather than by unrolling iterations.

## `sq.rates`

```python
curve = sq.rates.DiscountCurve(times, discount_factors, interpolation="log_linear")
curve.discount(t)
curve.zero_rate(t)
curve.forward_rate(t1, t2)

curve = sq.rates.bootstrap(deposits=..., swaps=...)   # differentiable w.r.t. quotes
```

Because bootstrapping is differentiable, sensitivities to each market quote (bucketed rate
deltas) come straight from autograd.

## `sq.vol`

```python
sq.vol.implied_vol(price, *, spot, strike, expiry, rate, div=0.0, is_call=True)
sq.vol.implied_normal_vol(price, *, forward, strike, expiry, discount=1.0, is_call=True)

slice_ = sq.vol.SVI(a, b, rho, m, sigma)       # total variance as a function of log-moneyness
surface = sq.vol.SSVI(theta_curve, rho, phi)
sq.vol.dupire_local_vol(surface, spot, times, strikes)
```

`implied_vol` solves a whole chain in one call (rational initial guess + Newton safeguarded
by bisection) and is differentiable with respect to the price.

## `sq.models`

### Closed forms

```python
sq.models.black_scholes.price(*, spot, strike, expiry, rate, vol, div=0.0, is_call=True)
sq.models.black76.price(*, forward, strike, expiry, vol, discount=1.0, is_call=True)
sq.models.bachelier.price(*, forward, strike, expiry, normal_vol, discount=1.0, is_call=True)
sq.models.sabr.implied_vol(*, forward, strike, expiry, alpha, beta, rho, nu)   # Hagan 2002
sq.models.heston.characteristic_function(u, *, params, expiry, ...)
sq.models.merton_jump.price(*, spot, strike, expiry, rate, vol, jump_intensity,
                            jump_mean, jump_vol, div=0.0, is_call=True, num_terms=64)
```

### Parameter objects

```python
sq.models.BlackScholesParams(vol, div=0.0)
sq.models.HestonParams(v0, kappa, theta, xi, rho)
sq.models.SABRParams(alpha, beta, rho, nu)
sq.models.LocalVolParams(surface)
sq.models.HullWhiteParams(mean_reversion, vol, curve)
```

Every parameter is a tensor, so a batch of parameter sets prices in one call, and gradients
flow to each parameter.

## `sq.instruments`

```python
sq.instruments.EuropeanOption(strike, expiry, is_call=True)
sq.instruments.DigitalOption(strike, expiry, is_call=True, cash=1.0)
sq.instruments.BarrierOption(strike, expiry, barrier, kind="down_and_out", rebate=0.0,
                             is_call=True, monitoring="discrete", observation_times=None)
sq.instruments.AsianOption(strike, expiry, observation_times, average="arithmetic",
                           is_call=True)
sq.instruments.LookbackOption(expiry, observation_times, floating_strike=True, is_call=True)
sq.instruments.AmericanOption(strike, expiry, is_call=False, exercise_times=None)
```

Each has `payoff(paths, times) -> Tensor` returning undiscounted cash flows per path.

## `sq.engines`

```python
sq.engines.Analytic()
sq.engines.MonteCarlo(num_paths, num_steps=None, *, method="pseudo" | "sobol",
                      antithetic=True, control_variate=None, seed=None,
                      chunk_size=None, return_stderr=False)
sq.engines.PDE(num_space=400, num_time=400, *, theta=0.5, rannacher_steps=2)
sq.engines.Fourier(method="cos", num_terms=256)
sq.engines.Lattice(num_steps=500, kind="crr" | "trinomial")
```

All share `engine.price(instrument, model_params, market) -> Tensor`. `market` is a small
dataclass: `Market(spot, rate | curve, div=0.0)`.

American options use Longstaff-Schwartz regression in `MonteCarlo` and a projected
Crank-Nicolson step in `PDE`.

## `sq.greeks`

```python
sq.greeks.first_order(price, inputs: dict[str, Tensor], *, sum_batch=False) -> dict
sq.greeks.second_order(price_fn, inputs: dict[str, Tensor], pairs=None) -> dict
sq.greeks.jacobian(price_fn, inputs) -> dict            # per-option sensitivities
sq.greeks.bump(price_fn, inputs, *, rel=1e-4, scheme="central")   # for checking only
```

Full description in [greeks.md](greeks.md).

## `sq.calibration`

```python
result = sq.calibration.calibrate(
    model_params,                  # initial guess; tensors with requires_grad
    market_quotes,                 # prices or implied vols
    pricer,                        # callable(params) -> model quotes
    *, loss="vol_rmse", optimizer="lbfgs", bounds=..., max_iter=200,
)
result.params, result.loss, result.converged
```

Bounds are handled by reparameterising (softplus for positive parameters, `tanh` for
correlations), so standard unconstrained `torch.optim` optimisers apply.

## End-to-end example

```python
import torch
import smallquant as sq

f64 = dict(dtype=torch.float64, device="cuda")
params = sq.models.HestonParams(
    v0=torch.tensor(0.04, **f64, requires_grad=True),
    kappa=torch.tensor(1.5, **f64, requires_grad=True),
    theta=torch.tensor(0.04, **f64, requires_grad=True),
    xi=torch.tensor(0.5, **f64, requires_grad=True),
    rho=torch.tensor(-0.7, **f64, requires_grad=True),
)
market = sq.Market(spot=torch.tensor(100.0, **f64, requires_grad=True), rate=0.03)
option = sq.instruments.BarrierOption(strike=100.0, expiry=1.0, barrier=80.0,
                                      kind="down_and_out", observation_times=torch.linspace(0, 1, 253, **f64)[1:])

engine = sq.engines.MonteCarlo(num_paths=2**20, method="sobol", seed=7)
price = engine.price(option, params, market)
greeks = sq.greeks.first_order(price, {"spot": market.spot, "v0": params.v0, "rho": params.rho})
```
