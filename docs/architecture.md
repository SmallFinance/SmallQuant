# Architecture

## Package layout

```
smallquant/
├── __init__.py          # re-exports the public modules below
├── _tensor.py           # input coercion, dtype/device resolution, broadcasting helpers
├── math/
│   ├── distributions.py # normal pdf/cdf/inverse cdf, bivariate normal
│   ├── roots.py         # batched Newton, Brent and bisection
│   ├── interpolate.py   # linear, log-linear, monotone cubic, natural cubic spline
│   ├── integrate.py     # Gauss-Legendre / Gauss-Laguerre quadrature, Simpson
│   ├── linalg.py        # batched tridiagonal (Thomas) solver, Cholesky helpers
│   └── random.py        # pseudo-random and Sobol normals, Brownian bridge
├── rates/
│   ├── curves.py        # DiscountCurve, ZeroCurve, forward rates
│   └── bootstrap.py     # differentiable curve bootstrapping from deposits/swaps
├── vol/
│   ├── implied.py       # batched implied volatility (Black, Bachelier)
│   └── surfaces.py      # SVI / SSVI slices, interpolated surfaces, Dupire local vol
├── models/              # one module per model: closed forms + simulation dynamics
│   ├── black_scholes.py
│   ├── black76.py
│   ├── bachelier.py
│   ├── heston.py
│   ├── sabr.py
│   ├── merton_jump.py
│   ├── local_vol.py
│   └── hull_white.py
├── instruments/         # payoff definitions, independent of how they are priced
│   ├── vanilla.py       # European call/put, digital
│   ├── barrier.py
│   ├── asian.py
│   ├── lookback.py
│   └── american.py      # early-exercise flag + exercise schedule
├── engines/             # numerical methods that price an instrument under a model
│   ├── analytic.py      # dispatch to closed forms where they exist
│   ├── monte_carlo.py
│   ├── pde.py           # 1-D Crank-Nicolson finite differences
│   ├── fourier.py       # characteristic-function pricing (COS, Carr-Madan)
│   └── lattice.py       # binomial / trinomial trees
├── greeks.py            # autograd helpers for first and second order sensitivities
└── calibration.py       # fit model parameters to market quotes with torch.optim
```

## Layers

```
            calibration.py, greeks.py
                       │
       engines/  ◄─────┼─────►  instruments/
          │            │             │
          └──────► models/ ◄─────────┘
                       │
         rates/, vol/  │
                       ▼
                    math/
                       │
                   _tensor.py
```

Lower layers never import upper ones. `math/` knows nothing about finance; `models/` knows
nothing about engines.

## The three core abstractions

### Model

A model describes the dynamics of the underlying. Each model module exposes:

- **Closed-form functions** where they exist, as plain functions: `price(...)`,
  `characteristic_function(...)`.
- **A parameter dataclass** (for example `HestonParams`) holding tensors, used by engines and
  calibration. Dataclasses are registered as pytrees so `torch.func` transforms and
  `torch.compile` see through them.
- **Simulation dynamics**: `drift_diffusion(...)` or an exact-step `step(...)` function used
  by the Monte Carlo engine.

### Instrument

An instrument is a payoff plus contract terms (strike, expiry, barrier, observation dates,
exercise style). Instruments are frozen dataclasses with a `payoff(paths)` method that maps
simulated paths to cash flows. They contain no pricing logic.

### Engine

An engine prices an instrument under a model:

```python
engine.price(instrument, model_params, market) -> Tensor
```

Engines declare which (model, instrument) pairs they support and raise a clear
`NotImplementedError` otherwise. `engines.analytic` is tried first by the convenience
function `sq.price(...)`; callers can always pick an engine explicitly.

## Two API levels

1. **Functional (low level).** Plain functions on tensors, e.g.
   `sq.models.black_scholes.price(spot=..., strike=..., ...)`. No objects, easy to `vmap`,
   compile and differentiate. This is where most of the numerical code lives.
2. **Object (high level).** Instrument, model-params and engine objects composed together,
   e.g. `sq.engines.MonteCarlo(num_paths=2**18).price(barrier_option, heston_params, market)`.
   This level is a thin layer over the functional one.

## Existing code

`black_scholes.py` at the repository root becomes `smallquant/models/black_scholes.py`.
While moving it we fix three issues:

- `torch.sqrt(torch.tensor(2.0))` creates a CPU `float32` constant, which breaks CUDA inputs
  and silently mixes precision. Use `math.sqrt(2.0)` or the shared normal CDF in
  `math/distributions.py`.
- Inputs become `float32` when passed as Python floats. Inputs are coerced to the library
  default `float64` (see [conventions.md](conventions.md)).
- Only calls are supported. Add `is_call`, a dividend yield `div`, and the put via parity.

The example code at the bottom of the file moves to `examples/`.
