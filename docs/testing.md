# Testing

A pricing library is only useful if its numbers are right. Every feature ships with tests
of these kinds.

## Test layout

```
tests/
├── math/          # distributions, roots, interpolation, linear algebra, random
├── rates/
├── vol/
├── models/
├── engines/
├── greeks/
└── conftest.py    # device and dtype parametrisation
```

Run with `pytest`. Tests are parametrised over available devices (`cpu`, plus `cuda` / `mps`
when present) and over `float64` / `float32` where float32 is supported.

## Kinds of tests

1. **Published reference values.** Closed forms are checked against values from standard
   sources, with the source cited in the test. Example: Black-Scholes call with spot 100,
   strike 100, one year, 5% rate, 20% vol is 10.4506 (Hull, *Options, Futures, and Other
   Derivatives*). Each model's test file lists its references.
2. **Independent implementation.** Where `scipy` or QuantLib provides the same function,
   results are compared to it on random inputs. These are test-only dependencies.
3. **No-arbitrage identities.** Put-call parity, monotonicity in strike, convexity in
   strike, price bounds, zero-vol and zero-expiry limits.
4. **Cross-engine agreement.** The same instrument priced by the analytic, PDE, Fourier,
   lattice and Monte Carlo engines must agree: deterministic engines to a stated tolerance,
   Monte Carlo within 3 standard errors.
5. **Greeks.** Autograd Greeks vs closed-form Greeks, and vs central-difference bumps
   (common random numbers for Monte Carlo). `torch.autograd.gradcheck` and `gradgradcheck`
   in `float64` on every differentiable public function.
6. **Batching and devices.** A batched call equals the stack of individual calls; CPU and
   GPU results agree to dtype tolerance; outputs land on the input device with the expected
   dtype.
7. **Compilation.** Core functions run under `torch.compile(fullgraph=True)` without graph
   breaks and give the same results as eager mode.

## Tolerances

- Closed forms in `float64`: relative 1e-10 against references computed to full precision;
  absolute 1e-4 against published values that are printed to four decimal places.
- `float32`: relative 1e-5.
- Monte Carlo: within 3 standard errors, with a fixed seed so failures are reproducible.

## Benchmarks

`benchmarks/` holds timing scripts (chain pricing, implied vol over a surface, Heston Monte
Carlo with Greeks) run on demand, not in CI. Benchmark numbers in documentation are measured
by these scripts and labelled with hardware and versions; none are quoted before they are
measured.
