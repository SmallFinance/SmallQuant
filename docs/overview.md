# Overview

## Why SmallQuant

Pricing and risk are numerical problems with three properties that deep learning frameworks
already solve well:

1. **Sensitivities matter as much as prices.** Traders and risk systems need delta, gamma,
   vega, rho and many bucketed sensitivities. The usual approach is "bump and reprice": run
   the pricer again for each shifted input. That costs one extra valuation per sensitivity
   and gives noisy finite-difference estimates, especially under Monte Carlo. Reverse-mode
   automatic differentiation (adjoint AD) returns every first-order sensitivity for a small
   constant multiple of one valuation's cost, and the results are exact up to floating point.
2. **The work is massively parallel.** Monte Carlo paths, PDE grids, whole option chains and
   whole portfolios are independent or nearly independent calculations. GPUs execute them
   in parallel.
3. **Calibration is optimisation.** Fitting a model (Heston, SABR, local vol, Hull-White) to
   market quotes is a least-squares problem that needs gradients of model prices with respect
   to model parameters. That is exactly what an autograd framework with built-in optimisers
   provides.

Google's [TF Quant Finance](https://github.com/google/tf-quant-finance) proved the idea on
TensorFlow, but it is archived and unmaintained, and most numerical and ML work has moved to
PyTorch. Existing PyTorch efforts (for example
[torchquant](https://jialuechen.github.io/torchquant)) are useful references but small in
scope. SmallQuant aims to be the maintained, PyTorch-native equivalent of TF Quant Finance.

## Goals

- **Differentiable end to end.** Every price is a differentiable function of every market
  and model input. Greeks of any order come from autograd, not bumping.
- **Batched by default.** All functions accept tensors and broadcast, so one call prices a
  full chain, surface or portfolio.
- **Device agnostic.** The same code runs on CPU, CUDA and MPS with no changes; results land
  on the device of the inputs.
- **Accurate first, fast second.** `float64` by default, with tested reference values. Speed
  comes from batching, GPUs and `torch.compile`, never from silently dropping precision.
- **Composable.** Models, payoffs and numerical engines are separate pieces: a payoff can be
  priced by an analytic formula, Monte Carlo, a PDE solver or Fourier inversion where each
  applies.
- **Small dependency footprint.** Runtime dependency is PyTorch only.

## Non-goals

- **Trade booking and market-data plumbing.** No trade stores or market-data connectors.
  Callers pass year fractions and tensors into the pricing core. Holiday calendars and day
  counts live in the optional `smallquant.time` helper (see [time.md](time.md)); the core
  never depends on them.
- **A full QuantLib replacement.** QuantLib covers decades of instrument conventions.
  SmallQuant focuses on models and numerical methods where differentiability and parallelism
  pay off.
- **Machine-learning models of prices.** Neural pricers and deep hedging can be built on top
  of SmallQuant, but the library itself implements classical models.

## Who it is for

- Quants who want exact Greeks and fast calibration without writing adjoint code by hand.
- Risk teams running large Monte Carlo jobs (XVA, VaR, scenario sweeps) who want GPU speed.
- Researchers combining classical pricing with ML in one PyTorch graph (deep hedging,
  neural SDEs, learned vol surfaces).
