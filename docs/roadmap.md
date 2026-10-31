# Roadmap

Each phase ends with a tagged release, docs updated and every item covered by the tests in
[testing.md](testing.md).

## Phase 0: foundation

- ~~Package skeleton (`pyproject.toml`, `smallquant/`, `tests/`, `examples/`), CI running
  `pytest` on CPU.~~ Done. Still to add: linting with `ruff`.
- `_tensor.py`: coercion, dtype and device resolution, broadcasting helpers.
- `math/distributions.py`, `math/roots.py` (batched, implicit-gradient Newton and Brent).
- ~~Move `black_scholes.py` into `smallquant/models/` with puts and dividend yield.~~
  Done. Still to add: Black76 and Bachelier.
- `greeks.first_order`, `second_order`, `jacobian`, `bump`.
- `vol.implied_vol` and `vol.implied_normal_vol`.

**Outcome:** price and get exact Greeks and implied vols for vanilla options, batched, on any
device.

## Phase 1: Monte Carlo

- `math/random.py`: pseudo-random and scrambled Sobol normals, Brownian bridge.
- `instruments/`: European, digital, barrier, Asian, lookback.
- `engines/monte_carlo.py` with antithetics, control variates, chunking, standard errors.
- Black-Scholes and Bachelier exact simulation; Merton jump diffusion.
- Smoothing and likelihood-ratio Greeks for discontinuous payoffs.

**Outcome:** exotic payoffs with pathwise Greeks on GPU.

## Phase 2: stochastic volatility and calibration

- Heston: characteristic function, `engines/fourier.py` (COS and Carr-Madan), QE simulation.
- SABR (Hagan implied vol).
- SVI / SSVI slices and surfaces.
- `calibration.calibrate` with reparameterised bounds and L-BFGS.

**Outcome:** calibrate Heston or SABR to a market vol surface on GPU and get parameter
sensitivities.

## Phase 3: PDE, lattices and early exercise

- `math/linalg.tridiagonal_solve`, `engines/pde.py` (Crank-Nicolson with Rannacher
  start-up).
- `engines/lattice.py` (CRR binomial, trinomial).
- American options: projected Crank-Nicolson and Longstaff-Schwartz.
- Dupire local vol from a surface; local-vol Monte Carlo and PDE.

**Outcome:** American and local-vol pricing with Greeks.

## Phase 4: rates

- `rates/curves.py`, differentiable `rates/bootstrap.py`.
- Hull-White one-factor: bond prices, caps/floors, swaptions (Jamshidian), simulation.
- Bucketed rate deltas via autograd through bootstrapping.

**Outcome:** curve building and short-rate model pricing with quote-level sensitivities.

## Later, not yet scheduled

- Multi-asset models with correlation (basket and spread options).
- `Schedule`, `Business252` and further country calendars in `smallquant.time` (dates, day
  counts and holiday calendars already shipped; see [time.md](time.md)).
- XVA building blocks (exposure simulation on a portfolio).
- Documentation site built from these Markdown files.
