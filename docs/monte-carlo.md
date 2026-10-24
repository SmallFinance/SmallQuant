# Monte Carlo

## Path layout

Simulated paths are tensors of shape `[num_paths, num_steps + 1, dim]`:

- `num_paths` first, so reductions over paths are `mean(dim=0)` and chunking slices dim 0.
- Index `0` on the time axis is the initial state.
- `dim` is the number of state variables (1 for Black-Scholes, 2 for Heston: price and
  variance). Single-factor models still return the trailing dimension of size 1.

Paths are simulated in log space where the model allows, to keep prices positive and the
scheme stable.

## Time grids

The engine merges the instrument's observation and exercise times with an optional maximum
step size into one sorted time grid, so no date is missed and coarse grids stay accurate.

## Discretisation schemes

| Model | Default scheme |
| --- | --- |
| Black-Scholes | Exact lognormal step |
| Bachelier | Exact normal step |
| Merton jump | Exact step with Poisson jump counts |
| Heston | Andersen QE (quadratic-exponential) for variance, with martingale correction |
| Local vol | Log-Euler with surface interpolation at each step |
| Hull-White | Exact Gaussian step of the short rate and its integral |

Euler is available for every model as a reference.

## Random numbers

- **Pseudo-random:** `torch.Generator` on the target device, seeded by `seed=`. Same seed,
  device and dtype give the same paths.
- **Quasi-random:** `torch.quasirandom.SobolEngine` with scrambling, mapped to normals with
  the inverse normal CDF. Uses a **Brownian bridge** to assign the best Sobol dimensions to
  the largest-scale path features, which is what makes Sobol effective on long paths.
- Error estimates for Sobol use several independent scramblings (randomised QMC), because
  the sample standard error does not apply to quasi-random points.

## Variance reduction

- **Antithetic variates** (`antithetic=True`, default): each normal draw is paired with its
  negative.
- **Control variates** (`control_variate=`): a correlated instrument with a known price,
  e.g. the geometric Asian for the arithmetic Asian, or the European option for a barrier.
  The optimal coefficient is estimated from the same paths.
- **Moment matching** (optional): rescale the normals to exact mean 0 and variance 1.

## Memory and chunking

A Heston simulation with 2^20 paths, 252 steps and 2 state variables in `float64` is about
4 GB of path data, and reverse-mode AD keeps intermediate values on top of that.

- `chunk_size=` simulates and prices paths in chunks, accumulating the payoff sum (and its
  gradient) chunk by chunk. Results are identical to the unchunked run with the same seed.
- Payoffs that only need running statistics (running max for lookbacks, running sum for
  Asians, barrier hit flag) are computed on the fly, so full paths are never stored.
- `torch.utils.checkpoint` over time blocks trades recomputation for memory when Greeks are
  needed on very long paths.

## Standard errors

With `return_stderr=True`, `engine.price` returns `(price, stderr)`. Tests and examples
always report Monte Carlo results with their standard error, never as bare numbers.
