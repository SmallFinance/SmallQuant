# Conventions

Every module follows these rules. They are what make the library batched, device agnostic
and differentiable without per-function special cases.

## Inputs and broadcasting

- Public functions accept `torch.Tensor`, Python numbers or anything `torch.as_tensor`
  accepts.
- All tensor arguments broadcast together under standard PyTorch rules. The output shape is
  the broadcast shape of the inputs. A scalar `vol` with a `[N]` vector of strikes prices N
  options.
- Functions never loop in Python over batch elements.
- Shape-defining extra dimensions (time steps, paths, grid points) are documented per
  function and always placed in a fixed position: paths first, then time, then state
  dimension, e.g. Monte Carlo paths are `[num_paths, num_steps + 1, dim]`.

## dtype

- The library default is `torch.float64`. Option values and especially Greeks lose too much
  accuracy in `float32` (deep in- or out-of-the-money prices, second-order Greeks,
  finite-difference checks).
- Resolution order for a call: the `dtype=` argument if given, otherwise the highest
  floating dtype among the tensor inputs, otherwise `float64`. Python numbers never force
  `float32`.
- `float32` is supported and tested for Monte Carlo on GPUs, where it can be much faster.
  MPS has no `float64`, so on MPS the default falls back to `float32` with a one-time
  warning.

## Device

- Outputs live on the device of the tensor inputs. All inputs must share one device; mixing
  raises an error rather than copying silently.
- Constants are never created with a bare `torch.tensor(...)`, which lands on CPU. Use
  Python scalars in arithmetic, or create tensors with `device=` and `dtype=` taken from the
  inputs (`_tensor.py` provides helpers).

## Differentiability

- Every price is differentiable with respect to every floating-point input.
- No in-place operations on tensors that may require grad.
- No Python `if` on tensor values; use `torch.where` so batches with mixed branches work and
  `torch.compile` does not graph-break.
- Where a function is mathematically non-smooth (digital and barrier payoffs, `max(x, 0)` at
  the strike, early exercise), the docstring says so and [greeks.md](greeks.md) describes the
  available treatment.
- `torch.where` with an unsafe branch (e.g. `log` of a negative number) produces NaN
  gradients even when that branch is not selected. Mask inputs before the unsafe operation,
  not only the output.

## Naming

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

- Rates are continuously compounded unless a function says otherwise.
- Time is always a year fraction. Day counts and calendars are outside the core.
- Arguments are keyword-only after the first few, to prevent `strike`/`spot` mix-ups.

## Errors and validation

- Validation that requires reading tensor values (e.g. `vol > 0`) syncs the GPU, so it is
  off by default and enabled with `sq.set_validation(True)` or the `SMALLQUANT_VALIDATE=1`
  environment variable.
- Shape, dtype and device mismatches are checked always; they are free.

## Style

- Python 3.10+, type hints on public functions, Google-style docstrings with an `Args`,
  `Returns` and a short `Example` section.
- Formulas in docstrings cite their source (book, paper, equation number).
