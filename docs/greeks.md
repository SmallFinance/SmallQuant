# Greeks

Sensitivities are the main reason SmallQuant exists. This document explains how they are
computed and where the hard cases are.

## Autograd instead of bumping

Bump-and-reprice computes delta as `(V(S + h) - V(S - h)) / 2h`. That needs two extra
valuations per input, the step `h` trades truncation error against rounding error, and under
Monte Carlo the estimate is noisy unless the same random numbers are reused.

SmallQuant prices with differentiable tensor operations, so:

- **Reverse mode** (`price.backward()` / `torch.autograd.grad`) returns the gradient with
  respect to *all* inputs for a small constant multiple of one valuation. This is the
  "adjoint AD" (AAD) banks implement by hand, for free.
- **Forward mode** (`torch.func.jvp` / `jacfwd`) is cheaper when there are few inputs and many
  outputs, e.g. delta of each option in a large chain with respect to one spot.
- **Second order** (gamma, vanna, volga) uses forward-over-reverse
  (`torch.func.jacfwd(torch.func.grad(f))`), costing roughly one extra pass per direction.

## The helpers

```python
# Scalar or batched price already computed with inputs that require grad
g = sq.greeks.first_order(price, {"spot": spot, "vol": vol})
```

- If `price` has a batch dimension, each input's gradient is of `price.sum()`. When inputs
  are batched one-to-one with prices (one spot per option) that is exactly the per-option
  delta, because options do not depend on each other's inputs.
- When an input is shared across the batch (one scalar `vol`), the gradient is the
  sum of per-option sensitivities, which is the portfolio vega. Use `sq.greeks.jacobian` for
  per-option values in that case.

```python
# Second order needs the pricing function, not just the result
def f(spot, vol):
    return sq.models.black_scholes.price(spot=spot, strike=100.0, expiry=1.0,
                                         rate=0.05, vol=vol)

h = sq.greeks.second_order(f, {"spot": spot, "vol": vol},
                           pairs=[("spot", "spot"), ("spot", "vol"), ("vol", "vol")])
h["spot", "spot"]  # gamma
h["spot", "vol"]   # vanna
h["vol", "vol"]    # volga
```

Named Greeks map to inputs:

| Greek | Derivative |
| --- | --- |
| delta | dV/d spot |
| gamma | d²V/d spot² |
| vega | dV/d vol |
| theta | -dV/d expiry (sign flipped: value change as time passes) |
| rho | dV/d rate |
| vanna | d²V/d spot d vol |
| volga (vomma) | d²V/d vol² |

## Monte Carlo Greeks

Differentiating a Monte Carlo estimator through autograd gives the **pathwise** estimator:
the derivative of each path's discounted payoff, averaged. It is unbiased when the payoff is
Lipschitz continuous in the input (vanillas, Asians, most smooth payoffs) and has much lower
variance than bumping.

It fails when the payoff jumps:

- **Digitals and barriers.** The payoff is a step function of the terminal or path value, so
  its pathwise derivative is zero almost everywhere and autograd returns zero delta. That is
  wrong, not just noisy.
- **Gamma of a vanilla.** `max(S - K, 0)` has a kink, so the second pathwise derivative is
  zero almost everywhere.

SmallQuant offers three remedies, selected per instrument:

1. **Payoff smoothing** (`smoothing="sigmoid"`, width `eps`). Replace the step with a smooth
   approximation. Introduces a bias of order `eps` and is the simplest option. Barriers use
   a smoothed survival indicator.
2. **Likelihood ratio method** (`greeks_method="likelihood_ratio"`). Differentiate the
   density instead of the payoff. Unbiased for discontinuous payoffs, higher variance.
   Available for models with known transition densities (Black-Scholes, Bachelier).
3. **Conditional expectation / one-step survival** for discretely monitored barriers
   (Glasserman 2004, §7.2): replace the final indicator with its conditional probability
   given the previous step, making the payoff smooth in the inputs.

Every discontinuous instrument documents which of these it supports and its default.

## Early exercise

For American options priced with Longstaff-Schwartz, the exercise boundary is estimated by
regression and then held fixed (`torch.no_grad`) when computing Greeks. By the envelope
theorem the first-order sensitivity to the boundary is zero at the optimum, so this is the
standard and correct treatment for first-order Greeks.

## Differentiating through solvers

Implied volatility, root-finding, curve bootstrapping and calibration all solve
`g(x, theta) = 0` iteratively. Backpropagating through every iteration wastes memory and can
be inaccurate. SmallQuant instead attaches an implicit-function-theorem backward pass:

```
dx/dtheta = -(dg/dx)^-1 · dg/dtheta   evaluated at the solution
```

So `d implied_vol / d price = 1 / vega`, exactly, with no unrolled graph.

## Verifying Greeks

`sq.greeks.bump` exists to check autograd results in tests and in user code. Every model and
engine has tests comparing autograd Greeks with closed-form Greeks where available and with
central-difference bumps (common random numbers for Monte Carlo) otherwise. See
[testing.md](testing.md).
