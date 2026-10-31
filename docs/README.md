# SmallQuant documentation

SmallQuant is a PyTorch-native derivatives pricing library. Every price comes from a tensor
computation, so sensitivities (Greeks) are exact derivatives obtained through autograd, and
the same code runs batched on CPU, CUDA or Apple MPS.

> Status: design stage. These documents describe the library we are building.
> `smallquant.time` (dates, day counts, calendars) and `smallquant.models.black_scholes`
> exist today; see [roadmap.md](roadmap.md).

## Contents

| Document | What it covers |
| --- | --- |
| [overview.md](overview.md) | Why the library exists, goals and non-goals |
| [architecture.md](architecture.md) | Package layout and how the modules fit together |
| [conventions.md](conventions.md) | Tensor, dtype, device, broadcasting and naming rules every module follows |
| [api.md](api.md) | The planned public API, module by module, with examples |
| [greeks.md](greeks.md) | How sensitivities are computed: autograd, pathwise, likelihood ratio, smoothing |
| [monte-carlo.md](monte-carlo.md) | Path simulation, random numbers, variance reduction, memory |
| [time.md](time.md) | `smallquant.time`: dates, periods and day counters (pure Python, QuantLib-tested) |
| [testing.md](testing.md) | How correctness is verified |
| [roadmap.md](roadmap.md) | Delivery phases and what ships in each |
| [specs.md](specs.md) | Production build scope: numbered requirements, decisions and acceptance tests |

## Quick look

```python
import torch
import smallquant as sq

spot = torch.tensor([100.0, 110.0, 120.0], dtype=torch.float64, requires_grad=True)
vol = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)

price = sq.models.black_scholes.price(
    spot=spot, strike=100.0, expiry=1.0, rate=0.05, vol=vol, is_call=True
)

delta, vega = torch.autograd.grad(price.sum(), (spot, vol))
delta  # one per spot
vega   # summed over the batch because vol is a scalar
```

The planned `sq.greeks.first_order` helper wraps this call; see [greeks.md](greeks.md).
