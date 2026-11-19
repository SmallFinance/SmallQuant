# SmallQuant

PyTorch-native derivatives pricing. A price is a tensor computation, so Greeks come from autograd, and the same code runs on CPU, CUDA and Apple MPS.

## Install

```bash
uv sync
```

PyTorch is the only runtime dependency. QuantLib is used by the tests.

## Price a contract

```python
import torch
import smallquant as sq
from smallquant.instruments import EuropeanOption, ForwardStartOption
from smallquant.models.params import BlackScholesParams

dtype = torch.float64
market = sq.Market(
    spot=torch.tensor(100.0, dtype=dtype),
    rate=torch.tensor(0.05, dtype=dtype),
    div=torch.tensor(0.02, dtype=dtype),
)
params = BlackScholesParams(vol=torch.tensor(0.2, dtype=dtype))
call = torch.tensor(True)

european = EuropeanOption(
    strike=torch.tensor(100.0, dtype=dtype),
    expiry=torch.tensor(1.0, dtype=dtype),
    is_call=call,
)
forward = ForwardStartOption(
    moneyness=torch.tensor(1.1, dtype=dtype),
    reset=torch.tensor(0.5, dtype=dtype),
    expiry=torch.tensor(1.0, dtype=dtype),
    is_call=call,
)
print(float(sq.price(european, params, market)))
print(float(sq.price(forward, params, market)))
```

`sq.price` uses the analytic engine. Pass `engine=sq.engines.MonteCarlo(...)` when the contract needs a simulation. With `BlackScholesParams` the analytic engine also prices digitals, continuous barriers, geometric Asians, lookbacks, compound options, choosers, cliquets and double barriers. Set `BarrierOption(..., monitoring="continuous")` for the closed-form barrier; the default monitoring is discrete and is priced by Monte Carlo.

The build scope is `docs/specs.md`. European Monte Carlo timings are in `docs/european-monte-carlo.md`.
