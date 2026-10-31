# SmallQuant

PyTorch-native derivatives pricing. Prices are tensor computations, so Greeks are exact
derivatives from autograd and the same code runs batched on CPU, CUDA or Apple MPS.

> Status: design stage. `smallquant.models.black_scholes` and `smallquant.time` (dates,
> day counts, holiday calendars) exist today. See [docs/roadmap.md](docs/roadmap.md).

## Install

```bash
uv sync          # creates .venv with torch, plus pytest and QuantLib for tests
uv run pytest    # QuantLib reference tests
```

## Quick look

```python
import torch
import smallquant as sq

spot = torch.tensor([100.0, 110.0, 120.0], dtype=torch.float64, requires_grad=True)
price = sq.models.black_scholes.price(
    spot=spot, strike=100.0, expiry=1.0, rate=0.05, vol=0.2
)
(delta,) = torch.autograd.grad(price.sum(), spot)
```

## Documentation

- [docs/README.md](docs/README.md): design documents
- [docs/specs.md](docs/specs.md): production build scope

## References

* https://jialuechen.github.io/torchquant
* https://github.com/google/tf-quant-finance
