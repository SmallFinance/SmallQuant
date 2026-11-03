"""Price a small batch of European calls with Black-Scholes-Merton."""

import torch

from smallquant.models.black_scholes import price

spot = torch.tensor([100.0, 110.0, 120.0], dtype=torch.float64)
strike = torch.tensor([100.0, 100.0, 100.0], dtype=torch.float64)
expiry = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
rate = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)
vol = torch.tensor([0.2, 0.25, 0.3], dtype=torch.float64)

call_prices = price(
    spot=spot, strike=strike, expiry=expiry, rate=rate, vol=vol, is_call=True
)
print("European call prices:", call_prices)
