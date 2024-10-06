import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    """
    Computes the Black-Scholes price for a European call option.
    
    Args:
        S: torch.Tensor, stock price
        K: torch.Tensor, strike price
        T: torch.Tensor, time to maturity
        r: torch.Tensor, risk-free interest rate
        sigma: torch.Tensor, volatility of the stock
    
    Returns:
        call_price: torch.Tensor, the price of the European call option
    """
    # Calculate d1 and d2
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    
    # Calculate N(d1) and N(d2) using the cumulative distribution function (CDF) of the normal distribution
    Nd1 = 0.5 * (1 + torch.erf(d1 / torch.sqrt(torch.tensor(2.0))))
    Nd2 = 0.5 * (1 + torch.erf(d2 / torch.sqrt(torch.tensor(2.0))))
    
    # Calculate the call option price using the Black-Scholes formula
    call_price = S * Nd1 - K * torch.exp(-r * T) * Nd2
    return call_price


# Example inputs
S = torch.tensor([100.0])  # Stock price
K = torch.tensor([100.0])  # Strike price
T = torch.tensor([1.0])    # Time to maturity (in years)
r = torch.tensor([0.05])   # Risk-free interest rate
sigma = torch.tensor([0.2]) # Volatility

# Calculate the option price
call_price = black_scholes(S, K, T, r, sigma)
print(f"European Call Option Price: {call_price.item():.4f}")
