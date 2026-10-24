import torch
import torch.nn.functional as F

def black_scholes(S, K, T, r, sigma):
    """
    Computes the Black-Scholes price for European call options, supporting vectorized inputs.
    
    Args:
        S: torch.Tensor, stock prices
        K: torch.Tensor, strike prices
        T: torch.Tensor, times to maturity
        r: torch.Tensor, risk-free interest rates
        sigma: torch.Tensor, volatilities of the stock
    
    Returns:
        call_price: torch.Tensor, the prices of the European call options
    """
    # Ensure all inputs are tensors and support broadcasting
    S, K, T, r, sigma = map(torch.as_tensor, (S, K, T, r, sigma))
    
    # Calculate d1 and d2
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    
    # Calculate N(d1) and N(d2) using the cumulative distribution function (CDF) of the normal distribution
    Nd1 = 0.5 * (1 + torch.erf(d1 / 2.0 ** 0.5))
    Nd2 = 0.5 * (1 + torch.erf(d2 / 2.0 ** 0.5))
    
    # Calculate the call option prices using the Black-Scholes formula
    call_price = S * Nd1 - K * torch.exp(-r * T) * Nd2

    return call_price


if __name__ == "__main__":
    # Example vector inputs
    S = torch.tensor([100.0, 110.0, 120.0])  # Stock prices
    K = torch.tensor([100.0, 100.0, 100.0])  # Strike prices
    T = torch.tensor([1.0, 0.5, 0.25])       # Times to maturity (in years)
    r = torch.tensor([0.05, 0.05, 0.05])     # Risk-free interest rates
    sigma = torch.tensor([0.2, 0.25, 0.3])   # Volatilities

    # Calculate the option prices
    call_prices = black_scholes(S, K, T, r, sigma)
    print("European Call Option Prices:", call_prices)
