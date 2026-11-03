"""Black-Scholes-Merton closed form for European options."""

import math

import torch


def _as_tensors(*values):
    """Coerce inputs to tensors sharing dtype/device; default dtype is float64."""
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    if tensors:
        dtype = tensors[0].dtype
        device = tensors[0].device
        for t in tensors[1:]:
            if t.dtype != dtype or t.device != device:
                raise ValueError("all tensor inputs must share the same dtype and device")
    else:
        dtype = torch.float64
        device = None
    return tuple(torch.as_tensor(v, dtype=dtype, device=device) for v in values)


def price(*, spot, strike, expiry, rate, vol, div=0.0, is_call=True):
    """European Black-Scholes-Merton price.

    Args:
        spot: underlying spot
        strike: strike
        expiry: time to expiry in years
        rate: continuously compounded risk-free rate
        vol: volatility
        div: continuous dividend / carry yield
        is_call: True for call, False for put (may be a bool or a boolean tensor)

    Returns:
        Option price tensor, broadcast from the inputs.

    At ``expiry <= 0`` or ``vol <= 0`` the price is its limit, the discounted forward
    intrinsic value ``max(+/-(spot * e^{-div T} - strike * e^{-rate T}), 0)``, with finite
    gradients.
    """
    spot, strike, expiry, rate, vol, div = _as_tensors(
        spot, strike, expiry, rate, vol, div
    )

    # Mask degenerate inputs before sqrt/division so the unused branch has no NaN gradient.
    degenerate = (expiry <= 0) | (vol <= 0)
    safe_expiry = torch.where(degenerate, torch.ones_like(expiry), expiry)
    safe_vol = torch.where(degenerate, torch.ones_like(vol), vol)

    sqrt_t = torch.sqrt(safe_expiry)
    d1 = (torch.log(spot / strike) + (rate - div + 0.5 * safe_vol**2) * safe_expiry) / (
        safe_vol * sqrt_t
    )
    d2 = d1 - safe_vol * sqrt_t

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    nd1 = 0.5 * (1.0 + torch.erf(d1 * inv_sqrt2))
    nd2 = 0.5 * (1.0 + torch.erf(d2 * inv_sqrt2))

    expiry_or_zero = torch.clamp(expiry, min=0.0)
    discounted_spot = spot * torch.exp(-div * expiry_or_zero)
    discounted_strike = strike * torch.exp(-rate * expiry_or_zero)
    call = discounted_spot * nd1 - discounted_strike * nd2
    put = discounted_strike * (1.0 - nd2) - discounted_spot * (1.0 - nd1)

    call_limit = torch.relu(discounted_spot - discounted_strike)
    put_limit = torch.relu(discounted_strike - discounted_spot)
    call = torch.where(degenerate, call_limit, call)
    put = torch.where(degenerate, put_limit, put)

    if isinstance(is_call, torch.Tensor):
        return torch.where(is_call, call, put)
    return call if is_call else put
