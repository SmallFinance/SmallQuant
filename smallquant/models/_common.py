"""Shared tensor helpers for closed-form prices."""

from __future__ import annotations

import math

import torch

_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)
_PI = math.pi

# Gauss-Legendre nodes and weights on [-1, 1], order 20.
_GL20_X = torch.tensor(
    [
        -9.9312859918509488e-01,
        -9.6397192727791381e-01,
        -9.1223442825132595e-01,
        -8.3911697182221878e-01,
        -7.4633190646015080e-01,
        -6.3605368072651502e-01,
        -5.1086700195082713e-01,
        -3.7370608871541955e-01,
        -2.2778585114164507e-01,
        -7.6526521133497338e-02,
        7.6526521133497338e-02,
        2.2778585114164507e-01,
        3.7370608871541955e-01,
        5.1086700195082713e-01,
        6.3605368072651502e-01,
        7.4633190646015080e-01,
        8.3911697182221878e-01,
        9.1223442825132595e-01,
        9.6397192727791381e-01,
        9.9312859918509488e-01,
    ]
)
_GL20_W = torch.tensor(
    [
        1.7614007139152753e-02,
        4.0601429800386120e-02,
        6.2672048334109054e-02,
        8.3276741576704741e-02,
        1.0193011981724050e-01,
        1.1819453196151833e-01,
        1.3168863844917655e-01,
        1.4209610931838204e-01,
        1.4917298647260382e-01,
        1.5275338713072589e-01,
        1.5275338713072589e-01,
        1.4917298647260382e-01,
        1.4209610931838204e-01,
        1.3168863844917655e-01,
        1.1819453196151833e-01,
        1.0193011981724050e-01,
        8.3276741576704741e-02,
        6.2672048334109054e-02,
        4.0601429800386120e-02,
        1.7614007139152753e-02,
    ]
)

# Drezner 1978 quadrature nodes.
_DR_X = torch.tensor([0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334])
_DR_Y = torch.tensor([0.10024215, 0.48281397, 1.06094980, 1.77972940, 2.66976040])


def as_tensors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Coerce inputs to tensors sharing dtype and device. Default dtype is float64."""
    tensors = [v for v in values if isinstance(v, torch.Tensor)]
    if tensors:
        dtype = tensors[0].dtype
        device = tensors[0].device
        for tensor in tensors[1:]:
            if tensor.dtype != dtype or tensor.device != device:
                raise ValueError("all tensor inputs must share the same dtype and device")
    else:
        dtype = torch.float64
        device = None
    return tuple(torch.as_tensor(v, dtype=dtype, device=device) for v in values)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT2))


def norm_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) * _INV_SQRT2PI


def option_sign(is_call: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return torch.where(is_call, torch.ones_like(like), -torch.ones_like(like))


def black_price(
    forward: torch.Tensor,
    strike: torch.Tensor,
    std_dev: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Black-76 price. ``std_dev`` is volatility times sqrt(expiry)."""
    sign = option_sign(is_call, forward)
    intrinsic = torch.clamp(sign * (forward - strike), min=0.0) * discount
    safe = torch.clamp(std_dev, min=1e-12)
    d1 = torch.log(forward / strike) / safe + 0.5 * safe
    d2 = d1 - safe
    formula = discount * sign * (forward * norm_cdf(sign * d1) - strike * norm_cdf(sign * d2))
    return torch.where(std_dev > 0, formula, intrinsic)


def bachelier_price(
    forward: torch.Tensor,
    strike: torch.Tensor,
    std_dev: torch.Tensor,
    discount: torch.Tensor,
    is_call: torch.Tensor,
) -> torch.Tensor:
    """Bachelier (normal) price. ``std_dev`` is normal volatility times sqrt(expiry)."""
    sign = option_sign(is_call, forward)
    gap = sign * (forward - strike)
    intrinsic = discount * torch.clamp(gap, min=0.0)
    safe = torch.clamp(std_dev, min=1e-12)
    h = gap / safe
    formula = discount * (safe * norm_pdf(h) + gap * norm_cdf(h))
    return torch.where(std_dev > 0, formula, intrinsic)


def _gl20(values_at_nodes: torch.Tensor) -> torch.Tensor:
    """Integrate values tabulated on the order-20 Gauss-Legendre nodes."""
    weights = _GL20_W.to(dtype=values_at_nodes.dtype, device=values_at_nodes.device)
    return (values_at_nodes * weights).sum(dim=-1)


def bivariate_west(x: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """West 2004 bivariate normal CDF, Genz hybrid quadrature."""
    x, y, rho = as_tensors(x, y, rho)
    rho = torch.clamp(rho, -0.999999, 0.999999)
    h = -x
    k = -y
    hk = h * k
    hs = 0.5 * (h * h + k * k)
    nodes = _GL20_X.to(dtype=x.dtype, device=x.device)

    asr = torch.asin(rho)
    sn = torch.sin(asr.unsqueeze(-1) * (-nodes + 1.0) * 0.5)
    eqn3 = torch.exp((sn * hk.unsqueeze(-1) - hs.unsqueeze(-1)) / (1.0 - sn * sn))
    bvn_low = _gl20(eqn3) * asr * (0.25 / _PI)
    bvn_low = torch.where(rho.abs() > 0, bvn_low, torch.zeros_like(bvn_low))
    bvn_low = bvn_low + norm_cdf(-h) * norm_cdf(-k)

    k_hi = torch.where(rho < 0, -k, k)
    hk_hi = h * k_hi
    ass = (1.0 - rho) * (1.0 + rho)
    a = torch.sqrt(torch.clamp(ass, min=0.0))
    bs = (h - k_hi) ** 2
    c = (4.0 - hk_hi) / 8.0
    d = (12.0 - hk_hi) / 16.0
    asr_hi = -(bs / torch.clamp(ass, min=1e-18) + hk_hi) / 2.0
    bvn_hi = torch.where(
        asr_hi > -100.0,
        a
        * torch.exp(torch.clamp(asr_hi, max=80.0))
        * (1.0 - c * (bs - ass) * (1.0 - d * bs / 5.0) / 3.0 + c * d * ass * ass / 5.0),
        torch.zeros_like(a),
    )
    b_len = torch.sqrt(bs)
    tail = torch.exp(torch.clamp(-hk_hi / 2.0, max=80.0)) * 2.506628274631 * norm_cdf(
        -b_len / torch.clamp(a, min=1e-18)
    ) * b_len * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
    bvn_hi = bvn_hi - torch.where(-hk_hi < 100.0, tail, torch.zeros_like(tail))
    half_a = a / 2.0
    xs = half_a.unsqueeze(-1) * (-nodes + 1.0)
    xs = xs * xs
    rs = torch.sqrt(torch.clamp(1.0 - xs, min=1e-18))
    asr6 = -(bs.unsqueeze(-1) / torch.clamp(xs, min=1e-18) + hk_hi.unsqueeze(-1)) / 2.0
    eqn6 = half_a.unsqueeze(-1) * torch.exp(torch.clamp(asr6, max=80.0)) * (
        torch.exp(-hk_hi.unsqueeze(-1) * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
        - (1.0 + c.unsqueeze(-1) * xs * (1.0 + d.unsqueeze(-1) * xs))
    )
    eqn6 = torch.where(asr6 > -100.0, eqn6, torch.zeros_like(eqn6))
    bvn_hi = (bvn_hi + _gl20(eqn6)) / (-2.0 * _PI)
    bvn_hi = torch.where(rho.abs() < 1.0, bvn_hi, torch.zeros_like(bvn_hi))
    positive = bvn_hi + norm_cdf(-torch.maximum(h, k_hi))
    negative = -bvn_hi
    lower = torch.where(
        k_hi > h,
        torch.where(h >= 0, norm_cdf(-h) - norm_cdf(-k_hi), norm_cdf(k_hi) - norm_cdf(h)),
        torch.zeros_like(h),
    )
    negative = negative + lower
    bvn_hi = torch.where(rho > 0, positive, negative)
    return torch.where(rho.abs() < 0.925, bvn_low, bvn_hi)


def _drezner_base(a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    rho2 = rho * rho
    scale = torch.sqrt(torch.clamp(2.0 * (1.0 - rho2), min=1e-18))
    a1 = a / scale
    b1 = b / scale
    ys = _DR_Y.to(dtype=a.dtype, device=a.device)
    xs = _DR_X.to(dtype=a.dtype, device=a.device)
    # (..., 5, 1) and (..., 1, 5)
    yi = ys.view(*([1] * a.ndim), 5, 1)
    yj = ys.view(*([1] * a.ndim), 1, 5)
    xi = xs.view(*([1] * a.ndim), 5, 1)
    xj = xs.view(*([1] * a.ndim), 1, 5)
    a1e = a1.unsqueeze(-1).unsqueeze(-1)
    b1e = b1.unsqueeze(-1).unsqueeze(-1)
    rho_e = rho.unsqueeze(-1).unsqueeze(-1)
    term = torch.exp(
        a1e * (2.0 * yi - a1e) + b1e * (2.0 * yj - b1e) + 2.0 * rho_e * (yi - a1e) * (yj - b1e)
    )
    total = (xi * xj * term).sum(dim=(-1, -2))
    return torch.sqrt(torch.clamp(1.0 - rho2, min=0.0)) / _PI * total


def bivariate_dr78(a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Drezner 1978 bivariate normal CDF."""
    a, b, rho = as_tensors(a, b, rho)
    rho = torch.clamp(rho, -0.999999, 0.999999)
    na = norm_cdf(a)
    nb = norm_cdf(b)

    def base_call(aa, bb, rr):
        return _drezner_base(aa, bb, rr)

    # Reductions of M(u, 0; r) down to the negative quadrant.
    def m_with_zero(u, r):
        # u <= 0, r <= 0
        base = base_call(torch.minimum(u, torch.zeros_like(u)), torch.zeros_like(u), torch.minimum(r, torch.zeros_like(r)))
        # u <= 0, r >= 0: N(u) - M(u, 0; -r)
        reduced_pos_rho = na_of(u) - base_call(u, torch.zeros_like(u), -r.abs())
        # u >= 0, r <= 0: N(u) + 0.5 - 1 + M(-u, 0; r)
        reduced_neg_rho = na_of(u) - 0.5 + base_call(-u, torch.zeros_like(u), -r.abs())
        # u >= 0, r >= 0: 0.5 - M(-u, 0; -r)
        both_pos = 0.5 - base_call(-u.abs(), torch.zeros_like(u), -r.abs())
        return torch.where(
            u <= 0,
            torch.where(r <= 0, base, na_of(u) - base_call(u, torch.zeros_like(u), -r)),
            torch.where(r <= 0, na_of(u) + norm_cdf(torch.zeros_like(u)) - 1.0 + base_call(-u, torch.zeros_like(u), r), both_pos),
        )

    def na_of(u):
        return norm_cdf(u)

    # Case a*b*rho > 0
    disc = torch.sqrt(torch.clamp(a * a - 2.0 * rho * a * b + b * b, min=1e-18))
    sign_a = torch.where(a > 0, torch.ones_like(a), -torch.ones_like(a))
    sign_b = torch.where(b > 0, torch.ones_like(b), -torch.ones_like(b))
    rho1 = (rho * a - b) * sign_a / disc
    rho2 = (rho * b - a) * sign_b / disc
    rho1 = torch.clamp(rho1, -0.999999, 0.999999)
    rho2 = torch.clamp(rho2, -0.999999, 0.999999)
    delta = (1.0 - sign_a * sign_b) / 4.0
    split = m_with_zero(a, rho1) + m_with_zero(b, rho2) - delta

    direct = base_call(a, b, rho)
    a_neg_b_pos = na - base_call(a, -b, -rho)
    a_pos_b_neg = nb - base_call(-a, b, -rho)
    both_pos_rho_neg = na + nb - 1.0 + base_call(-a, -b, rho)

    neg_quad = (a <= 0) & (b <= 0) & (rho <= 0)
    case_ab = (a <= 0) & (b >= 0) & (rho >= 0)
    case_ba = (a >= 0) & (b <= 0) & (rho >= 0)
    case_pp = (a >= 0) & (b >= 0) & (rho <= 0)
    case_split = a * b * rho > 0
    out = torch.where(neg_quad, direct, split)
    out = torch.where(case_ab, a_neg_b_pos, out)
    out = torch.where(case_ba, a_pos_b_neg, out)
    out = torch.where(case_pp, both_pos_rho_neg, out)
    out = torch.where(case_split, split, out)
    # Probability bounds used by QuantLib on saturated margins.
    saturated = torch.where(na >= nb, na, nb)
    small = torch.where(na <= nb, na, nb)
    out = torch.where(1.0 - saturated < 1e-15, small, out)
    out = torch.where(small < 1e-15, small, out)
    return out
