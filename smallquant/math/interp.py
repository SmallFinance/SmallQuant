"""Interpolation that differentiates in the query and in the node values."""

from __future__ import annotations

import torch

from smallquant._tensor import align
from smallquant.math.tridiagonal import solve as solve_tridiagonal


def linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise linear interpolation of ``fp`` on knots ``xp``.

    Queries outside the knots stay at the end values. Gradients flow through
    ``x`` and ``fp``. Knot positions are constants.
    """
    x, xp, fp = align(x, xp, fp)
    flat = x.reshape(-1)
    index = torch.searchsorted(xp.detach(), flat.detach()).clamp(1, xp.numel() - 1)
    left = xp[index - 1]
    right = xp[index]
    weight = ((flat - left) / (right - left).clamp(min=1e-18)).clamp(0.0, 1.0)
    left_value = _take(fp, index - 1)
    right_value = _take(fp, index)
    # weight is [queries]; values are [queries, *batch] or [queries]
    value = left_value + weight.reshape(weight.shape + (1,) * (left_value.ndim - 1)) * (
        right_value - left_value
    )
    return value.reshape(*x.shape, *fp.shape[:-1]) if fp.ndim > 1 else value.reshape(x.shape)


def log_linear(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Exponential of a linear interpolant in ``log(fp)``."""
    return torch.exp(linear(x, xp, torch.log(fp.clamp(min=1e-18))))


def natural_cubic(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Natural cubic spline. The second derivative is zero at both ends."""
    x, xp, fp = align(x, xp, fp)
    spacing = xp[1:] - xp[:-1]
    slope = (fp[..., 1:] - fp[..., :-1]) / spacing
    if xp.numel() < 3:
        return linear(x, xp, fp)
    diag = (spacing[:-1] + spacing[1:]) / 3.0
    rhs = slope[..., 1:] - slope[..., :-1]
    sub = spacing[1:-1] / 6.0
    super_diag = spacing[1:-1] / 6.0
    batch = rhs.reshape(-1, rhs.shape[-1])
    zeros = torch.zeros(batch.shape[0], 1, dtype=fp.dtype, device=fp.device)
    lower_pad = torch.cat((zeros, sub.expand(batch.shape[0], -1)), dim=-1)
    upper_pad = torch.cat((super_diag.expand(batch.shape[0], -1), zeros), dim=-1)
    diag_b = diag.expand(batch.shape[0], -1)
    interior = solve_tridiagonal(lower_pad, diag_b, upper_pad, batch)
    interior = interior.reshape(rhs.shape)
    moment = torch.cat(
        (torch.zeros_like(fp[..., :1]), interior, torch.zeros_like(fp[..., :1])),
        dim=-1,
    )
    return _spline(x, xp, fp, moment)


def monotone_cubic(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Fritsch-Carlson monotone cubic Hermite spline."""
    x, xp, fp = align(x, xp, fp)
    spacing = xp[1:] - xp[:-1]
    delta = (fp[..., 1:] - fp[..., :-1]) / spacing
    tangent = torch.zeros_like(fp)
    tangent[..., 0] = delta[..., 0]
    tangent[..., -1] = delta[..., -1]
    if xp.numel() > 2:
        tangent[..., 1:-1] = 0.5 * (delta[..., :-1] + delta[..., 1:])
    alpha = torch.where(delta > 0, tangent[..., :-1] / delta, torch.zeros_like(delta))
    beta = torch.where(delta > 0, tangent[..., 1:] / delta, torch.zeros_like(delta))
    norm = torch.sqrt(alpha * alpha + beta * beta)
    shrink = torch.clamp(norm / 3.0, min=1.0)
    alpha = alpha / shrink
    beta = beta / shrink
    tangent = tangent.clone()
    tangent[..., :-1] = torch.where(delta > 0, alpha * delta, torch.zeros_like(delta))
    tangent[..., 1:] = torch.where(delta > 0, beta * delta, torch.zeros_like(delta))
    return _hermite(x, xp, fp, tangent)


def _take(fp: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if fp.ndim == 1:
        return fp[index]
    expanded = index.reshape(*([1] * (fp.ndim - 1)), -1).expand(*fp.shape[:-1], index.numel())
    return torch.gather(fp, -1, expanded).movedim(-1, 0)


def _locate(x: torch.Tensor, xp: torch.Tensor):
    flat = x.reshape(-1)
    index = torch.searchsorted(xp.detach(), flat.detach()).clamp(1, xp.numel() - 1)
    left = xp[index - 1]
    right = xp[index]
    span = (right - left).clamp(min=1e-18)
    weight = ((flat - left) / span).clamp(0.0, 1.0)
    return index, span, weight


def _reshape(value: torch.Tensor, x: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    if fp.ndim == 1:
        return value.reshape(x.shape)
    return value.reshape(*x.shape, *fp.shape[:-1])


def _spline(x, xp, fp, moment):
    index, span, weight_right = _locate(x, xp)
    weight_left = 1.0 - weight_right
    left_y = _take(fp, index - 1)
    right_y = _take(fp, index)
    left_m = _take(moment, index - 1)
    right_m = _take(moment, index)
    extra = (1,) * (left_y.ndim - 1)
    weight_left = weight_left.reshape(weight_left.shape + extra)
    weight_right = weight_right.reshape(weight_right.shape + extra)
    span = span.reshape(span.shape + extra)
    value = (
        weight_left * left_y
        + weight_right * right_y
        + ((weight_left**3 - weight_left) * left_m + (weight_right**3 - weight_right) * right_m)
        * span**2
        / 6.0
    )
    return _reshape(value, x, fp)


def _hermite(x, xp, fp, tangent):
    index, span, t = _locate(x, xp)
    left_y = _take(fp, index - 1)
    right_y = _take(fp, index)
    left_d = _take(tangent, index - 1)
    right_d = _take(tangent, index)
    extra = (1,) * (left_y.ndim - 1)
    t = t.reshape(t.shape + extra)
    span = span.reshape(span.shape + extra)
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)
    value = h00 * left_y + h10 * span * left_d + h01 * right_y + h11 * span * right_d
    return _reshape(value, x, fp)
