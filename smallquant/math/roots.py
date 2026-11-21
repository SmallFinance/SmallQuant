"""Batched root finders with an implicit-function backward pass."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from smallquant._tensor import align

Residual = Callable[..., torch.Tensor]


def _tolerance(tol: torch.Tensor | None, like: torch.Tensor, default: float) -> torch.Tensor:
    if tol is None:
        return torch.full_like(like, default)
    return torch.broadcast_to(align(tol, like)[0], like.shape)


def newton(
    residual: Residual,
    x0: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    *,
    tol: torch.Tensor | None = None,
    max_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``residual(x, *params) = 0`` with a fixed number of Newton steps.

    ``residual`` is called as ``residual(x, *params)``. ``tol`` broadcasts with
    ``x0``; ``None`` is ``1e-12``. An element holds its root once its residual
    is within its own ``tol``. The iteration count does not depend on
    convergence, and nothing is copied to the host inside the loop. The
    returned root is differentiable through the implicit function theorem. The
    second return is a convergence mask.
    """
    x0, *params = align(x0, *params)
    tol_t = _tolerance(tol, x0, 1e-12)
    with torch.no_grad():
        x = x0.clone()
        value = torch.zeros_like(x)
        settled = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        for _ in range(max_iter):
            value, slope = _value_and_slope(residual, x, params)
            settled = settled | (value.abs() <= tol_t)
            step = value / slope.clamp(min=1e-14)
            x = torch.where(settled, x, x - step)
        value, _ = _value_and_slope(residual, x, params)
        converged = value.abs() <= tol_t
    return _implicit(residual, x, params), converged


def brent(
    residual: Residual,
    lo: torch.Tensor,
    hi: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    *,
    tol: torch.Tensor | None = None,
    max_iter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bracketed Brent/bisection root of ``residual(x, *params)``.

    ``lo`` and ``hi`` must have opposite signs of the residual. ``tol``
    broadcasts with the bracket; ``None`` is ``1e-12``. An element holds its
    bracket once either end is within its own ``tol``. Every element runs
    ``max_iter`` steps. The root uses the implicit function theorem.
    """
    lo, hi, *params = align(lo, hi, *params)
    tol_t = _tolerance(tol, lo, 1e-12)
    with torch.no_grad():
        a = lo.clone()
        b = hi.clone()
        fa, _ = _value_and_slope(residual, a, params)
        fb, _ = _value_and_slope(residual, b, params)
        settled = torch.zeros(a.shape, dtype=torch.bool, device=a.device)
        for _ in range(max_iter):
            settled = settled | (fa.abs() <= tol_t) | (fb.abs() <= tol_t)
            mid = 0.5 * (a + b)
            # Inverse quadratic when the three points are distinct, else bisection.
            fm, _ = _value_and_slope(residual, mid, params)
            use_left = fa * fm <= 0
            b = torch.where(settled, b, torch.where(use_left, mid, b))
            a = torch.where(settled, a, torch.where(use_left, a, mid))
            fb = torch.where(settled, fb, torch.where(use_left, fm, fb))
            fa = torch.where(settled, fa, torch.where(use_left, fa, fm))
            # A Newton step from the midpoint, rejected when it leaves the bracket.
            slope_ok = fm.abs() > 0
            newton = mid - fm / torch.where(slope_ok, _slope_only(residual, mid, params), torch.ones_like(mid))
            inside = (newton > torch.minimum(a, b)) & (newton < torch.maximum(a, b))
            trial = torch.where(inside, newton, 0.5 * (a + b))
            ft, _ = _value_and_slope(residual, trial, params)
            use_left = fa * ft <= 0
            b = torch.where(settled, b, torch.where(use_left, trial, b))
            a = torch.where(settled, a, torch.where(use_left, a, trial))
            fb = torch.where(settled, fb, torch.where(use_left, ft, fb))
            fa = torch.where(settled, fa, torch.where(use_left, fa, ft))
        root = 0.5 * (a + b)
        value, _ = _value_and_slope(residual, root, params)
        converged = value.abs() <= tol_t
    return _implicit(residual, root, params), converged


def _slope_only(residual: Residual, x: torch.Tensor, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    _, slope = _value_and_slope(residual, x, params)
    return slope


def _value_and_slope(
    residual: Residual, x: torch.Tensor, params: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        point = x.detach().requires_grad_(True)
        value = residual(point, *[p.detach() for p in params])
        (slope,) = torch.autograd.grad(value.sum(), point, retain_graph=False)
    return value.detach(), slope.detach()


def _implicit(residual: Residual, root: torch.Tensor, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
    class _Bound(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, root_value: torch.Tensor, *param_values: torch.Tensor) -> torch.Tensor:
            ctx.residual = residual
            ctx.save_for_backward(root_value, *param_values)
            return root_value

        @staticmethod
        def backward(ctx: Any, grad_root: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            root_value, *param_values = ctx.saved_tensors
            with torch.enable_grad():
                point = root_value.detach().requires_grad_(True)
                leaves = tuple(param.detach().requires_grad_(True) for param in param_values)
                value = ctx.residual(point, *leaves)
                grads = torch.autograd.grad(value.sum(), (point, *leaves))
            scale = grad_root / grads[0].clamp(min=1e-14)
            return (None, *tuple(-grad * scale for grad in grads[1:]))

    return _Bound.apply(root, *params)
