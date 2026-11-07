"""Batched root finders with an implicit-function backward pass."""

from __future__ import annotations

import torch

from smallquant._tensor import align


def newton(residual, x0, params, *, tol=1e-12, max_iter=50):
    """Solve ``residual(x, *params) = 0`` with a fixed number of Newton steps.

    ``residual`` is called as ``residual(x, *params)``. The iteration count does
    not depend on convergence, and nothing is copied to the host inside the
    loop. The returned root is differentiable through the implicit function
    theorem. The second return is a convergence mask.
    """
    x0, *params = align(x0, *params)
    with torch.no_grad():
        x = x0.clone()
        value = torch.zeros_like(x)
        for _ in range(max_iter):
            value, slope = _value_and_slope(residual, x, params)
            step = value / slope.clamp(min=1e-14)
            x = x - step
        value, _ = _value_and_slope(residual, x, params)
        converged = value.abs() <= tol
    return _implicit(residual, x, params), converged


def brent(residual, lo, hi, params, *, tol=1e-12, max_iter=100):
    """Bracketed Brent/bisection root of ``residual(x, *params)``.

    ``lo`` and ``hi`` must have opposite signs of the residual. Every element
    runs ``max_iter`` steps. The root uses the implicit function theorem.
    """
    lo, hi, *params = align(lo, hi, *params)
    with torch.no_grad():
        a = lo.clone()
        b = hi.clone()
        fa, _ = _value_and_slope(residual, a, params)
        fb, _ = _value_and_slope(residual, b, params)
        for _ in range(max_iter):
            mid = 0.5 * (a + b)
            # Inverse quadratic when the three points are distinct, else bisection.
            fm, _ = _value_and_slope(residual, mid, params)
            use_left = fa * fm <= 0
            b = torch.where(use_left, mid, b)
            a = torch.where(use_left, a, mid)
            fb = torch.where(use_left, fm, fb)
            fa = torch.where(use_left, fa, fm)
            # A Newton step from the midpoint, rejected when it leaves the bracket.
            slope_ok = fm.abs() > 0
            newton = mid - fm / torch.where(slope_ok, _slope_only(residual, mid, params), torch.ones_like(mid))
            inside = (newton > torch.minimum(a, b)) & (newton < torch.maximum(a, b))
            trial = torch.where(inside, newton, 0.5 * (a + b))
            ft, _ = _value_and_slope(residual, trial, params)
            use_left = fa * ft <= 0
            b = torch.where(use_left, trial, b)
            a = torch.where(use_left, a, trial)
            fb = torch.where(use_left, ft, fb)
            fa = torch.where(use_left, fa, ft)
        root = 0.5 * (a + b)
        value, _ = _value_and_slope(residual, root, params)
        converged = value.abs() <= tol
    return _implicit(residual, root, params), converged


def _slope_only(residual, x, params):
    _, slope = _value_and_slope(residual, x, params)
    return slope


def _value_and_slope(residual, x, params):
    with torch.enable_grad():
        point = x.detach().requires_grad_(True)
        value = residual(point, *[p.detach() for p in params])
        (slope,) = torch.autograd.grad(value.sum(), point, retain_graph=False)
    return value.detach(), slope.detach()


def _implicit(residual, root, params):
    class _Bound(torch.autograd.Function):
        @staticmethod
        def forward(ctx, root_value, *param_values):
            ctx.residual = residual
            ctx.save_for_backward(root_value, *param_values)
            return root_value

        @staticmethod
        def backward(ctx, grad_root):
            root_value, *param_values = ctx.saved_tensors
            with torch.enable_grad():
                point = root_value.detach().requires_grad_(True)
                leaves = tuple(param.detach().requires_grad_(True) for param in param_values)
                value = ctx.residual(point, *leaves)
                grads = torch.autograd.grad(value.sum(), (point, *leaves))
            scale = grad_root / grads[0].clamp(min=1e-14)
            return (None, *tuple(-grad * scale for grad in grads[1:]))

    return _Bound.apply(root, *params)
