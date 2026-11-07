"""Calibration with bound reparameterisation and an implicit gradient."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from smallquant._tensor import align


@dataclass
class CalibrationResult:
    params: object
    loss: torch.Tensor
    converged: bool
    iterations: int
    grad_norm: torch.Tensor


def calibrate(
    model_params,
    market_quotes,
    pricer,
    *,
    loss="vol_rmse",
    optimizer="lbfgs",
    bounds=None,
    max_iter=200,
    tol=1e-8,
):
    """Fit ``model_params`` to ``market_quotes``.

    ``pricer(params)`` returns a tensor shaped like ``market_quotes``.
    ``bounds`` maps a field name to ``"positive"``, ``"correlation"`` or a
    ``(lower, upper)`` pair. Positive fields use softplus and correlations use
    tanh, so LBFGS runs unconstrained.

    The returned parameters carry a gradient to ``market_quotes`` through the
    implicit function theorem when the quotes require grad. ``converged=False``
    is a normal return.

    Example:
        >>> quotes = torch.tensor(0.2, dtype=torch.float64)
    """
    if optimizer != "lbfgs":
        raise ValueError("only optimizer='lbfgs' is implemented")
    if loss not in ("vol_rmse", "price_rmse"):
        raise ValueError("loss must be 'vol_rmse' or 'price_rmse'")
    quotes = align(market_quotes)[0]
    bounds = bounds or {}
    names = [field.name for field in fields(model_params) if isinstance(getattr(model_params, field.name), torch.Tensor)]
    raw0 = []
    for name in names:
        raw0.append(_to_unconstrained(getattr(model_params, name).detach().reshape(-1), bounds.get(name)))
    raw = torch.nn.Parameter(torch.cat(raw0))

    def rebuild(vector):
        pieces = {}
        cursor = 0
        for name in names:
            value = getattr(model_params, name)
            count = value.numel()
            piece = vector[cursor : cursor + count].reshape(value.shape)
            pieces[name] = _from_unconstrained(piece, bounds.get(name))
            cursor += count
        return replace(model_params, **pieces)

    def objective(vector):
        model = pricer(rebuild(vector))
        error = model - quotes
        mse = (error * error).mean()
        if loss == "vol_rmse":
            return torch.sqrt(mse.clamp(min=0.0))
        return torch.sqrt(mse.clamp(min=0.0))

    opt = torch.optim.LBFGS(
        [raw], max_iter=max_iter, tolerance_grad=tol, line_search_fn="strong_wolfe"
    )
    iterations = max_iter

    def closure():
        opt.zero_grad()
        value = objective(raw)
        value.backward()
        return value

    opt.step(closure)
    final = objective(raw.detach())
    grad_norm = raw.grad.detach().norm() if raw.grad is not None else final.new_zeros(())
    fitted = rebuild(raw.detach())
    if quotes.requires_grad:
        fitted = _attach_sensitivity(fitted, names, bounds, quotes, pricer, raw.detach())
    return CalibrationResult(
        params=fitted,
        loss=final.detach(),
        converged=bool(grad_norm < tol),
        iterations=iterations,
        grad_norm=grad_norm.detach(),
    )


def _to_unconstrained(value, bound):
    if bound == "positive":
        return torch.log(torch.expm1(value.clamp(min=1e-8)))
    if bound == "correlation":
        return torch.atanh(value.clamp(-0.999, 0.999))
    if isinstance(bound, tuple):
        lo, hi = bound
        unit = ((value - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
        return torch.log(unit / (1 - unit))
    return value


def _from_unconstrained(value, bound):
    if bound == "positive":
        return torch.nn.functional.softplus(value)
    if bound == "correlation":
        return torch.tanh(value)
    if isinstance(bound, tuple):
        lo, hi = bound
        unit = torch.sigmoid(value)
        return lo + (hi - lo) * unit
    return value


def _attach_sensitivity(fitted, names, bounds, quotes, pricer, raw_star):
    """theta(quotes) ≈ theta* - H^{-1} cross @ (quotes - quotes*)."""
    raw_star = raw_star.detach()

    def loss_of(raw_vector, quote_vector):
        pieces = {}
        cursor = 0
        for name in names:
            value = getattr(fitted, name)
            count = value.numel()
            piece = raw_vector[cursor : cursor + count].reshape(value.shape)
            pieces[name] = _from_unconstrained(piece, bounds.get(name))
            cursor += count
        model = pricer(replace(fitted, **pieces))
        error = model - quote_vector.reshape(model.shape)
        return (error * error).mean()

    raw_leaf = raw_star.detach().requires_grad_(True)
    quote_leaf = quotes.detach().reshape(-1).requires_grad_(True)
    value = loss_of(raw_leaf, quote_leaf)
    (grad_raw,) = torch.autograd.grad(value, raw_leaf, create_graph=True)
    hessian_rows = []
    cross_rows = []
    for index in range(grad_raw.numel()):
        raw_row, quote_row = torch.autograd.grad(grad_raw[index], (raw_leaf, quote_leaf), retain_graph=True)
        hessian_rows.append(raw_row)
        cross_rows.append(quote_row)
    hessian = torch.stack(hessian_rows)
    cross = torch.stack(cross_rows)
    hessian = hessian + torch.eye(hessian.shape[0], dtype=hessian.dtype, device=hessian.device) * 1e-8
    sensitivity = -torch.linalg.solve(hessian, cross)
    delta = (quotes - quotes.detach()).reshape(-1)
    adjusted = raw_star + sensitivity @ delta
    pieces = {}
    cursor = 0
    for name in names:
        value = getattr(fitted, name)
        count = value.numel()
        piece = adjusted[cursor : cursor + count].reshape(value.shape)
        pieces[name] = _from_unconstrained(piece, bounds.get(name))
        cursor += count
    return replace(fitted, **pieces)
