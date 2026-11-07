"""Autograd Greeks. Prices are differentiated, never bumped, except ``bump``."""

from __future__ import annotations

import torch


def first_order(price: torch.Tensor, inputs: dict[str, torch.Tensor], *, sum_batch: bool = True):
    """Gradients of ``price.sum()``.

    An input with one element per price receives the per-option gradient. A
    shared input receives the portfolio sum. With ``sum_batch=False`` a shared
    input raises; use :func:`jacobian` for per-option sensitivities of a shared
    input.

    Args:
        price: batched price that still has a gradient graph
        inputs: named tensors the price depends on
        sum_batch: when false, shared inputs are rejected

    Returns:
        Mapping of input name to gradient.
    """
    if not sum_batch:
        for name, value in inputs.items():
            shared = value.numel() != price.numel() and value.numel() != 1
            if shared or (value.numel() == 1 and price.numel() != 1):
                raise ValueError(
                    f"{name} is shared across the batch; pass sum_batch=True or use jacobian"
                )
    names = list(inputs)
    tensors = [inputs[name] for name in names]
    grads = torch.autograd.grad(price.sum(), tensors, retain_graph=True, allow_unused=True)
    return {
        name: (grad if grad is not None else torch.zeros_like(tensor))
        for name, grad, tensor in zip(names, grads, tensors)
    }


def second_order(price_fn, inputs: dict[str, torch.Tensor], pairs=None):
    """Unique second derivatives via forward-over-reverse.

    ``price_fn`` takes the same dictionary of tensors and returns a price.
    ``pairs`` lists ``(name_i, name_j)``; the default is every unique pair,
    including the pure second derivatives.
    """
    names = list(inputs)
    values = [inputs[name] for name in names]

    def wrapped(xs):
        mapped = {name: xs[index] for index, name in enumerate(names)}
        return price_fn(mapped).sum()

    hessian = torch.func.jacfwd(torch.func.grad(wrapped))(values)
    if pairs is None:
        pairs = [(left, right) for i, left in enumerate(names) for right in names[i:]]
    result = {}
    for left, right in pairs:
        i = names.index(left)
        j = names.index(right)
        row = hessian[i]
        # jacfwd of a tuple-valued grad is a tuple (or list) of Jacobians.
        block = row[j] if isinstance(row, (tuple, list)) else row
        result[(left, right)] = block
    return result


def jacobian(price_fn, inputs: dict[str, torch.Tensor]):
    """Per-option Jacobian of a vector price with respect to each input."""
    names = list(inputs)
    values = [inputs[name] for name in names]

    def wrapped(xs):
        mapped = {name: xs[index] for index, name in enumerate(names)}
        return price_fn(mapped)

    blocks = torch.func.jacrev(wrapped)(values)
    return {name: blocks[index] for index, name in enumerate(names)}


def bump(price_fn, inputs: dict[str, torch.Tensor], *, rel: float = 1e-4, scheme: str = "central"):
    """Finite-difference check. This is not a pricing Greek.

    Example:
        >>> def f(data):
        ...     return data["spot"] * data["spot"]
        >>> bump(f, {"spot": torch.tensor(3.0)})["spot"]
    """
    if scheme not in ("central", "forward"):
        raise ValueError("scheme must be 'central' or 'forward'")
    base = price_fn(inputs)
    out = {}
    for name, value in inputs.items():
        step = torch.where(value == 0, torch.full_like(value, rel), value.abs() * rel)
        up = dict(inputs)
        up[name] = value + step
        if scheme == "forward":
            out[name] = (price_fn(up) - base) / step
        else:
            down = dict(inputs)
            down[name] = value - step
            out[name] = (price_fn(up) - price_fn(down)) / (2.0 * step)
    return out
