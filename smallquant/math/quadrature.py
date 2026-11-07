"""Quadrature rules. Nodes are constants, so the integral differentiates through ``f``."""

from __future__ import annotations

import torch


def gauss_legendre(function, a, b, n: int = 20):
    """Gauss-Legendre integral of ``function`` on ``[a, b]``.

    ``function`` receives the nodes stacked on a new trailing axis and returns
    values of that shape.
    """
    nodes, weights = _legendre(n, a if isinstance(a, torch.Tensor) else torch.as_tensor(a))
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    sample = mid.unsqueeze(-1) + half.unsqueeze(-1) * nodes
    values = function(sample)
    return half * (values * weights).sum(dim=-1)


def gauss_laguerre(function, n: int = 15, scale: torch.Tensor | None = None):
    """Gauss-Laguerre integral of ``function(x) * exp(-x)`` on ``[0, inf)``."""
    device = scale.device if isinstance(scale, torch.Tensor) else torch.device("cpu")
    dtype = scale.dtype if isinstance(scale, torch.Tensor) else torch.float64
    nodes, weights = _laguerre(n, dtype, device)
    if scale is not None:
        nodes = nodes * scale.unsqueeze(-1)
        values = function(nodes) * scale.unsqueeze(-1)
    else:
        values = function(nodes)
    return (values * weights).sum(dim=-1)


def simpson(function, a, b, n: int = 128):
    """Composite Simpson integral. ``n`` is rounded up to an even count."""
    steps = n + (n % 2)
    grid = torch.linspace(0, 1, steps + 1, dtype=_dtype(a), device=_device(a))
    sample = a.unsqueeze(-1) + (b - a).unsqueeze(-1) * grid
    values = function(sample)
    width = (b - a) / steps
    odd = values[..., 1:-1:2].sum(dim=-1)
    even = values[..., 2:-1:2].sum(dim=-1)
    return width / 3.0 * (values[..., 0] + values[..., -1] + 4.0 * odd + 2.0 * even)


def _dtype(value):
    return value.dtype if isinstance(value, torch.Tensor) else torch.float64


def _device(value):
    return value.device if isinstance(value, torch.Tensor) else torch.device("cpu")


def _legendre(n: int, like: torch.Tensor):
    key = (n, like.dtype, like.device)
    cached = _LEGENDRE.get(key)
    if cached is None:
        nodes, weights = _legendre_nodes(n, like.dtype, like.device)
        _LEGENDRE[key] = (nodes, weights)
        cached = _LEGENDRE[key]
    return cached


def _laguerre(n, dtype, device):
    key = (n, dtype, device)
    cached = _LAGUERRE.get(key)
    if cached is None:
        _LAGUERRE[key] = _laguerre_nodes(n, dtype, device)
    return _LAGUERRE[key]


_LEGENDRE: dict = {}
_LAGUERRE: dict = {}


def _legendre_nodes(n, dtype, device):
    # Newton on the Legendre polynomial via the recurrence. Nodes are constants.
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    guess = torch.cos(torch.pi * (k - 0.25) / (n + 0.5))
    for _ in range(12):
        value, derivative = _legendre_pair(guess, n)
        guess = guess - value / derivative
    nodes = guess
    _, derivative = _legendre_pair(nodes, n)
    weights = 2.0 / ((1.0 - nodes**2) * derivative**2)
    order = torch.argsort(nodes)
    return nodes[order], weights[order]


def _legendre_pair(x, n):
    previous = torch.ones_like(x)
    current = x.clone()
    for k in range(2, n + 1):
        nxt = ((2 * k - 1) * x * current - (k - 1) * previous) / k
        previous, current = current, nxt
    derivative = n * (previous - x * current) / (1.0 - x * x).clamp(min=1e-18)
    return current, derivative


def _laguerre_nodes(n, dtype, device):
    k = torch.arange(1, n + 1, dtype=dtype, device=device)
    guess = (torch.pi * (k - 0.25)) ** 2 / (4.0 * n)
    # Golub-Welsch on the Jacobi matrix is stable for modest n.
    diagonal = 2.0 * k - 1.0
    off = torch.sqrt(k[:-1])
    matrix = torch.diag(diagonal) + torch.diag(off, 1) + torch.diag(off, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    weights = eigenvectors[0, :] ** 2
    order = torch.argsort(eigenvalues)
    return eigenvalues[order], weights[order]
