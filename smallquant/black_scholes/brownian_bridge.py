"""No-touch probabilities of a Brownian bridge for one or two barriers.

A Monte Carlo path is observed only on its time grid. Between two grid points
``x_start`` and ``x_end`` the continuous path, conditional on those end points,
is a Brownian bridge. Multiplying the per-step no-touch probabilities below
turns a discretely monitored barrier into a continuously monitored one, and
the result is differentiable in the path (one-step survival).

Inputs typically have shape ``[num_paths, num_steps]``; every argument
broadcasts. For a log-normal model pass log prices and log barriers, with
``variance = vol**2 * dt``.

Laid out after TF Quant Finance's ``tf_quant_finance.black_scholes.brownian_bridge``.

References
----------
[1] P. Beaglehole, P. Dybvig and G. Zhou. Going to extremes: correcting
    simulation bias in exotic option valuation. Financial Analysts Journal
    53(1), 62-68, 1997.
[2] E. Gobet. Advanced Monte Carlo methods for barrier and related exotic
    options. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1265669
"""

from __future__ import annotations

import torch

from .._tensor import align

__all__ = ["brownian_bridge_double", "brownian_bridge_single"]


def brownian_bridge_single(
    *,
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    variance: torch.Tensor,
    barrier: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Probability that a Brownian bridge does not touch ``barrier``.

    The bridge runs from ``x_start`` to ``x_end`` with total variance
    ``variance``. When both end points lie strictly on the same side of the
    barrier (above or below, either works) the reflection principle gives [1]

        P(no touch) = 1 - exp(-2 (barrier - x_start)(barrier - x_end) / variance).

    When the end points straddle or touch the barrier the bridge has touched it
    and the result is 0. With ``variance == 0`` the bridge is a straight line,
    so the result is 1 on the same side and 0 otherwise.

    Args:
        x_start: Bridge value at the start of the interval.
        x_end: Bridge value at the end of the interval.
        variance: Total variance of the driving Brownian motion over the
            interval (``vol**2 * dt``). Non-negative.
        barrier: Barrier level.
        dtype: Optional dtype for the computation (TEN-02).

    Returns:
        The no-touch probability, in ``[0, 1]``, with the broadcast shape.
    """
    x_start, x_end, variance, barrier = align(x_start, x_end, variance, barrier, dtype=dtype)
    gap = (barrier - x_start) * (barrier - x_end)
    same_side = gap > 0
    positive_var = variance > 0
    valid = same_side & positive_var
    safe_gap = torch.where(valid, gap, torch.ones_like(gap))
    safe_var = torch.where(valid, variance, torch.ones_like(variance))
    survive = -torch.expm1(-2.0 * safe_gap / safe_var)
    degenerate = same_side.to(dtype=survive.dtype)
    return torch.where(valid, survive, torch.where(positive_var, torch.zeros_like(survive), degenerate))


def brownian_bridge_double(
    *,
    x_start: torch.Tensor,
    x_end: torch.Tensor,
    variance: torch.Tensor,
    upper_barrier: torch.Tensor,
    lower_barrier: torch.Tensor,
    n_cutoff: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Probability that a Brownian bridge touches neither barrier.

    The bridge runs from ``x_start`` to ``x_end`` with total variance ``v``
    inside the corridor ``(L, U)``, width ``d = U - L``. Summing the images of
    the end point reflected in both barriers gives [2]

        P(no touch) = sum_{k=-n}^{n} [ exp(-2 k d (k d + x_end - x_start) / v)
                                      - exp(-2 (k d + x_start - U)(k d + x_end - U) / v) ],

    truncated at ``n = n_cutoff``. The ``k = 0`` term is the single upper
    barrier result; the others correct for the lower barrier and for repeated
    reflections. Every exponent is non-positive inside the corridor, and the
    terms decay like ``exp(-2 k^2 d^2 / v)``, so a few terms suffice unless
    ``v`` is large compared with ``d^2``.

    End points on or outside a barrier give 0. With ``variance == 0`` the
    result is 1 strictly inside the corridor and 0 otherwise. The truncated
    sum is clamped to ``[0, 1]``.

    Args:
        x_start: Bridge value at the start of the interval.
        x_end: Bridge value at the end of the interval.
        variance: Total variance over the interval (``vol**2 * dt``).
            Non-negative.
        upper_barrier: Upper barrier ``U``.
        lower_barrier: Lower barrier ``L``, below ``upper_barrier``.
        n_cutoff: Python int; image terms ``k = -n_cutoff..n_cutoff`` are kept.
        dtype: Optional dtype for the computation (TEN-02).

    Returns:
        The no-touch probability, in ``[0, 1]``, with the broadcast shape.
    """
    if isinstance(n_cutoff, bool) or not isinstance(n_cutoff, int) or n_cutoff < 0:
        raise ValueError(f"n_cutoff must be a non-negative int, got {n_cutoff!r}")
    x_start, x_end, variance, upper, lower = align(
        x_start, x_end, variance, upper_barrier, lower_barrier, dtype=dtype
    )
    inside = (x_start > lower) & (x_start < upper) & (x_end > lower) & (x_end < upper)
    positive_var = variance > 0
    valid = inside & positive_var

    # Masked elements get a harmless point mid-corridor and unit variance so
    # neither the value nor the gradient of the discarded branch is NaN.
    middle = 0.5 * (upper + lower)
    start = torch.where(valid, x_start, middle)
    end = torch.where(valid, x_end, middle)
    var = torch.where(valid, variance, torch.ones_like(variance))
    width = upper - lower

    # Image terms on a trailing axis.
    k = torch.arange(-n_cutoff, n_cutoff + 1, dtype=start.dtype, device=start.device)
    shift = k * width.unsqueeze(-1)
    start = start.unsqueeze(-1)
    end = end.unsqueeze(-1)
    var = var.unsqueeze(-1)
    top = upper.unsqueeze(-1)
    direct = shift * (shift + end - start)
    reflected = (shift + start - top) * (shift + end - top)
    terms = torch.exp(-2.0 * direct / var) - torch.exp(-2.0 * reflected / var)
    survive = terms.sum(dim=-1).clamp(0.0, 1.0)

    degenerate = inside.to(dtype=survive.dtype)
    return torch.where(valid, survive, torch.where(positive_var, torch.zeros_like(survive), degenerate))
