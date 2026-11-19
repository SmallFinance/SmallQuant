"""Brownian-bridge no-touch probabilities.

Closed forms are checked against brute-force simulation of Brownian bridges on
a fine grid. A grid observes the path only at its nodes, so it misses some
touches; Broadie, Glasserman and Kou (1997) show this equals continuous
monitoring of a barrier moved away from the path by ``0.5826 * sqrt(dt_var)``,
where ``dt_var`` is the variance of one fine step. The simulation is compared
with the closed form at that shifted barrier, within 4 standard errors plus a
small allowance for the remaining O(dt) bias.
"""

import math

import pytest
import torch

from smallquant.black_scholes import brownian_bridge, brownian_bridge_double, brownian_bridge_single
from smallquant.payoffs import barrier_survival

F64 = torch.float64
BGK = 0.5826  # -zeta(1/2) / sqrt(2 pi)

NUM_BRIDGES = 100_000
NUM_STEPS = 1_000
CHUNK = 10_000


def t(value, dtype=F64):
    return torch.tensor(value, dtype=dtype)


def _simulate_extremes(x_start, x_end, variance, seed):
    """Min and max over the grid of ``NUM_BRIDGES`` bridges, ``NUM_STEPS`` steps each."""
    gen = torch.Generator().manual_seed(seed)
    step_std = math.sqrt(variance / NUM_STEPS)
    frac = torch.linspace(0.0, 1.0, NUM_STEPS + 1, dtype=F64)
    lows, highs = [], []
    for _ in range(NUM_BRIDGES // CHUNK):
        steps = torch.randn(CHUNK, NUM_STEPS, generator=gen, dtype=F64) * step_std
        walk = torch.cat([torch.zeros(CHUNK, 1, dtype=F64), steps.cumsum(dim=1)], dim=1)
        bridge = x_start + walk - frac * walk[:, -1:] + frac * (x_end - x_start)
        lows.append(bridge.min(dim=1).values)
        highs.append(bridge.max(dim=1).values)
    return torch.cat(lows), torch.cat(highs), math.sqrt(variance / NUM_STEPS)


def _mc(no_touch):
    p = no_touch.to(F64).mean().item()
    se = math.sqrt(max(p * (1.0 - p), 1e-12) / no_touch.numel())
    return p, se


# ---------------------------------------------------------------- reference values


def test_tf_docstring_example():
    x_start = t([[4.5, 4.5, 4.5], [4.5, 4.6, 4.7]])
    x_end = t([[5.0, 4.9, 4.8], [4.8, 4.9, 5.0]])
    variance = t([[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]])
    double = brownian_bridge_double(
        x_start=x_start,
        x_end=x_end,
        variance=variance,
        upper_barrier=t(5.1),
        lower_barrier=t(4.4),
        n_cutoff=3,
    )
    single = brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, barrier=t(5.1))
    expected_double = t([[0.45842169, 0.21510919, 0.52704599], [0.09394963, 0.73302813, 0.22595022]])
    expected_single = t([[0.69880579, 0.69880579, 0.97267628], [0.69880579, 0.86466472, 0.32967995]])
    torch.testing.assert_close(double, expected_double, atol=1e-8, rtol=0)
    torch.testing.assert_close(single, expected_single, atol=1e-8, rtol=0)


# ---------------------------------------------------------------- single barrier


@pytest.mark.parametrize(
    "x_start, x_end, barrier, variance",
    [
        (0.0, 0.1, 0.5, 0.25),  # upper barrier
        (0.2, -0.1, 0.4, 0.1),  # upper barrier, close
        (0.0, 0.1, -0.4, 0.25),  # lower barrier
        (0.3, 0.0, -0.2, 0.16),  # lower barrier, close
    ],
)
def test_single_vs_simulation(x_start, x_end, barrier, variance):
    low, high, step_std = _simulate_extremes(x_start, x_end, variance, seed=11)
    above = barrier > x_start
    no_touch = (high < barrier) if above else (low > barrier)
    p_mc, se = _mc(no_touch)
    shifted = barrier + BGK * step_std if above else barrier - BGK * step_std
    p_shifted = brownian_bridge_single(
        x_start=t(x_start), x_end=t(x_end), variance=t(variance), barrier=t(shifted)
    ).item()
    assert abs(p_mc - p_shifted) < 4.0 * se + 1e-3
    # The unshifted closed form is the continuous limit; the grid sees fewer touches.
    p_exact = brownian_bridge_single(
        x_start=t(x_start), x_end=t(x_end), variance=t(variance), barrier=t(barrier)
    ).item()
    assert p_exact <= p_mc + 4.0 * se
    assert abs(p_mc - p_exact) < 0.03


def test_single_wrong_side_and_degenerate_variance():
    x_start = t([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    x_end = t([0.2, 0.8, 0.2, 0.0, 0.2, 0.6])
    variance = t([0.1, 0.1, 0.0, 0.1, 0.1, 0.0])
    barrier = t([0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    out = brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, barrier=barrier)
    # same side, straddle, zero variance same side, straddle from above, on barrier,
    # zero variance straddle
    assert out[0] > 0 and out[0] < 1
    torch.testing.assert_close(out[1:], t([0.0, 1.0, 0.0, 0.0, 0.0]))


def test_single_symmetric_in_side():
    # Reflecting everything through 0 must not change the probability.
    up = brownian_bridge_single(x_start=t(0.1), x_end=t(0.2), variance=t(0.3), barrier=t(0.6))
    down = brownian_bridge_single(x_start=t(-0.1), x_end=t(-0.2), variance=t(0.3), barrier=t(-0.6))
    torch.testing.assert_close(up, down)


# ---------------------------------------------------------------- double barrier


def test_double_converges_in_n_cutoff():
    x_start = t([[0.0, 0.3], [-0.4, 0.1]])
    x_end = t([[0.2, -0.3], [0.4, 0.0]])
    variance = t([[0.5, 1.0], [2.0, 4.0]])  # corridor width d = 1
    kwargs = dict(x_start=x_start, x_end=x_end, variance=variance, upper_barrier=t(0.5), lower_barrier=t(-0.5))
    reference = brownian_bridge_double(n_cutoff=40, **kwargs)
    errors = [(brownian_bridge_double(n_cutoff=n, **kwargs) - reference).abs().max().item() for n in range(9)]
    # Terms decay like exp(-2 k^2 d^2 / v): the error falls faster than geometrically.
    assert all(later < earlier for earlier, later in zip(errors[:8], errors[1:8]))
    assert errors[3] < 2e-3  # the default n_cutoff, at v as large as 4 d^2
    assert errors[8] < 1e-14


@pytest.mark.parametrize("far_side", ["lower", "upper"])
def test_double_reduces_to_single(far_side):
    x_start = t([0.0, 0.1, -0.2, 0.3])
    x_end = t([0.1, -0.1, 0.25, 0.3])
    variance = t([0.05, 0.2, 0.1, 0.4])
    if far_side == "lower":
        upper, lower, near = t(0.5), t(-50.0), t(0.5)
    else:
        upper, lower, near = t(50.0), t(-0.5), t(-0.5)
    double = brownian_bridge_double(
        x_start=x_start, x_end=x_end, variance=variance, upper_barrier=upper, lower_barrier=lower
    )
    single = brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, barrier=near)
    torch.testing.assert_close(double, single, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "x_start, x_end, lower, upper, variance",
    [
        (0.0, 0.1, -0.5, 0.5, 0.2),
        (0.3, -0.2, -0.4, 0.45, 0.15),
        (0.0, 0.0, -0.6, 0.6, 0.8),
    ],
)
def test_double_vs_simulation(x_start, x_end, lower, upper, variance):
    low, high, step_std = _simulate_extremes(x_start, x_end, variance, seed=23)
    p_mc, se = _mc((low > lower) & (high < upper))
    p_shifted = brownian_bridge_double(
        x_start=t(x_start),
        x_end=t(x_end),
        variance=t(variance),
        upper_barrier=t(upper + BGK * step_std),
        lower_barrier=t(lower - BGK * step_std),
    ).item()
    assert abs(p_mc - p_shifted) < 4.0 * se + 1e-3


def test_double_in_unit_interval():
    gen = torch.Generator().manual_seed(5)
    x_start = torch.rand(400, 50, generator=gen, dtype=F64) * 2.4 - 1.2
    x_end = torch.rand(400, 50, generator=gen, dtype=F64) * 2.4 - 1.2
    variance = torch.rand(400, 50, generator=gen, dtype=F64) * 5.0
    variance[0, :5] = 0.0
    for n in (0, 1, 3, 6):
        out = brownian_bridge_double(
            x_start=x_start,
            x_end=x_end,
            variance=variance,
            upper_barrier=t(1.0),
            lower_barrier=t(-1.0),
            n_cutoff=n,
        )
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()
    outside = (x_start.abs() >= 1.0) | (x_end.abs() >= 1.0)
    assert (out[outside] == 0).all()


def test_double_rejects_bad_n_cutoff():
    kwargs = dict(x_start=t(0.0), x_end=t(0.0), variance=t(0.1), upper_barrier=t(1.0), lower_barrier=t(-1.0))
    for bad in (-1, 1.5, True, t(3)):
        with pytest.raises(ValueError):
            brownian_bridge_double(n_cutoff=bad, **kwargs)


# ---------------------------------------------------------------- tensor contract


def test_broadcasting_paths_steps():
    num_paths, num_steps = 7, 5
    gen = torch.Generator().manual_seed(1)
    x_start = torch.rand(num_paths, num_steps, generator=gen, dtype=F64) * 0.4 - 0.2
    x_end = torch.rand(num_paths, num_steps, generator=gen, dtype=F64) * 0.4 - 0.2
    variance = torch.linspace(0.01, 0.1, num_steps, dtype=F64)  # per step, shared by paths
    upper = t([[0.3]] * num_paths)  # per path
    lower = t(-0.3)
    double = brownian_bridge_double(
        x_start=x_start, x_end=x_end, variance=variance, upper_barrier=upper, lower_barrier=lower
    )
    single = brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, barrier=upper)
    assert double.shape == (num_paths, num_steps)
    assert single.shape == (num_paths, num_steps)
    for i in range(num_paths):
        for j in range(num_steps):
            one = brownian_bridge_double(
                x_start=x_start[i, j],
                x_end=x_end[i, j],
                variance=variance[j],
                upper_barrier=t(0.3),
                lower_barrier=lower,
            )
            torch.testing.assert_close(double[i, j], one)


def test_gradients_finite_including_masked():
    x_start = t([0.0, 0.0, 0.6, 0.0, 0.0]).requires_grad_()
    x_end = t([0.1, 0.1, 0.1, 0.7, 0.1]).requires_grad_()
    variance = t([0.2, 0.0, 0.2, 0.2, 1e-30]).requires_grad_()
    upper = t(0.5).requires_grad_()
    lower = t(-0.5).requires_grad_()
    double = brownian_bridge_double(
        x_start=x_start, x_end=x_end, variance=variance, upper_barrier=upper, lower_barrier=lower
    )
    single = brownian_bridge_single(x_start=x_start, x_end=x_end, variance=variance, barrier=upper)
    (double.sum() + single.sum()).backward()
    for tensor in (x_start, x_end, variance, upper, lower):
        assert torch.isfinite(tensor.grad).all()
    assert x_start.grad[0] != 0 and variance.grad[0] != 0 and upper.grad != 0


def test_gradcheck():
    x_start = t([0.0, 0.2]).requires_grad_()
    x_end = t([0.1, -0.1]).requires_grad_()
    variance = t([0.2, 0.3]).requires_grad_()
    upper = t(0.5).requires_grad_()
    lower = t(-0.4).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b, c, hi, lo: brownian_bridge_double(
            x_start=a, x_end=b, variance=c, upper_barrier=hi, lower_barrier=lo
        ),
        (x_start, x_end, variance, upper, lower),
    )
    assert torch.autograd.gradcheck(
        lambda a, b, c, u: brownian_bridge_single(x_start=a, x_end=b, variance=c, barrier=u),
        (x_start, x_end, variance, upper),
    )


def test_float32_kept_and_dtype_argument():
    f32 = torch.float32
    args = dict(x_start=t(0.0, f32), x_end=t(0.1, f32), variance=t(0.2, f32))
    assert brownian_bridge_single(barrier=t(0.5, f32), **args).dtype == f32
    assert brownian_bridge_double(upper_barrier=t(0.5, f32), lower_barrier=t(-0.5, f32), **args).dtype == f32
    assert brownian_bridge_single(barrier=t(0.5, f32), dtype=F64, **args).dtype == F64
    # Integer tensors default to float64.
    out = brownian_bridge_single(
        x_start=torch.tensor(0), x_end=torch.tensor(0), variance=torch.tensor(1), barrier=torch.tensor(1)
    )
    assert out.dtype == F64


def test_python_floats_rejected():
    with pytest.raises(TypeError):
        brownian_bridge_single(x_start=0.0, x_end=t(0.1), variance=t(0.2), barrier=t(0.5))
    with pytest.raises(TypeError):
        brownian_bridge_double(
            x_start=t(0.0), x_end=t(0.1), variance=t(0.2), upper_barrier=0.5, lower_barrier=t(-0.5)
        )


def test_module_exported():
    assert brownian_bridge.brownian_bridge_single is brownian_bridge_single
    assert brownian_bridge.brownian_bridge_double is brownian_bridge_double


# ---------------------------------------------------------------- payoffs.py agreement


@pytest.mark.parametrize("kind", ["down_and_out", "up_and_out"])
def test_single_matches_payoffs_barrier_survival(kind):
    """``payoffs.barrier_survival`` inlines the same bridge probability in log space.

    With ``strike = 0`` and a call, its payoff is ``S_T * prod(survive)``, so dividing
    by ``S_T`` recovers the product of per-step survival probabilities.
    """
    gen = torch.Generator().manual_seed(3)
    num_paths, num_steps = 64, 12
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=F64)
    vol = t(0.25)
    shocks = torch.randn(num_paths, num_steps, generator=gen, dtype=F64) * vol * math.sqrt(1.0 / num_steps)
    log_paths = torch.cat([torch.zeros(num_paths, 1, dtype=F64), shocks.cumsum(dim=1)], dim=1)
    spot = 100.0 * torch.exp(log_paths)
    barrier = t(80.0) if kind == "down_and_out" else t(125.0)

    payoff = barrier_survival(
        spot.unsqueeze(-1), times, strike=t(0.0), is_call=t(True, torch.bool), barrier=barrier, vol=vol, kind=kind
    )
    theirs = payoff / spot[:, -1]

    variance = vol * vol * (times[1:] - times[:-1])
    per_step = brownian_bridge_single(
        x_start=torch.log(spot[:, :-1]),
        x_end=torch.log(spot[:, 1:]),
        variance=variance,
        barrier=torch.log(barrier),
    )
    ours = per_step.prod(dim=-1)
    # Some paths must breach so the knocked-out branch is exercised too.
    assert (ours == 0).any() and (ours > 0).any()
    torch.testing.assert_close(ours, theirs, atol=1e-12, rtol=1e-10)
