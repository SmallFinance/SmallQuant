"""Time a European call: SmallQuant Monte Carlo against QuantLib's MCEuropeanEngine.

The contract is spot 100, strike 100, one year, rate 5%, volatility 20%.
Each run uses N paths and T time steps of geometric Brownian motion.
SmallQuant is timed in float32 on CPU, at each thread count, and on MPS when
this machine has one. QuantLib stays on the CPU.

    uv run python scripts/european_mc.py
    uv run python scripts/european_mc.py --paths 100000 1000000 --steps 1 64 256
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import QuantLib as ql
import torch

from smallquant.black_scholes.vanilla_prices import option_price
from smallquant.engines.monte_carlo import price as mc_price
from smallquant.models import geometric_brownian_motion
from smallquant.payoffs import european

SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
VOL = 0.20
DIV = 0.0
EVALUATION = ql.Date(15, ql.January, 2025)


def closed_form() -> float:
    value = option_price(
        spot=torch.tensor(SPOT, dtype=torch.float64),
        strike=torch.tensor(STRIKE, dtype=torch.float64),
        expiry=torch.tensor(EXPIRY, dtype=torch.float64),
        rate=torch.tensor(RATE, dtype=torch.float64),
        vol=torch.tensor(VOL, dtype=torch.float64),
        div=torch.tensor(DIV, dtype=torch.float64),
        is_call=torch.tensor(True),
    )
    return float(value)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def smallquant_once(device: torch.device, paths: int, steps: int, seed: int) -> tuple[float, float]:
    dtype = torch.float32
    times = torch.linspace(0, EXPIRY, steps + 1, dtype=dtype, device=device)
    spot = torch.tensor(SPOT, dtype=dtype, device=device)
    rate = torch.tensor(RATE, dtype=dtype, device=device)
    vol = torch.tensor(VOL, dtype=dtype, device=device)
    div = torch.tensor(DIV, dtype=dtype, device=device)
    strike = torch.tensor(STRIKE, dtype=dtype, device=device)
    is_call = torch.tensor(True, device=device)
    market = dict(spot=spot, times=times, rate=rate, vol=vol, div=div)

    def step(normals: torch.Tensor) -> torch.Tensor:
        return geometric_brownian_motion.step(normals=normals, **market)

    def payoff(paths_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return european(paths_tensor, grid, strike=strike, is_call=is_call)

    estimate, stderr = mc_price(
        payoff,
        step,
        times=times,
        discount=torch.exp(-rate * times[-1]),
        num_paths=paths,
        seed=seed,
        antithetic=False,
        return_stderr=True,
    )
    return float(estimate), float(stderr)


def quantlib_market() -> tuple[ql.VanillaOption, ql.BlackScholesMertonProcess]:
    ql.Settings.instance().evaluationDate = EVALUATION
    day_count = ql.Actual365Fixed()

    def flat(rate: float) -> ql.YieldTermStructureHandle:
        return ql.YieldTermStructureHandle(ql.FlatForward(EVALUATION, rate, day_count, ql.Continuous))

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(SPOT)),
        flat(DIV),
        flat(RATE),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(EVALUATION, ql.NullCalendar(), VOL, day_count)),
    )
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, STRIKE),
        ql.EuropeanExercise(EVALUATION + 365),
    )
    return option, process


def quantlib_once(
    option: ql.VanillaOption,
    process: ql.BlackScholesMertonProcess,
    paths: int,
    steps: int,
    seed: int,
) -> tuple[float, float]:
    option.setPricingEngine(
        ql.MCEuropeanEngine(
            process,
            "PseudoRandom",
            timeSteps=steps,
            requiredSamples=paths,
            maxSamples=paths,
            seed=seed,
        )
    )
    return option.NPV(), option.errorEstimate()


def _time(repeats: int, run) -> tuple[float, float, float]:
    price, stderr = run()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        price, stderr = run()
        samples.append(time.perf_counter() - started)
    return price, stderr, statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, nargs="+", default=[100_000])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, os.cpu_count() or 1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if any(paths < 2 for paths in args.paths):
        raise SystemExit("--paths must be at least 2")
    if any(steps < 1 for steps in args.steps):
        raise SystemExit("--steps must be at least 1")

    print(f"closed form {closed_form():.6f}")
    print(
        f"{'engine':<12} {'device':<6} {'dtype':<8} {'threads':>7} "
        f"{'paths':>10} {'steps':>6} {'price':>10} {'stderr':>10} "
        f"{'seconds':>8} {'Msteps/s':>10} {'vs QL':>8}"
    )

    option, process = quantlib_market()
    mps = torch.device("mps") if torch.backends.mps.is_available() else None

    for paths in args.paths:
        for steps in args.steps:
            quantlib_price, quantlib_stderr, quantlib_seconds = _time(
                args.repeats,
                lambda: quantlib_once(option, process, paths, steps, args.seed),
            )
            _report(
                "quantlib", "cpu", "float64", 1, paths, steps,
                quantlib_price, quantlib_stderr, quantlib_seconds, quantlib_seconds,
            )

            for threads in args.threads:
                torch.set_num_threads(threads)
                device = torch.device("cpu")

                def run_cpu(device=device, paths=paths, steps=steps):
                    estimate, stderr = smallquant_once(device, paths, steps, args.seed)
                    _sync(device)
                    return estimate, stderr

                price, stderr, seconds = _time(args.repeats, run_cpu)
                _report(
                    "smallquant", "cpu", "float32", threads, paths, steps,
                    price, stderr, seconds, quantlib_seconds,
                )

            if mps is not None:
                def run_mps(device=mps, paths=paths, steps=steps):
                    estimate, stderr = smallquant_once(device, paths, steps, args.seed)
                    _sync(device)
                    return estimate, stderr

                run_mps()
                price, stderr, seconds = _time(args.repeats, run_mps)
                _report(
                    "smallquant", "mps", "float32", 0, paths, steps,
                    price, stderr, seconds, quantlib_seconds,
                )


def _report(
    engine: str,
    device: str,
    dtype: str,
    threads: int,
    paths: int,
    steps: int,
    price: float,
    stderr: float,
    seconds: float,
    quantlib_seconds: float,
) -> None:
    throughput = paths * steps / seconds / 1e6
    speedup = quantlib_seconds / seconds
    thread_label = "-" if threads == 0 else str(threads)
    print(
        f"{engine:<12} {device:<6} {dtype:<8} {thread_label:>7} "
        f"{paths:10d} {steps:6d} {price:10.4f} {stderr:10.4f} "
        f"{seconds:8.4f} {throughput:10.2f} {speedup:8.2f}"
    )


if __name__ == "__main__":
    main()
