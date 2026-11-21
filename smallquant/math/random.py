"""Pseudo-random normals and the QuantLib Brownian bridge."""

from __future__ import annotations

import torch


def standard_normal(
    *,
    num_paths: int,
    num_steps: int,
    dim: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    antithetic: bool = False,
    method: str = "pseudo",
) -> torch.Tensor:
    """Standard normals of shape ``[num_paths, num_steps, dim]``.

    With ``antithetic`` the second half of the paths is the negation of the
    first half, and ``num_paths`` must be even. The draw is the same tensor
    ``NormalStream.take`` returns in successive slices.
    """
    return NormalStream(
        num_paths=num_paths,
        num_steps=num_steps,
        dim=dim,
        seed=seed,
        dtype=dtype,
        device=device,
        antithetic=antithetic,
        method=method,
    ).take(num_paths)


class NormalStream:
    """Sequential slices of one ``standard_normal`` draw.

    Each ``take`` advances along the path axis. Concatenating the slices
    reproduces the single draw, including the antithetic half, without
    allocating the full tensor. Pseudo-random blocks have a fixed path count:
    PyTorch's CPU normal kernel does not resume a partial call, so the block
    boundaries stay put no matter how the caller slices. A slice in the
    antithetic half replays those same blocks from ``seed``.
    """

    def __init__(
        self,
        *,
        num_paths: int,
        num_steps: int,
        dim: int,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
        antithetic: bool = False,
        method: str = "pseudo",
    ) -> None:
        if num_paths < 1 or num_steps < 1 or dim < 1:
            raise ValueError("num_paths, num_steps and dim must be positive")
        if method not in ("pseudo", "sobol"):
            raise ValueError("method must be 'pseudo' or 'sobol'")
        if antithetic and num_paths % 2 != 0:
            raise ValueError("num_paths must be even when antithetic is on")
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dim = dim
        self.seed = seed
        self.dtype = dtype
        self.device = device
        self.antithetic = antithetic
        self.method = method
        self.base = num_paths // 2 if antithetic else num_paths
        self.pos = 0
        self._live: _PseudoCursor | None = None
        self._replay: _PseudoCursor | None = None
        if method == "sobol":
            self._engine = torch.quasirandom.SobolEngine(
                dimension=num_steps * dim, scramble=True, seed=seed
            )

    def take(self, count: int) -> torch.Tensor:
        """The next ``count`` paths. ``count`` must not run past ``num_paths``."""
        if count < 1 or self.pos + count > self.num_paths:
            raise ValueError("take runs past the end of the stream")
        parts: list[torch.Tensor] = []
        left = count
        while left:
            if self.pos < self.base:
                n = min(left, self.base - self.pos)
                parts.append(self._draw_base(n))
                self.pos += n
                left -= n
            else:
                n = min(left, self.num_paths - self.pos)
                lo = self.pos - self.base
                parts.append(-self._replay_base(lo, lo + n))
                self.pos += n
                left -= n
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _cursor(self, replay: bool) -> _PseudoCursor:
        if replay:
            if self._replay is None:
                self._replay = _PseudoCursor(
                    self.base, self.num_steps, self.dim, self.seed, self.dtype, self.device
                )
            return self._replay
        if self._live is None:
            self._live = _PseudoCursor(
                self.base, self.num_steps, self.dim, self.seed, self.dtype, self.device
            )
        return self._live

    def _draw_base(self, count: int) -> torch.Tensor:
        if self.method == "pseudo":
            return self._cursor(False).slice(self.pos, self.pos + count)
        return _sobol_draw(self._engine, count, self.num_steps, self.dim, self.dtype, self.device)

    def _replay_base(self, lo: int, hi: int) -> torch.Tensor:
        if self.method == "pseudo":
            return self._cursor(True).slice(lo, hi)
        engine = torch.quasirandom.SobolEngine(
            dimension=self.num_steps * self.dim, scramble=True, seed=self.seed
        )
        if lo:
            engine.fast_forward(lo)
        return _sobol_draw(engine, hi - lo, self.num_steps, self.dim, self.dtype, self.device)


class _PseudoCursor:
    """Base-path draws in fixed blocks, sliced to whatever the caller asks.

    ``slice`` only moves forward. Skipping discards blocks instead of storing them.
    """

    def __init__(
        self, base: int, num_steps: int, dim: int, seed: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        self.base = base
        self.num_steps = num_steps
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.block = 4096
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.buf: torch.Tensor | None = None
        self.buf_start = 0
        self.produced = 0
        self.served = 0

    def slice(self, start: int, end: int) -> torch.Tensor:
        if start < self.served:
            raise RuntimeError("pseudo stream was read out of order")
        while self.served < start:
            self._pull(start)
        parts: list[torch.Tensor] = []
        while self.served < end:
            parts.append(self._pull(end))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _pull(self, end: int) -> torch.Tensor:
        if self.buf is None or self.served >= self.produced:
            count = min(self.block, self.base - self.produced)
            if count < 1:
                raise RuntimeError("pseudo stream ran past the base paths")
            self.buf = torch.randn(
                count,
                self.num_steps,
                self.dim,
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            self.buf_start = self.produced
            self.produced += count
        local = self.served - self.buf_start
        take = min(end, self.produced) - self.served
        self.served += take
        piece = self.buf[local : local + take]
        if self.served == self.produced:
            self.buf = None
        return piece


def _sobol_draw(
    engine: torch.quasirandom.SobolEngine,
    count: int,
    num_steps: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Scrambled Sobol normals. The engine is drawn on CPU so the stream ignores device."""
    from .distributions import normal_icdf

    uniforms = engine.draw(count).clamp(1e-12, 1.0 - 1e-12)
    sample = normal_icdf(uniforms.to(dtype=torch.float64)).to(dtype=dtype, device=device)
    return sample.reshape(count, num_steps, dim)


def brownian_bridge(normals: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """QuantLib Brownian-bridge reordering of standard normals.

    ``normals`` has shape ``[num_paths, num_steps, dim]``. ``times`` is the
    path grid including 0, of length ``num_steps + 1``. The result has the
    same shape and contains the normalized Wiener increments
    ``(W_t - W_s) / sqrt(t - s)`` that QuantLib's ``PathGenerator`` passes to
    ``evolve``. The terminal value is driven by the first normal of each path.
    """
    if normals.ndim != 3:
        raise ValueError("normals must have shape [num_paths, num_steps, dim]")
    if times.ndim != 1:
        raise ValueError("times must be a 1-d grid")
    steps = times.numel() - 1
    if steps < 1 or normals.shape[1] != steps:
        raise ValueError("times length must be num_steps + 1")
    _require_origin(times)

    step_times = [float(value) for value in times[1:].detach().cpu().tolist()]
    layout = _bridge_layout(step_times)
    points: list[torch.Tensor | None] = [None] * steps
    points[steps - 1] = layout.std_dev[0] * normals[:, 0, :]
    for index in range(1, steps):
        left = layout.left_index[index]
        right = layout.right_index[index]
        built = layout.bridge_index[index]
        shock = layout.std_dev[index] * normals[:, index, :]
        right_level = points[right]
        if right_level is None:
            raise RuntimeError("Brownian bridge read an unbuilt point")
        if left != 0:
            left_level = points[left - 1]
            if left_level is None:
                raise RuntimeError("Brownian bridge read an unbuilt point")
            points[built] = (
                layout.left_weight[index] * left_level
                + layout.right_weight[index] * right_level
                + shock
            )
        else:
            points[built] = layout.right_weight[index] * right_level + shock
    levels = torch.stack([point for point in points if point is not None], dim=1)
    if levels.shape[1] != steps:
        raise RuntimeError("Brownian bridge left a point unbuilt")

    dt = torch.diff(times).reshape(1, steps, 1).sqrt()
    increments = torch.empty_like(levels)
    increments[:, 0, :] = levels[:, 0, :] / dt[:, 0, :]
    increments[:, 1:, :] = (levels[:, 1:, :] - levels[:, :-1, :]) / dt[:, 1:, :]
    return increments


class _BridgeLayout:
    def __init__(self, step_times: list[float]) -> None:
        size = len(step_times)
        constructed = [0] * size
        self.bridge_index = [0] * size
        self.left_index = [0] * size
        self.right_index = [0] * size
        self.left_weight = [0.0] * size
        self.right_weight = [0.0] * size
        self.std_dev = [0.0] * size

        constructed[size - 1] = 1
        self.bridge_index[0] = size - 1
        self.std_dev[0] = step_times[-1] ** 0.5
        cursor = 0
        for index in range(1, size):
            while constructed[cursor] != 0:
                cursor += 1
            right = cursor
            while constructed[right] == 0:
                right += 1
            built = cursor + ((right - 1 - cursor) >> 1)
            constructed[built] = index
            self.bridge_index[index] = built
            self.left_index[index] = cursor
            self.right_index[index] = right
            if cursor != 0:
                span = step_times[right] - step_times[cursor - 1]
                self.left_weight[index] = (step_times[right] - step_times[built]) / span
                self.right_weight[index] = (step_times[built] - step_times[cursor - 1]) / span
                self.std_dev[index] = (
                    (step_times[built] - step_times[cursor - 1])
                    * (step_times[right] - step_times[built])
                    / span
                ) ** 0.5
            else:
                self.left_weight[index] = (step_times[right] - step_times[built]) / step_times[right]
                self.right_weight[index] = step_times[built] / step_times[right]
                self.std_dev[index] = (
                    step_times[built] * (step_times[right] - step_times[built]) / step_times[right]
                ) ** 0.5
            cursor = right + 1
            if cursor >= size:
                cursor = 0


def _bridge_layout(step_times: list[float]) -> _BridgeLayout:
    return _BridgeLayout(step_times)


def _require_origin(times: torch.Tensor) -> None:
    origin = torch.zeros(1, dtype=times.dtype, device=times.device)
    if not torch.equal(times[:1], origin):
        raise ValueError("times must start at 0")
