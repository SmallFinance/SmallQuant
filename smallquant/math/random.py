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
) -> torch.Tensor:
    """Standard normals of shape ``[num_paths, num_steps, dim]``.

    With ``antithetic`` the second half of the paths is the negation of the
    first half, and ``num_paths`` must be even. One ``torch.Generator`` draw
    produces the base sample, so a later slice into chunks is the same tensor
    as the unchunked draw.
    """
    if num_paths < 1 or num_steps < 1 or dim < 1:
        raise ValueError("num_paths, num_steps and dim must be positive")
    base = num_paths
    if antithetic:
        if num_paths % 2 != 0:
            raise ValueError("num_paths must be even when antithetic is on")
        base = num_paths // 2
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample = torch.randn(base, num_steps, dim, dtype=dtype, device=device, generator=generator)
    if antithetic:
        sample = torch.cat((sample, -sample), dim=0)
    return sample


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
