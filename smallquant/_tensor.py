"""Dtype, device and validation contract shared by every public function.

Under ``torch.compile`` the validation flag and the default dtype are read when
the graph is traced, and they stay constant for that compiled graph.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_default_dtype = torch.float64
_validation = os.environ.get("SMALLQUANT_VALIDATE", "0") not in ("0", "", "false", "False")
_mps_warned = False

_FLOAT_RANK = {
    torch.float64: 4,
    torch.float32: 3,
    torch.bfloat16: 2,
    torch.float16: 1,
}


def set_default_dtype(dtype: torch.dtype) -> None:
    """Set the dtype used when every floating input is absent."""
    global _default_dtype
    if not dtype.is_floating_point:
        raise ValueError("default dtype must be floating")
    _default_dtype = dtype


def get_default_dtype() -> torch.dtype:
    return _default_dtype


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = get_default_dtype()
    set_default_dtype(dtype)
    try:
        yield
    finally:
        set_default_dtype(previous)


def set_validation(enabled: bool) -> None:
    """Turn value checks on or off for this process.

    Shape, dtype and device checks always run. Value checks such as ``vol > 0``
    sync the device, so they stay off unless this is set or
    ``SMALLQUANT_VALIDATE=1``. Enabling validation does not change results when
    the inputs are valid. ``torch.compile`` captures the flag at trace time.
    """
    global _validation
    _validation = bool(enabled)


def validation_enabled() -> bool:
    return _validation


@contextmanager
def validation(enabled: bool = True) -> Iterator[None]:
    previous = validation_enabled()
    set_validation(enabled)
    try:
        yield
    finally:
        set_validation(previous)


def align(*values: torch.Tensor, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, ...]:
    """Cast floating inputs onto one dtype and device.

    Non-tensors raise. Bool tensors are left as bools. Integer tensors are cast
    to the floating dtype. Mixed devices raise. Mixed floating dtypes promote
    to the widest. On MPS, a resolved ``float64`` becomes ``float32`` with one
    warning per process; an explicit ``dtype=torch.float64`` raises.
    """
    if not values:
        return ()
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"public numeric arguments must be torch.Tensor, got {type(value).__name__}"
            )
    devices = {value.device for value in values}
    if len(devices) > 1:
        listed = ", ".join(sorted(str(device) for device in devices))
        raise ValueError(f"all tensor inputs must share one device, got {listed}")
    device = values[0].device
    explicit = dtype is not None
    if dtype is None:
        floating = [value.dtype for value in values if value.is_floating_point()]
        if not floating:
            dtype = get_default_dtype()
        else:
            dtype = max(floating, key=lambda item: _FLOAT_RANK.get(item, 0))
    dtype = _mps_dtype(dtype, device, explicit)
    return tuple(_cast(value, dtype) for value in values)


def check(name: str, tensor: torch.Tensor, ok: torch.Tensor) -> None:
    """Raise when validation is on and ``ok`` is false for any element."""
    if not _validation:
        return
    count = int((~ok).sum().item())
    if count:
        raise ValueError(f"{name} failed a value check on {count} element(s)")


def _cast(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == torch.bool or value.dtype == dtype:
        return value
    return value.to(dtype=dtype)


def _mps_dtype(dtype: torch.dtype, device: torch.device, explicit: bool) -> torch.dtype:
    global _mps_warned
    if device.type != "mps" or dtype != torch.float64:
        return dtype
    if explicit:
        raise ValueError("dtype float64 is not available on MPS")
    if not _mps_warned:
        import warnings

        warnings.warn(
            "MPS has no float64; using float32 for this process",
            UserWarning,
            stacklevel=4,
        )
        _mps_warned = True
    return torch.float32
