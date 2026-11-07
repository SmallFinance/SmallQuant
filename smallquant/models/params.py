"""Parameter containers registered as pytrees."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from smallquant._tensor import align


def _register(cls):
    names = [field.name for field in fields(cls)]

    def flatten(value):
        return tuple(getattr(value, name) for name in names), None

    def unflatten(values, _context):
        return cls(*values)

    try:
        torch.utils._pytree.register_pytree_node(cls, flatten, unflatten)
    except ValueError:
        pass
    return cls


@_register
@dataclass
class BlackScholesParams:
    vol: torch.Tensor


@_register
@dataclass
class Black76Params:
    vol: torch.Tensor


@_register
@dataclass
class BachelierParams:
    normal_vol: torch.Tensor


@_register
@dataclass
class MertonParams:
    vol: torch.Tensor
    jump_intensity: torch.Tensor
    jump_mean: torch.Tensor
    jump_vol: torch.Tensor


@_register
@dataclass
class HestonParams:
    v0: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    sigma: torch.Tensor
    rho: torch.Tensor


@_register
@dataclass
class SABRParams:
    alpha: torch.Tensor
    beta: torch.Tensor
    rho: torch.Tensor
    nu: torch.Tensor


@_register
@dataclass
class LocalVolParams:
    """Local volatility on a strike-expiry grid, ``local_vols`` shaped ``[expiry, strike]``."""

    strikes: torch.Tensor
    expiries: torch.Tensor
    local_vols: torch.Tensor


@_register
@dataclass
class HullWhiteParams:
    a: torch.Tensor
    sigma: torch.Tensor
    short_rate: torch.Tensor


def stack_fields(params) -> torch.Tensor:
    """Concatenate floating fields of a parameter object into one vector."""
    values = []
    for field in fields(params):
        value = getattr(params, field.name)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            values.append(value.reshape(-1))
    return align(*values)[0].reshape(-1) if False else torch.cat([v.reshape(-1) for v in values])
