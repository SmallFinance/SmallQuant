"""Single entry point. Analytic is the default and it never falls back to Monte Carlo."""

from __future__ import annotations

from typing import TypeVar

import torch

from smallquant.engines.pricing import Analytic, _Engine
from smallquant.market import Market

_I = TypeVar("_I")
_M = TypeVar("_M")


def price(
    instrument: _I, model_params: _M, market: Market, engine: _Engine | None = None
) -> torch.Tensor:
    """Price ``instrument`` with ``model_params`` and ``market``.

    With ``engine=None`` the analytic engine is used when it supports the pair.
    Otherwise this raises and names the pair; it does not switch to Monte Carlo.
    """
    if engine is None:
        engine = Analytic()
    if not engine.supports(instrument, model_params):
        listed = ", ".join(f"{left}/{right}" for left, right in engine.supported_pairs()) or "none"
        raise NotImplementedError(
            f"{type(engine).__name__} does not price {type(instrument).__name__} "
            f"with {type(model_params).__name__}. Supported: {listed}"
        )
    return engine.price(instrument, model_params, market)
