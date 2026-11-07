"""Single entry point. Analytic is the default and it never falls back to Monte Carlo."""

from __future__ import annotations

from smallquant.engines.pricing import Analytic
from smallquant.market import Market


def price(instrument, model_params, market: Market, engine=None):
    """Price ``instrument`` with ``model_params`` and ``market``.

    With ``engine=None`` the analytic engine is used when it supports the pair.
    Otherwise this raises and names the pair; it does not switch to Monte Carlo.
    """
    if engine is None:
        engine = Analytic()
    if not engine.supports(instrument, model_params):
        listed = ", ".join(f"{left}/{right}" for left, right in engine.supported_pairs()) or "none"
        raise NotImplementedError(
            f"Analytic does not price {type(instrument).__name__} with {type(model_params).__name__}. "
            f"Pass an engine. Supported: {listed}"
        )
    return engine.price(instrument, model_params, market)
