"""Exotic Black-Scholes closed forms, one module per family.

Built on ``smallquant.black_scholes``.
"""

from . import (
    binary_barrier_prices,
    chooser_prices,
    cliquet_prices,
    compound_prices,
    digital_american_prices,
    dividend_prices,
    double_barrier_binary_prices,
    double_barrier_prices,
    extensible_prices,
    forward_start_prices,
    lookback_prices,
    margrabe_prices,
    partial_time_barrier_prices,
    quanto_prices,
    soft_barrier_prices,
    stulz_prices,
    two_asset_barrier_prices,
    two_asset_correlation_prices,
)

__all__ = [
    "binary_barrier_prices",
    "chooser_prices",
    "cliquet_prices",
    "compound_prices",
    "digital_american_prices",
    "dividend_prices",
    "double_barrier_prices",
    "double_barrier_binary_prices",
    "extensible_prices",
    "forward_start_prices",
    "lookback_prices",
    "margrabe_prices",
    "partial_time_barrier_prices",
    "quanto_prices",
    "soft_barrier_prices",
    "stulz_prices",
    "two_asset_barrier_prices",
    "two_asset_correlation_prices",
]
