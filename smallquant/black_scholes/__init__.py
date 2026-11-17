"""Black-Scholes pricing: vanilla, binary, barrier and Asian prices, implied volatility,
a CRR tree and American approximations. Exotic closed forms are in ``smallquant.exotics``.

Laid out after TF Quant Finance's ``tf_quant_finance.black_scholes``.
"""

from . import (
    approximations,
    asian_prices,
    brownian_bridge,
    crr_binomial_tree,
    vanilla_prices,
    variance_swaps,
)
from .brownian_bridge import brownian_bridge_double, brownian_bridge_single
from .crr_binomial_tree import option_price_binomial
from .implied_vol_approximation import implied_vol as implied_vol_approx
from .implied_vol_lib import ImpliedVolMethod, implied_vol
from .implied_vol_newton_root import implied_vol as implied_vol_newton
from .implied_vol_utils import UnderlyingDistribution as ImpliedVolUnderlyingDistribution
from .vanilla_prices import asset_or_nothing_price, barrier_price, binary_price, option_price
from .variance_swaps import fair_strike as variance_swap_fair_strike

__all__ = [
    "ImpliedVolMethod",
    "ImpliedVolUnderlyingDistribution",
    "approximations",
    "asian_prices",
    "asset_or_nothing_price",
    "barrier_price",
    "binary_price",
    "brownian_bridge",
    "brownian_bridge_double",
    "brownian_bridge_single",
    "crr_binomial_tree",
    "implied_vol",
    "implied_vol_approx",
    "implied_vol_newton",
    "option_price",
    "option_price_binomial",
    "vanilla_prices",
    "variance_swap_fair_strike",
    "variance_swaps",
]
