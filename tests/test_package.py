"""Public package surface."""

import subprocess
import sys

import smallquant as sq


def test_models_reachable_from_top_level():
    value = sq.models.black_scholes.price(
        spot=100.0, strike=100.0, expiry=1.0, rate=0.05, vol=0.2
    )
    assert value.item() > 0


def test_core_does_not_import_time():
    code = "import sys, smallquant; assert 'smallquant.time' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
