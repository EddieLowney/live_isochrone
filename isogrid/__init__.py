"""isogrid — a live travel-time surface built under a hard API budget.

Quick start::

    from isogrid import Config, Engine

    components = Config.load("config/prototype.yaml").build()
    engine = Engine(components)
    engine.bootstrap_baseline()
    surface, report = engine.run_cycle()

Every component is swappable by name from the config file. See `registry.py`.
"""

from .config import Components, Config
from .engine import Engine
from .types import Cell, CellState, Coord, Measurement, Surface

# Import for side effects: each module registers its implementations.
from . import budget, grid, masks, reconstruct, sampling, store  # noqa: E402,F401
from .providers import synthetic  # noqa: E402,F401

from .providers import otp, raptor, synthetic_transit, tomtom  # noqa: E402,F401

try:  # optional, needs `requests`
    from .providers import mapbox  # noqa: E402,F401
except Exception:  # pragma: no cover
    pass

__all__ = [
    "Cell",
    "CellState",
    "Components",
    "Config",
    "Coord",
    "Engine",
    "Measurement",
    "Surface",
]
__version__ = "0.1.0"
