"""Portal adapter registry.

Each adapter is a :class:`PortalAdapter` Strategy implementing one property
portal. :data:`ADAPTERS` is the single source of truth the orchestrator
iterates over for both discovery and enrichment — registering a new portal
means adding one entry here (and one new module beside this file).
"""

from src.tools.scraper.adapters.base import PortalAdapter
from src.tools.scraper.adapters.fincaraiz import FincaRaizAdapter
from src.tools.scraper.adapters.metrocuadrado import MetroCuadradoAdapter

#: Active adapters, in deterministic discovery order.
ADAPTERS: list[PortalAdapter] = [
    FincaRaizAdapter(),
    MetroCuadradoAdapter(),
]

__all__ = ["PortalAdapter", "ADAPTERS"]
