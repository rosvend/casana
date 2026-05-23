"""Decoupled web-search abstraction.

Exposes a backend-agnostic :class:`SearchProvider` Strategy and a
:func:`get_search_provider` factory. ``news_agent`` searches through this
layer so the backend (DuckDuckGo today; SearXNG / Tavily / Crawl4AI later)
can be swapped and A/B-compared without touching agent code.

    from src.tools.search import get_search_provider
    hits = get_search_provider().search("seguridad Medellin 2026")
"""

from __future__ import annotations

import logging
import os

from src.tools.search.base import SearchProvider, SearchResult
from src.tools.search.providers import PROVIDERS

logger = logging.getLogger(__name__)

#: Production default — picked by ``scripts/benchmark_search`` as the best
#: balance of latency (~0.7s/query) and content depth (~320-char summaries).
DEFAULT_PROVIDER = "tavily"
#: Provider activated by the LOCAL_SEARCH feature toggle — the zero-cost,
#: self-hosted fallback (SearXNG container + Crawl4AI sandbox).
LOCAL_PROVIDER = "searxng_crawl4ai"


def _truthy(value: str | None) -> bool:
    """``LOCAL_SEARCH``-style env var: yes / true / 1 / on (case-insensitive)."""
    return (value or "").strip().lower() in {"1", "yes", "true", "on"}


def get_search_provider(name: str | None = None) -> SearchProvider:
    """Return a :class:`SearchProvider` instance.

    Resolution order:

    1. Explicit ``name`` argument — wins outright (used by tests / scripts).
    2. ``LOCAL_SEARCH=yes`` env var — the feature toggle that flips the agent
       onto the free self-hosted SearXNG + Crawl4AI stack.
    3. ``SEARCH_PROVIDER`` env var — explicit per-deployment override.
    4. :data:`DEFAULT_PROVIDER` — Tavily, per benchmark results.

    An unknown name falls back to the default with a warning, so a typo in
    an env var never hard-crashes the graph.
    """
    if name is None:
        if _truthy(os.getenv("LOCAL_SEARCH")):
            name = LOCAL_PROVIDER
        else:
            name = os.getenv("SEARCH_PROVIDER")
    chosen = name or DEFAULT_PROVIDER
    provider_cls = PROVIDERS.get(chosen)
    if provider_cls is None:
        logger.warning(
            "unknown search provider %r — falling back to %r", chosen, DEFAULT_PROVIDER
        )
        provider_cls = PROVIDERS[DEFAULT_PROVIDER]
    return provider_cls()


__all__ = ["SearchProvider", "SearchResult", "get_search_provider"]
