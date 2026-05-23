"""The :class:`SearchProvider` Strategy interface + the normalized result model.

Every supported web-search backend is implemented as one concrete
:class:`SearchProvider` subclass that hides its SDK behind a single method.
Callers (e.g. ``news_agent``) depend only on this ABC and on
:class:`SearchResult` — never on a backend SDK — so backends are swappable
and A/B-comparable without touching agent code.

This mirrors the :class:`src.tools.scraper.adapters.PortalAdapter` Strategy:
adding a backend is a new subclass plus one entry in
:data:`src.tools.search.providers.PROVIDERS` — nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One normalized web-search hit, backend-agnostic.

    Every :class:`SearchProvider` maps its backend's native payload onto this
    shape so downstream consumers see a single, stable contract regardless of
    which engine produced the hit.
    """

    title: str = Field(..., description="Result headline.")
    url: str = Field(..., description="Canonical result link.")
    snippet: str = Field(default="", description="Backend-provided text excerpt.")


class SearchProvider(ABC):
    """Strategy interface shared by every web-search backend.

    Subclasses set the ``name`` class attribute (the registry slug, also used
    for the ``SEARCH_PROVIDER`` env var and logging) and implement
    :meth:`search`.
    """

    #: Registry slug — selects this provider via the factory / ``SEARCH_PROVIDER``.
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run one query and return up to ``max_results`` normalized hits.

        Contract: this method MUST NOT raise. A failing or rate-limited
        backend returns an empty list and logs a warning, so a dead search
        engine merely degrades ``news_agent`` rather than crashing the graph.
        """
