"""The :class:`PortalAdapter` Strategy interface.

Every supported property portal is implemented as one concrete subclass that
bundles all three site-specific behaviours behind a single object:

- :meth:`build_search_url` — turn resolved search params into a portal URL.
- :meth:`parse_card`       — shallow-parse one search-results card into a stub.
- :meth:`parse_detail`     — deep-parse a detail page into a :class:`Listing`.

The orchestrator in :mod:`src.tools.scraper` treats every adapter uniformly:
it never branches on the portal name. Adding a portal is a new subclass plus
one entry in :data:`src.tools.scraper.adapters.ADAPTERS` — nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.state.listings import Listing
from src.tools.scraper.core import _PROPERTY_TYPE_MAP


class PortalAdapter(ABC):
    """Strategy interface shared by every property-portal scraper.

    Subclasses set the four class attributes below and implement the three
    abstract methods. :meth:`type_slug` and :meth:`matches_host` are concrete
    defaults so neither the orchestrator nor subclasses re-implement the
    property-type lookup or host-routing logic.
    """

    #: Portal slug — used for logging, ``Listing.source_site``, and the id prefix.
    name: str
    #: Index into a :data:`_PROPERTY_TYPE_MAP` value tuple selecting this
    #: portal's spelling of a property type (e.g. FR ``apartamentos`` vs MC
    #: ``apartamento``).
    slug_field: int
    #: CSS selector matching one search-results card.
    card_selector: str
    #: ``netloc`` substrings that route a detail URL to this adapter.
    hosts: tuple[str, ...]

    def type_slug(self, property_type: str) -> str:
        """Resolve a user-facing property type to this portal's URL slug."""
        return _PROPERTY_TYPE_MAP[property_type][self.slug_field]

    def matches_host(self, host: str) -> bool:
        """True if ``host`` (a lower-cased netloc) belongs to this portal."""
        return any(h in host for h in self.hosts)

    @abstractmethod
    def build_search_url(
        self,
        slug: str,
        transaction: str,
        location: str,
        filters: dict[str, int],
        zone: str | None = None,
    ) -> str:
        """Build the portal's search-results URL for the given parameters.

        ``filters`` carries canonical, None-stripped, integer-valued keys
        (see :func:`src.tools.scraper.core._collect_filters`). ``zone`` is the
        optional sub-municipal slug (e.g. ``"chapinero"``); each adapter slots
        it into its own portal-specific position.
        """

    @abstractmethod
    def parse_card(self, card: Any, base_url: str) -> dict | None:
        """Shallow-parse one search card into a ``{"id", "url", "price"}`` stub.

        Returns ``None`` when the card lacks a usable link.
        """

    @abstractmethod
    def parse_detail(self, page: Any, url: str) -> Listing | None:
        """Deep-parse a fetched detail page into a fully validated ``Listing``."""
