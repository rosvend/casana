"""Discover & Enrich scraping tools for Finca Raíz and Metro Cuadrado.

Exposes two ``@tool``-decorated callables:

- :func:`search_listings` — *discoverer*. Hits a portal's search-results page
  and returns lightweight ``{"id", "url", "price"}`` dicts. Cheap.
- :func:`extract_property_details` — *enricher*. Routes a single property URL
  to the correct site-specific parser, deep-scrapes the detail page, and
  returns a fully validated :class:`Listing`.

The two-stage split lets an agent shortlist before paying the stealthy-fetch
cost per property. Both tools degrade gracefully: missing DOM nodes yield
``None`` fields, never exceptions.

Architecture: portal-specific logic lives behind the :class:`PortalAdapter`
Strategy in :mod:`src.tools.scraper.adapters`; shared infrastructure lives in
:mod:`src.tools.scraper.core`. This module is strictly orchestration — it
iterates :data:`ADAPTERS` uniformly and never branches on a portal name.

Run the package directly to exercise the full pipeline against live sites:

    uv run python -m src.tools.scraper
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langchain_core.tools import tool
from pydantic import ValidationError

from src.state.listings import Listing
from src.tools.scraper.adapters import ADAPTERS, PortalAdapter
from src.tools.scraper.core import (
    _PROPERTY_TYPE_MAP,
    _TRANSACTION_MAP,
    _collect_filters,
    _fetch_page,
    _passes_filters,
    _safe,
)

logger = logging.getLogger(__name__)

MAX_LISTINGS_PER_SOURCE = 10


def _discover_one(
    adapter: PortalAdapter,
    location: str,
    property_type: str,
    transaction: str,
    filters: dict[str, int],
    zone: str | None = None,
) -> list[dict]:
    """Run one portal's discovery pass: build URL, fetch, shallow-parse cards."""
    name = adapter.name
    slug = adapter.type_slug(property_type)
    url = _safe(
        adapter.build_search_url, slug, transaction, location, filters,
        default=None, zone=zone,
    )
    if not url:
        logger.warning("[%s] url builder failed — skipping", name)
        return []
    logger.info("[%s] discovering: %s", name, url)

    try:
        page = _fetch_page(url)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] fetch failed: %s", name, e)
        return []

    status = getattr(page, "status", None)
    if isinstance(status, int) and status >= 400:
        logger.warning("[%s] HTTP %s — skipping", name, status)
        return []

    try:
        cards = page.css(adapter.card_selector)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] selector %r failed: %s", name, adapter.card_selector, e)
        return []
    logger.info("[%s] %d card(s) matched", name, len(cards))

    out: list[dict] = []
    seen_urls: set[str] = set()
    for card in cards:
        if len(out) >= MAX_LISTINGS_PER_SOURCE:
            break
        rec = _safe(adapter.parse_card, card, url)
        if not rec or rec["url"] in seen_urls:
            continue
        if not _passes_filters(rec, filters):
            continue
        seen_urls.add(rec["url"])
        out.append(rec)
    return out


@tool
def search_listings(
    location: str = "medellin",
    property_type: str = "apartamentos",
    transaction: str = "arriendo",
    zone: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    estrato: int | None = None,
    min_area_m2: int | None = None,
    max_area_m2: int | None = None,
    parking_lots: int | None = None,
    longevity: int | None = None,
) -> list[dict]:
    """Discover candidate property URLs across Finca Raíz and Metro Cuadrado.

    Returns a lightweight list of ``{"id", "url", "price"}`` dicts — enough to
    deduplicate and shortlist before paying the deep-scrape cost.

    Filters are pushed into the portal URL when the portal supports them
    (verified live), then a post-filter pass enforces ``min_price``/``max_price``
    against the shallow card's parsed price as a safety net.

    Per-portal URL filter support:

    - **Finca Raíz**: min/max price, bedrooms, bathrooms, estrato, min/max
      area, parking. Longevity is post-filter only.
    - **Metro Cuadrado**: max price + price range, bedrooms (single),
      bathrooms (single), estrato (single), parking (single), area range.
      Min-only price, min-only area, and longevity are post-filter only.

    Args:
        location: City slug, e.g. ``"medellin"``, ``"bogota"``.
        property_type: One of ``"apartamentos"``, ``"casas"``, ``"locales"``,
            ``"oficinas"``, ``"fincas"``. Unknown values fall back to apartamentos.
        transaction: ``"arriendo"`` (rent) or ``"venta"`` (sale).
        zone: Optional sub-municipal slug, e.g. ``"chapinero"``. When set, each
            portal slots it into its own URL position (Finca Raíz: before the
            city; Metro Cuadrado: after the city).
        min_price: Minimum asking price in COP.
        max_price: Maximum asking price in COP.
        bedrooms: Exact bedroom count.
        bathrooms: Exact bathroom count.
        estrato: Colombian socio-economic stratum (1-6).
        min_area_m2: Minimum built area in m².
        max_area_m2: Maximum built area in m².
        parking_lots: Exact parking space count.
        longevity: Property age in years; currently post-filter only and
            effectively unenforceable until the deep parser surfaces it.
    """
    if property_type not in _PROPERTY_TYPE_MAP:
        logger.warning("unknown property_type %r — falling back to 'apartamentos'", property_type)
        property_type = "apartamentos"
    if transaction not in _TRANSACTION_MAP:
        logger.warning("unknown transaction %r — falling back to 'arriendo'", transaction)
        transaction = "arriendo"

    filters = _collect_filters(
        min_price=min_price,
        max_price=max_price,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        estrato=estrato,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
        parking_lots=parking_lots,
        longevity=longevity,
    )

    # Each portal is an independent blocking fetch — discover them concurrently
    # so neither source waits on the other. Order is preserved (executor.map
    # yields in submission order) so results stay deterministic.
    with ThreadPoolExecutor(max_workers=len(ADAPTERS)) as executor:
        per_adapter = list(executor.map(
            lambda adapter: _discover_one(
                adapter, location, property_type, transaction, filters, zone=zone
            ),
            ADAPTERS,
        ))
    results: list[dict] = [rec for sublist in per_adapter for rec in sublist]
    logger.info("discovered %d total listing stub(s) across %d source(s)",
                len(results), len(ADAPTERS))
    return results


@tool
def extract_property_details(url: str) -> Listing | None:
    """Deep-scrape a single property page and return a validated Listing.

    Routes by URL host: Finca Raíz vs Metro Cuadrado. Returns ``None`` if the
    page can't be fetched, the URL is from an unsupported host, or validation
    fails. Individual missing fields surface as ``None`` on the Listing, not
    as exceptions.

    Args:
        url: A property detail URL produced by :func:`search_listings`.
    """
    if not url:
        logger.warning("extract_property_details called with empty url")
        return None

    host = urlparse(url).netloc.lower()
    adapter = next((a for a in ADAPTERS if a.matches_host(host)), None)
    if adapter is None:
        logger.warning("unsupported host %r — cannot route deep scrape", host)
        return None
    source = adapter.name

    logger.info("[%s] enriching: %s", source, url)
    try:
        page = _fetch_page(url)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] fetch failed: %s", source, e)
        return None

    status = getattr(page, "status", None)
    if isinstance(status, int) and status >= 400:
        logger.warning("[%s] HTTP %s — aborting", source, status)
        return None

    try:
        listing = adapter.parse_detail(page, url)
    except ValidationError as ve:
        logger.warning("[%s] Listing validation failed: %s", source, ve)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] deep-scrape parser raised: %s", source, e)
        return None

    if listing is None:
        logger.warning("[%s] parser returned None", source)
    return listing


__all__ = ["search_listings", "extract_property_details"]
