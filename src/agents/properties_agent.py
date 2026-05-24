"""`properties_node` — the LangGraph node that finds candidate properties.

This is the orchestration layer over the two-stage scraper in
:mod:`src.tools.scraper`. It reads the structured user brief from the shared
state, splits the constraints into two buckets, and runs a discover → enrich →
filter pipeline:

1. **URL-pushable constraints** (``location``, ``property_type``,
   ``transaction_type``, ``price``, ``bedrooms``) are handed to
   :func:`search_listings`, which bakes them straight into the portal search
   URLs — cheap, no per-property cost.
2. **In-memory constraints** (``bathrooms``, ``parking_lots``, ``estrato``,
   ``area_m2``) can't be expressed in every portal URL reliably, so they are
   enforced *after* the deep scrape, against the fully parsed ``Listing``.

The node deep-scrapes at most 5 discovered URLs — drawn evenly from every
portal and fetched concurrently — to keep latency bounded, then returns the
survivors under ``raw_listings``, the state key the downstream
``whatsapp_agent`` consumes. Finca Raíz and Metro Cuadrado are equally
important sources of truth, so neither discovery nor enrichment lets one
portal crowd the other out.

Requirements live in the state as a flat ``list[Constraint]`` keyed by a
snake_case English ``field`` name that mirrors the :class:`Listing` model.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

from src.state import Constraint, Listing, PropertyFinderState, StructuredRequirements
from src.tools import extract_property_details, search_listings

logger = logging.getLogger(__name__)

#: Deep-scraping is expensive (a stealthy Cloudflare-solving fetch each), so
#: cap how many discovered URLs we enrich per node run. The cap is split
#: evenly across sources by :func:`_balanced_shortlist`.
MAX_ENRICH = 5

# Constraint.field -> search_listings portal slug. Accepts both the canonical
# English form (matching the Listing model) and the Spanish portal form;
# search_listings itself falls back to a safe default on anything unknown.
_PROPERTY_TYPE_SLUGS: dict[str, str] = {
    "apartment": "apartamentos",
    "apartamento": "apartamentos",
    "apartamentos": "apartamentos",
    "house": "casas",
    "casa": "casas",
    "casas": "casas",
    "commercial": "locales",
    "local": "locales",
    "locales": "locales",
    "office": "oficinas",
    "oficina": "oficinas",
    "oficinas": "oficinas",
    "country_house": "fincas",
    "finca": "fincas",
    "fincas": "fincas",
}

_TRANSACTION_SLUGS: dict[str, str] = {
    "rent": "arriendo",
    "arriendo": "arriendo",
    "sale": "venta",
    "venta": "venta",
}

# Constraint.field values enforced after the deep scrape rather than via URL.
_INMEMORY_FIELDS = frozenset({"bathrooms", "parking_lots", "estrato", "area_m2"})


def _extract_params(
    requirements: StructuredRequirements | None,
) -> tuple[dict, list[Constraint]]:
    """Split the constraint list into URL params and in-memory constraints.

    Returns ``(url_params, inmemory)`` where ``url_params`` is ready to hand
    to :func:`search_listings` and ``inmemory`` is the subset of ``Constraint``
    objects enforced post-scrape (see :data:`_INMEMORY_FIELDS`).

    Missing constraints fall back to sensible defaults: ``medellin`` /
    ``apartamentos`` / ``arriendo``.
    """
    url_params: dict = {
        "location": "medellin",
        "property_type": "apartamentos",
        "transaction": "arriendo",
    }
    inmemory: list[Constraint] = []

    constraints = requirements.constraints if requirements else []
    for c in constraints:
        field = c.field

        if field == "location":
            if isinstance(c.exact_value, str) and c.exact_value.strip():
                url_params["location"] = c.exact_value.strip().lower()

        elif field == "zone":
            if isinstance(c.exact_value, str) and c.exact_value.strip():
                url_params["zone"] = c.exact_value.strip().lower()

        elif field == "property_type":
            if isinstance(c.exact_value, str):
                url_params["property_type"] = _PROPERTY_TYPE_SLUGS.get(
                    c.exact_value.strip().lower(), c.exact_value.strip().lower()
                )

        elif field == "transaction_type":
            if isinstance(c.exact_value, str):
                url_params["transaction"] = _TRANSACTION_SLUGS.get(
                    c.exact_value.strip().lower(), c.exact_value.strip().lower()
                )

        elif field == "price":
            if c.min_value is not None:
                url_params["min_price"] = int(c.min_value)
            if c.max_value is not None:
                url_params["max_price"] = int(c.max_value)
            if c.exact_value is not None and isinstance(c.exact_value, (int, float)):
                # An exact price collapses to a [value, value] band.
                url_params["min_price"] = int(c.exact_value)
                url_params["max_price"] = int(c.exact_value)

        elif field == "bedrooms":
            value = c.exact_value if c.exact_value is not None else c.min_value
            if isinstance(value, (int, float)):
                url_params["bedrooms"] = int(value)

        elif field in _INMEMORY_FIELDS:
            inmemory.append(c)

        else:
            logger.debug("ignoring unrecognized constraint field %r", field)

    return url_params, inmemory


def _constraint_ok(value: float | int | None, c: Constraint) -> bool:
    """Check a single scraped value against one constraint, leniently.

    A ``None`` value (the portal omitted the field) always passes — we won't
    discard an otherwise-good property just because a number was missing.
    """
    if value is None:
        return True
    if c.exact_value is not None and value != c.exact_value:
        return False
    if c.min_value is not None and value < c.min_value:
        return False
    if c.max_value is not None and value > c.max_value:
        return False
    return True


def _passes_inmemory(listing: Listing, inmemory: list[Constraint]) -> bool:
    """True if the listing satisfies every in-memory constraint."""
    for c in inmemory:
        value = getattr(listing, c.field, None)
        if not _constraint_ok(value, c):
            logger.info(
                "drop %s — %s=%r fails constraint %r",
                listing.id, c.field, value, c.model_dump(exclude_none=True),
            )
            return False
    return True


def _balanced_shortlist(stubs: list[dict], limit: int) -> list[dict]:
    """Pick up to ``limit`` stubs, interleaved round-robin by source site.

    Finca Raíz and Metro Cuadrado are equally important sources of truth, so
    the deep scrape must draw evenly from both. A flat ``stubs[:limit]`` slice
    would be all-Finca-Raíz (it leads the discovery order); interleaving keeps
    every portal represented.
    """
    by_source: dict[str, list[dict]] = {}
    for stub in stubs:
        source = str(stub.get("id", "")).split(":", 1)[0] or "unknown"
        by_source.setdefault(source, []).append(stub)
    shortlist: list[dict] = []
    for row in zip_longest(*by_source.values()):
        shortlist.extend(stub for stub in row if stub is not None)
    return shortlist[:limit]


def _enrich(stub: dict) -> Listing | None:
    """Deep-scrape one discovered stub into a full Listing (``None`` on failure).

    Exception-safe so it can be fanned out across a thread pool without one
    bad fetch aborting the batch.
    """
    url = stub.get("url")
    if not url:
        return None
    try:
        return extract_property_details.invoke({"url": url})
    except Exception as e:  # noqa: BLE001
        logger.warning("extract_property_details failed for %s: %s", url, e)
        return None


def properties_node(state: PropertyFinderState) -> dict:
    """Discover, enrich, and filter property listings for the user's brief.

    Reads ``state["requirements"]`` and writes ``raw_listings`` — the listings
    that survived both URL-level discovery filtering and the in-memory
    post-filter.
    """
    requirements = state.get("requirements")
    url_params, inmemory = _extract_params(requirements)
    logger.info("properties_node: url_params=%s, %d in-memory constraint(s)",
                url_params, len(inmemory))

    # 1. Discover — cheap, URL-filtered stubs. search_listings queries both
    #    portals concurrently and returns the merged stub list.
    try:
        stubs: list[dict] = search_listings.invoke(url_params)
    except Exception as e:  # noqa: BLE001 — never let a scrape kill the node
        logger.warning("search_listings failed: %s", e)
        stubs = []
    logger.info("discovered %d stub(s)", len(stubs))

    # 2. Shortlist — bound the deep-scrape cost, drawing evenly from every
    #    source so one portal can't crowd the other out.
    shortlist = _balanced_shortlist(stubs, MAX_ENRICH)

    # 3. Enrich — deep-scrape the shortlist concurrently. Each
    #    extract_property_details call is independent blocking I/O, so a thread
    #    pool overlaps the per-portal fetches instead of serializing them.
    listings: list[Listing] = []
    if shortlist:
        with ThreadPoolExecutor(max_workers=len(shortlist)) as executor:
            listings = [lst for lst in executor.map(_enrich, shortlist) if lst is not None]
    logger.info("enriched %d/%d listing(s)", len(listings), len(shortlist))

    # 4. In-memory filter — enforce the constraints URLs can't carry.
    valid_listings = [lst for lst in listings if _passes_inmemory(lst, inmemory)]
    logger.info("%d listing(s) survived the in-memory filter", len(valid_listings))

    return {"raw_listings": valid_listings}
