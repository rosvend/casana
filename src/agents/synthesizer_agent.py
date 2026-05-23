"""`synthesizer_node` — the dedupe-and-merge LangGraph node.

Sits between the parallel fetch branches (properties/whatsapp + news) and the
evaluator. Pure Python: no LLM, no network. Two passes:

1. **Intra-portal dedup** — collapse duplicate ``listing.id`` from the same
   portal (last write wins; ``id`` is namespaced ``{source_site}:{listing_id}``).
2. **Cross-portal fuzzy merge** — greedy cluster the unique listings by a
   conjunction of heuristics (property/transaction type, bedrooms, price within
   5%, area within 10% when known, same zone, coordinates within 100m when
   both have them). Each cluster becomes one :class:`Candidate` carrying the
   safest field values across its members (max price, max bathrooms, max
   parking) and the full list of contributing source URLs.

The full ``news_results`` dict is attached to every candidate's
``relevant_news``: ``news_agent`` already targets the user's single zone so
all items are relevant; per-candidate zone filtering would currently drop most
of them. Revisit if ``news_agent`` later produces multi-zone results.
"""

from __future__ import annotations

import logging
import math

from src.state import Candidate, Listing, NewsResults, PropertyFinderState, VerifiedListing

logger = logging.getLogger(__name__)

#: Maximum |Δprice| / max(price) allowed when matching across portals.
_PRICE_TOLERANCE = 0.05
#: Maximum |Δarea| / max(area) allowed when matching across portals.
_AREA_TOLERANCE = 0.10
#: Maximum Haversine distance (meters) allowed between two listings' coordinates.
_COORD_MAX_M = 100.0
#: Earth radius used by the Haversine formula.
_EARTH_RADIUS_M = 6_371_000.0


def synthesizer_node(state: PropertyFinderState) -> dict:
    """Merge raw listings into candidates and attach news context.

    Reads ``state["raw_listings"]`` and ``state["news_results"]``. Returns
    ``{"candidates": list[Candidate]}``. An empty listings input yields an
    empty candidate list — never raises.
    """
    raw_listings: list[Listing] = state.get("raw_listings") or []
    news_results: NewsResults = state.get("news_results") or {}

    if not raw_listings:
        logger.info("synthesizer_node: no raw_listings to merge")
        return {"candidates": []}

    unique = _dedup_by_id(raw_listings)
    groups = _group_by_fuzzy_match(unique)
    candidates = [_merge_group(g, news_results) for g in groups]

    logger.info(
        "synthesizer_node: %d raw -> %d unique -> %d candidate(s)",
        len(raw_listings), len(unique), len(candidates),
    )
    return {"candidates": candidates}


def _dedup_by_id(listings: list[Listing]) -> list[Listing]:
    """Collapse duplicates sharing the same ``listing.id`` (last write wins)."""
    by_id: dict[str, Listing] = {}
    for listing in listings:
        by_id[listing.id] = listing
    return list(by_id.values())


def _group_by_fuzzy_match(listings: list[Listing]) -> list[list[Listing]]:
    """Greedy clustering: append each listing to the first group it matches."""
    groups: list[list[Listing]] = []
    for listing in listings:
        for group in groups:
            if _is_match(listing, group[0]):
                group.append(listing)
                break
        else:
            groups.append([listing])
    return groups


def _is_match(a: Listing, b: Listing) -> bool:
    """All heuristics must hold for two listings to belong in the same cluster."""
    if not _norm_eq(a.property_type, b.property_type):
        return False
    if not _norm_eq(a.transaction_type, b.transaction_type):
        return False
    if a.bedrooms is None or b.bedrooms is None or a.bedrooms != b.bedrooms:
        return False
    if not _within_ratio(a.price, b.price, _PRICE_TOLERANCE, required=True):
        return False
    if not _within_ratio(a.area_m2, b.area_m2, _AREA_TOLERANCE, required=False):
        return False
    if not _norm_eq(a.zone, b.zone):
        return False
    if not _coords_close_enough(a.coordinates, b.coordinates):
        return False
    return True


def _norm_eq(a: str | None, b: str | None) -> bool:
    """Case/whitespace-insensitive string equality. Both must be present."""
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


def _within_ratio(
    a: float | None, b: float | None, tolerance: float, *, required: bool
) -> bool:
    """``min/max >= 1 - tolerance``. ``required`` controls missing-value handling.

    When ``required`` (e.g. price), a missing value on either side fails the
    match. When not required (e.g. area), a missing value is neutral and the
    check passes — letting other heuristics decide.
    """
    if a is None or b is None:
        return not required
    hi = max(a, b)
    if hi == 0:
        return a == b
    return min(a, b) / hi >= 1.0 - tolerance


def _coords_close_enough(
    a: dict[str, float] | None, b: dict[str, float] | None
) -> bool:
    """Haversine < 100m. Missing coords on either side is neutral (passes)."""
    if not a or not b:
        return True
    try:
        return _haversine_m(a, b) < _COORD_MAX_M
    except (KeyError, TypeError):
        return True


def _haversine_m(a: dict[str, float], b: dict[str, float]) -> float:
    lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
    lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _merge_group(group: list[Listing], news_results: NewsResults) -> Candidate:
    """Build one :class:`Candidate` from a cluster of matching listings."""
    representative = group[0]
    merged_listing = _safe_field_merge(representative, group)

    source_urls = [listing.url for listing in group]
    if len(group) > 1:
        sites = sorted({listing.source_site for listing in group})
        match_notes = f"Merged from {len(group)} sources: {', '.join(sites)}"
    else:
        match_notes = None

    return Candidate(
        listing=merged_listing,
        relevant_news=news_results,
        source_urls=source_urls,
        match_notes=match_notes,
    )


def _safe_field_merge(rep: Listing, group: list[Listing]) -> Listing | VerifiedListing:
    """Apply safer-for-the-user field choices on top of the representative.

    - ``price``: max across the group (worst-case for the user's budget).
    - ``bathrooms`` / ``parking_lots``: max — assume the lower portal just
      omitted the count.

    Other fields keep the representative's values. Returns the same Pydantic
    subclass as ``rep`` (``Listing`` or ``VerifiedListing``).
    """
    return rep.model_copy(update={
        "price": _max_or(rep.price, (l.price for l in group)),
        "bathrooms": _max_or(rep.bathrooms, (l.bathrooms for l in group)),
        "parking_lots": _max_or(rep.parking_lots, (l.parking_lots for l in group)),
    })


def _max_or(fallback, values):
    """``max`` of non-None values; ``fallback`` when none are present."""
    present = [v for v in values if v is not None]
    return max(present) if present else fallback
