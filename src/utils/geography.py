"""Classify an extracted place string as a Colombian municipality or a sub-municipal zone.

The DANE municipality list (1,122 entries, shipped in
``data/Departamentos_y_municipios_codigos_20260523.csv``) is the ground truth
for the ``location`` constraint. Anything not in that list but matched by
:data:`KNOWN_ZONES` is split into ``(parent_city, zone)``.

TODO(geography): replace :data:`KNOWN_ZONES` with a real datasource. The
hardcoded dict is brittle, biased toward Bogotá + Medellín, and rots whenever
the product touches a new neighborhood. Failure modes seen in production:
    - LLM emits joined strings like ``"chapinero_bogota"`` that don't match
      any single entry — even though both halves are known. Pre-splitting on
      ``_`` / ``,`` before canonicalization would catch this without new data.
    - Common variants ("chapinero alto", "chapinero central", "el chico")
      collapse to one canonical zone with no disambiguation.
    - Smaller cities (Cali, Barranquilla, Bucaramanga) have ~0 coverage.

Candidate replacements, in order of expected ROI:
    1. Per-city neighborhood CSVs in ``data/neighborhoods/<city>.csv`` sourced
       from each city's open-data portal (Bogotá: Catastro Distrital UPZ
       /barrios; Medellín: Alcaldía comunas y barrios; etc.). Same pattern as
       DANE — deterministic, offline, testable. Start with the top 5 cities.
    2. OSM Nominatim (self-hosted or rate-limited public instance) as a
       fallback for misses, with local caching since neighborhood names
       don't change often. Avoid Google Maps (cost + ToS).
    3. RAG / vector search is overkill here — a neighborhood name is a
       discrete identifier, not a fuzzy concept. Skip unless we hit fuzzy
       *meaning* problems (e.g. "barrio chévere cerca del estadio").
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "Departamentos_y_municipios_codigos_20260523.csv"
)

KNOWN_ZONES: dict[str, str] = {
    "chapinero": "bogota",
    "usaquen": "bogota",
    "suba": "bogota",
    "fontibon": "bogota",
    "kennedy": "bogota",
    "el poblado": "medellin",
    "laureles": "medellin",
    "belen": "medellin",
    "las brisas": "bogota",
    "alameda 170": "bogota",
    "montevideo": "bogota",
    "nueva castilla": "bogota",
    "guaymaral": "bogota",
}


class GeoResult(TypedDict):
    location: str
    zone: str | None


def _normalize(text: str) -> str:
    """Lowercase, strip accents, and collapse compound separators to spaces.

    The LLM occasionally emits a compound location like ``"chapinero_bogota"``
    or ``"chapinero,bogota"`` instead of two separate constraints. Mapping
    ``[_/,]`` to space lets the existing token-boundary matcher in
    :func:`_find_known_zone_substring` recover the (zone, parent_city) split
    without a separate code path. The token-boundary regex uses ``\\w`` which
    treats ``_`` as a word character, so leaving ``_`` in place would silently
    block the match.
    """
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    cleaned = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[_/,]", " ", cleaned)


# Strips a trailing administrative-division suffix like ", D.C." from a
# normalized name. Bogotá ships in the DANE list as "BOGOTÁ, D.C." while
# user input is invariably bare ("bogota"); this lets both forms match.
_ADMIN_SUFFIX_RE = re.compile(r",?\s*d\.?\s*c\.?\s*$")


@lru_cache(maxsize=1)
def _municipality_set() -> frozenset[str]:
    """Load the DANE list once; cache the normalized name set.

    Each DANE row is added in two forms: the literal normalized name and a
    version with the ``, D.C.`` administrative suffix stripped — so user
    input ``"bogota"`` matches the canonical entry ``"BOGOTÁ, D.C."``.
    """
    names: set[str] = set()
    with _CSV_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("Nombre Municipio") or ""
            if name:
                norm = _normalize(name)
                names.add(norm)
                stripped = _ADMIN_SUFFIX_RE.sub("", norm).strip()
                if stripped and stripped != norm:
                    names.add(stripped)
    return frozenset(names)


def _find_known_zone_substring(norm: str) -> tuple[str, str] | None:
    """Token-boundary substring match for compound inputs like "laureles medellin".

    Returns ``(zone, parent_city)`` for the longest KNOWN_ZONES key that appears
    as a whole-token substring of ``norm``, or ``None``. Longest-first so
    multi-word zones like "el poblado" win over "el".
    """
    for zone in sorted(KNOWN_ZONES, key=len, reverse=True):
        if re.search(rf"(?<![\w]){re.escape(zone)}(?![\w])", norm):
            return zone, KNOWN_ZONES[zone]
    return None


def normalize_geography(extracted_loc: str) -> GeoResult:
    """Return ``{"location": ..., "zone": ...}`` for an extracted place string.

    - Valid municipality           → ``{"location": input, "zone": None}``
    - Known sub-municipal zone     → ``{"location": parent_city, "zone": input}``
    - Compound "zone city" string  → ``{"location": parent_city, "zone": zone}``
    - Anything else                → ``{"location": input, "zone": None}``

    The input casing is preserved on the way out so downstream lowercasing
    behaves consistently with the existing ``_extract_params`` logic.
    """
    if not isinstance(extracted_loc, str) or not extracted_loc.strip():
        return {"location": extracted_loc, "zone": None}

    norm = _normalize(extracted_loc)

    if norm in _municipality_set():
        return {"location": extracted_loc, "zone": None}

    if norm in KNOWN_ZONES:
        return {"location": KNOWN_ZONES[norm], "zone": extracted_loc}

    zone_hit = _find_known_zone_substring(norm)
    if zone_hit is not None:
        zone, city = zone_hit
        return {"location": city, "zone": zone}

    return {"location": extracted_loc, "zone": None}


def canonical_location(s: str | None) -> str | None:
    """Return the canonical (lowercase, accent-stripped) form of a city name.

    Resolves the input against the DANE municipality list and KNOWN_ZONES so
    every consumer (requirements, scraper URL builder, evaluator) compares
    the same string. Returns ``None`` for empty input AND for any string we
    can't resolve — callers MUST treat ``None`` as a hard signal that the
    input isn't a known place and decide whether to ask the user for
    clarification or omit the location from a URL. Previously this function
    returned the raw normalized input on a miss, which let unknown cities
    flow downstream and produce wildly off-location search results.

    Inputs carrying an administrative suffix like ``"Bogotá D.C."`` are
    canonicalized to the bare-name form (``"bogota"``) to mirror how the
    DANE-derived set is enriched in :func:`_municipality_set`.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    municipalities = _municipality_set()
    norm = _normalize(s)
    stripped = _ADMIN_SUFFIX_RE.sub("", norm).strip()
    if stripped and stripped in municipalities:
        return stripped
    if norm in municipalities:
        return norm
    if norm in KNOWN_ZONES:
        return KNOWN_ZONES[norm]
    zone_hit = _find_known_zone_substring(norm)
    if zone_hit is not None:
        _, city = zone_hit
        return city
    logger.warning("canonical_location: %r not in DANE list or KNOWN_ZONES", s)
    return None


def canonical_zone(s: str | None) -> str | None:
    """Return the canonical (lowercase, accent-stripped) form of a zone string.

    Zones aren't in the DANE list, so this is just :func:`_normalize`. Returns
    ``None`` for empty input.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    return _normalize(s)
