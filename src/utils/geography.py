"""Geography validator backed by data/normalized_zones.json.

The ETL in ``scripts/normalize_geodata.py`` produces ``normalized_zones.json``
from per-city geospatial inputs (~2,300 records, top 5 Colombian cities). This
module loads that JSON once at import-time into in-memory dicts and exposes:

- :func:`resolve_place` (``str -> PlaceMatch | None``) — rich match returning
  ``(city, upper_division, neighborhood, centroid)``. **Use this for new logic.**
- :func:`canonical_location` (``str -> str | None``) — legacy parent-city
  resolver kept for URL builders (fincaraiz, metrocuadrado) and the evaluator.
- :func:`canonical_zone` (``str -> str | None``) — normalized zone string.
- :func:`normalize_geography` (``str -> {location, zone}``) — legacy
  compound-string splitter; consider :func:`resolve_place` instead.

The DANE municipalities CSV remains the ground truth for city-level matches
(all 1,122 municipios). The neighborhood index covers the top 5 cities only;
other cities still match at the city level via DANE.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CSV_PATH = _DATA_DIR / "Departamentos_y_municipios_codigos_20260523.csv"
_ZONES_PATH = _DATA_DIR / "normalized_zones.json"

# Strips a trailing administrative suffix like ", D.C." from a normalized name.
# DANE lists Bogotá as "BOGOTÁ, D.C." while users invariably type bare "bogota".
_ADMIN_SUFFIX_RE = re.compile(r",?\s*d\.?\s*c\.?\s*$")

_TYPO_CUTOFF = 0.85  # difflib similarity threshold; ~1-char edits at len 8

# When a name appears in multiple cities (e.g. "chapinero" is a Bogotá localidad
# AND a Cali barrio), this order picks the canonical match — roughly population /
# real-estate market size. Users typing "chapinero" almost always mean Bogotá.
_CITY_PRIORITY = ("bogota", "medellin", "cali", "barranquilla", "cartagena")
_CITY_RANK = {c: i for i, c in enumerate(_CITY_PRIORITY)}


def _by_priority(matches: list["PlaceMatch"]) -> list["PlaceMatch"]:
    return sorted(matches, key=lambda m: _CITY_RANK.get(m.city, 99))


@dataclass(frozen=True)
class PlaceMatch:
    city: str                      # always set, normalized
    upper_division: str | None     # localidad/comuna, normalized
    neighborhood: str | None       # None means this is a zone-level or city-level match
    centroid_lat: float | None
    centroid_lon: float | None


class GeoResult(TypedDict):
    location: str
    zone: str | None


def _normalize(text: str) -> str:
    """Lowercase, strip accents, and collapse compound separators to spaces."""
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    cleaned = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[_/,]", " ", cleaned).strip()


@lru_cache(maxsize=1)
def _municipality_set() -> frozenset[str]:
    """Load the DANE list once; cache the normalized name set.

    Each row is added in two forms — the literal normalized name and a version
    with the ``, D.C.`` suffix stripped — so user input ``"bogota"`` matches
    the canonical entry ``"BOGOTÁ, D.C."``.
    """
    names: set[str] = set()
    with _CSV_PATH.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Nombre Municipio") or ""
            if not name:
                continue
            norm = _normalize(name)
            names.add(norm)
            stripped = _ADMIN_SUFFIX_RE.sub("", norm).strip()
            if stripped and stripped != norm:
                names.add(stripped)
    return frozenset(names)


@lru_cache(maxsize=1)
def _zones_payload() -> dict:
    return json.loads(_ZONES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _neighborhood_index() -> dict[str, list[PlaceMatch]]:
    """key = normalized neighborhood name → list of PlaceMatch (city-priority sorted)."""
    idx: dict[str, list[PlaceMatch]] = {}
    for r in _zones_payload()["neighborhoods"]:
        m = PlaceMatch(
            city=r["city"],
            upper_division=r.get("upper_division"),
            neighborhood=r["neighborhood"],
            centroid_lat=r["centroid"]["lat"],
            centroid_lon=r["centroid"]["lon"],
        )
        idx.setdefault(r["neighborhood"], []).append(m)
    return {k: _by_priority(v) for k, v in idx.items()}


@lru_cache(maxsize=1)
def _upper_division_index() -> dict[str, list[PlaceMatch]]:
    """key = normalized localidad/comuna name → list of PlaceMatch (city-priority sorted)."""
    idx: dict[str, list[PlaceMatch]] = {}
    for r in _zones_payload()["neighborhoods"]:
        up = r.get("upper_division")
        if not up:
            continue
        m = PlaceMatch(
            city=r["city"],
            upper_division=up,
            neighborhood=None,
            centroid_lat=r["centroid"]["lat"],
            centroid_lon=r["centroid"]["lon"],
        )
        idx.setdefault(up, []).append(m)
    return {k: _by_priority(v) for k, v in idx.items()}


def _find_city_token(tokens: list[str], munis: frozenset[str]) -> tuple[int, str] | None:
    """Locate a token that is a known municipality. Returns (index, canonical_city)."""
    for i, t in enumerate(tokens):
        if t in munis:
            return i, t
        stripped = _ADMIN_SUFFIX_RE.sub("", t).strip()
        if stripped and stripped in munis:
            return i, stripped
    return None


def resolve_place(s: str | None) -> PlaceMatch | None:
    """Resolve a free-form place string to a :class:`PlaceMatch`.

    Tries, in order:
    1. exact neighborhood match,
    2. exact upper_division (localidad/comuna) match,
    3. exact city match (DANE),
    4. compound split + retry per token,
    5. typo-tolerant match via ``difflib`` (neighborhood → upper_division → city).

    Returns ``None`` when no candidate clears the bar — callers MUST treat
    ``None`` as a hard signal to ask the user for clarification.
    """
    if not isinstance(s, str) or not s.strip():
        return None

    norm = _normalize(s)
    nbh = _neighborhood_index()
    ups = _upper_division_index()
    munis = _municipality_set()

    # 1. Exact match in either index. When the name appears in both indexes
    #    (e.g. "chapinero" → Bogotá localidad + Cali barrio), the
    #    higher-ranked city wins. Upper_division beats neighborhood within
    #    the same city, since the larger administrative unit is what users
    #    typically mean when they say a zone name without further qualifiers.
    nbh_hit = nbh.get(norm, [None])[0]
    ups_hit = ups.get(norm, [None])[0]
    if nbh_hit and ups_hit:
        if _CITY_RANK.get(ups_hit.city, 99) <= _CITY_RANK.get(nbh_hit.city, 99):
            return PlaceMatch(
                city=ups_hit.city,
                upper_division=norm,
                neighborhood=norm if ups_hit.city == nbh_hit.city else None,
                centroid_lat=ups_hit.centroid_lat,
                centroid_lon=ups_hit.centroid_lon,
            )
        return nbh_hit
    if ups_hit:
        return PlaceMatch(
            city=ups_hit.city,
            upper_division=norm,
            neighborhood=None,
            centroid_lat=ups_hit.centroid_lat,
            centroid_lon=ups_hit.centroid_lon,
        )
    if nbh_hit:
        return nbh_hit

    # 3. Exact city (DANE). Prefer the stripped form so "Bogotá D.C." resolves
    #    to canonical "bogota" instead of the noisier "bogota  d.c.".
    stripped = _ADMIN_SUFFIX_RE.sub("", norm).strip()
    for candidate in (stripped, norm):
        if candidate and candidate in munis:
            return PlaceMatch(
                city=candidate, upper_division=None, neighborhood=None,
                centroid_lat=None, centroid_lon=None,
            )

    # 4. Compound split. If one token names a city, take the rest as a
    #    *phrase* (preserving multi-word zone names like "el poblado") and
    #    look it up restricted to that city. This is how "chapinero bogota"
    #    picks Bogotá's Chapinero over Cali's barrio.
    tokens = [t for t in norm.split() if t]
    city_hit = _find_city_token(tokens, munis) if len(tokens) > 1 else None
    if city_hit is not None:
        city_idx, city = city_hit
        phrase = " ".join(t for i, t in enumerate(tokens) if i != city_idx).strip()
        if phrase:
            for cand in nbh.get(phrase, []):
                if cand.city == city:
                    return cand
            for cand in ups.get(phrase, []):
                if cand.city == city:
                    return PlaceMatch(
                        city=cand.city, upper_division=phrase, neighborhood=None,
                        centroid_lat=cand.centroid_lat, centroid_lon=cand.centroid_lon,
                    )
        # Degrade to city-only when the zone phrase didn't resolve.
        return PlaceMatch(
            city=city, upper_division=None, neighborhood=None,
            centroid_lat=None, centroid_lon=None,
        )

    # No city in the compound — fall back to per-token best-effort lookup.
    for token in tokens:
        if token in nbh:
            return nbh[token][0]
        if token in ups:
            m = ups[token][0]
            return PlaceMatch(
                city=m.city, upper_division=token, neighborhood=None,
                centroid_lat=m.centroid_lat, centroid_lon=m.centroid_lon,
            )

    # 5. Typo tolerance — neighborhoods, then upper_divisions, then cities.
    near = get_close_matches(norm, list(nbh.keys()), n=1, cutoff=_TYPO_CUTOFF)
    if near:
        logger.info("resolve_place: typo correction %r → %r (barrio)", s, near[0])
        return nbh[near[0]][0]
    near = get_close_matches(norm, list(ups.keys()), n=1, cutoff=_TYPO_CUTOFF)
    if near:
        logger.info("resolve_place: typo correction %r → %r (zona)", s, near[0])
        m = ups[near[0]][0]
        return PlaceMatch(
            city=m.city, upper_division=near[0], neighborhood=None,
            centroid_lat=m.centroid_lat, centroid_lon=m.centroid_lon,
        )
    near = get_close_matches(norm, list(munis), n=1, cutoff=_TYPO_CUTOFF)
    if near:
        logger.info("resolve_place: typo correction %r → %r (city)", s, near[0])
        return PlaceMatch(
            city=near[0], upper_division=None, neighborhood=None,
            centroid_lat=None, centroid_lon=None,
        )

    logger.warning("resolve_place: %r not found in normalized geo data", s)
    return None


def canonical_location(s: str | None) -> str | None:
    """Parent-city resolver kept for URL builders and the evaluator.

    Routes through :func:`resolve_place` and returns the canonical city slug,
    or ``None`` if nothing matched. Callers MUST treat ``None`` as a hard
    signal that the input isn't a known place.
    """
    m = resolve_place(s)
    return m.city if m else None


def canonical_zone(s: str | None) -> str | None:
    """Return the canonical (lowercase, accent-stripped) form of a zone string.

    Zones aren't validated against the index here — :func:`resolve_place` is
    the validator. This is just normalization for downstream URL builders and
    evaluator equality checks.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    return _normalize(s)


def normalize_geography(extracted_loc: str) -> GeoResult:
    """Legacy compound-string splitter — kept for backward compatibility.

    Prefer :func:`resolve_place` in new code. Returns the ``{location, zone}``
    dict shape that requirements_agent's older paths read.

    Behavior preserved from the pre-data-driven implementation:
    - Direct city match → ``{location: extracted_loc, zone: None}``.
    - Direct zone match (normalized input equals the zone key) → ``zone``
      keeps the caller's original casing.
    - Compound match (zone embedded in a larger string) → ``zone`` is the
      canonical normalized zone name from the index.
    - Anything else → ``{location: extracted_loc, zone: None}``.
    """
    if not isinstance(extracted_loc, str) or not extracted_loc.strip():
        return {"location": extracted_loc, "zone": None}

    norm = _normalize(extracted_loc)
    stripped = _ADMIN_SUFFIX_RE.sub("", norm).strip()
    munis = _municipality_set()
    if norm in munis or stripped in munis:
        return {"location": extracted_loc, "zone": None}

    m = resolve_place(extracted_loc)
    if m is None or (m.neighborhood is None and m.upper_division is None):
        return {"location": extracted_loc, "zone": None}

    zone_key = m.neighborhood or m.upper_division
    if zone_key and norm == zone_key:
        # Direct zone hit — preserve the user's casing.
        return {"location": m.city, "zone": extracted_loc}
    # Compound or upper_division hit — return the canonical zone identifier.
    return {"location": m.city, "zone": zone_key}
