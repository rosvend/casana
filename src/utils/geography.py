"""Classify an extracted place string as a Colombian municipality or a sub-municipal zone.

The DANE municipality list (1,122 entries, shipped in
``data/Departamentos_y_municipios_codigos_20260523.csv``) is the ground truth
for the ``location`` constraint. Anything not in that list but matched by
:data:`KNOWN_ZONES` is split into ``(parent_city, zone)``.

TODO: replace KNOWN_ZONES with a real datasource — e.g. a curated CSV of
neighborhoods keyed by municipality code, or a small admin API. The hardcoded
dict is an MVP shortcut and rots whenever the product covers a new neighborhood.
"""

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

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
    """Lowercase, strip whitespace and combining accents — used only for matching."""
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _municipality_set() -> frozenset[str]:
    """Load the DANE list once; cache the normalized name set."""
    names: set[str] = set()
    with _CSV_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("Nombre Municipio") or ""
            if name:
                names.add(_normalize(name))
    return frozenset(names)


def normalize_geography(extracted_loc: str) -> GeoResult:
    """Return ``{"location": ..., "zone": ...}`` for an extracted place string.

    - Valid municipality           → ``{"location": input, "zone": None}``
    - Known sub-municipal zone     → ``{"location": parent_city, "zone": input}``
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

    return {"location": extracted_loc, "zone": None}
