"""Schema-level assertions on data/normalized_zones.json.

Guards the runtime contract between the ETL (scripts/normalize_geodata.py)
and the loader in src/utils/geography.py.
"""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "normalized_zones.json"
EXPECTED_CITIES = {"bogota", "medellin", "cali", "barranquilla", "cartagena"}


def _load():
    return json.loads(DATA.read_text(encoding="utf-8"))


def test_normalized_zones_file_exists():
    assert DATA.exists(), f"missing: {DATA}. Run scripts/normalize_geodata.py."


def test_top_level_schema():
    d = _load()
    assert d["version"] == 1
    assert set(d["cities"]) == EXPECTED_CITIES
    assert isinstance(d["neighborhoods"], list) and len(d["neighborhoods"]) > 1500


def test_every_record_has_required_fields():
    for r in _load()["neighborhoods"]:
        assert r["city"] in EXPECTED_CITIES
        assert r["neighborhood"], r
        assert "upper_division" in r  # may be None for some
        c = r["centroid"]
        assert -82 <= c["lon"] <= -66, r
        assert -5 <= c["lat"] <= 14, r


def test_known_anchor_places_present():
    records = _load()["neighborhoods"]
    by_neighborhood = {(r["city"], r["neighborhood"]) for r in records}
    by_upper_div = {(r["city"], r.get("upper_division")) for r in records if r.get("upper_division")}
    # Chapinero is a localidad in Bogotá (no barrio literally named "chapinero").
    assert ("bogota", "chapinero") in by_upper_div
    # El Poblado and Laureles are both barrio names in Medellín.
    assert ("medellin", "el poblado") in by_neighborhood
    assert ("medellin", "laureles") in by_neighborhood
    # Bocagrande exists as a barrio in Cartagena.
    assert ("cartagena", "bocagrande") in by_neighborhood
    # Cali comuna labels are stored as "comuna N".
    assert ("cali", "comuna 14") in by_upper_div


def test_every_city_has_neighborhoods():
    records = _load()["neighborhoods"]
    counts = {c: 0 for c in EXPECTED_CITIES}
    for r in records:
        counts[r["city"]] += 1
    for city, n in counts.items():
        assert n > 100, f"{city} has only {n} records — ETL likely lost a source"
