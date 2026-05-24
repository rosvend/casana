"""Unit tests for src.utils.geography."""

from src.utils.geography import canonical_location, canonical_zone, normalize_geography


def test_known_municipality_returns_location_only():
    result = normalize_geography("Bogotá")
    assert result["location"] == "Bogotá"
    assert result["zone"] is None


def test_known_municipality_case_insensitive():
    result = normalize_geography("MEDELLÍN")
    assert result["location"] == "MEDELLÍN"
    assert result["zone"] is None


def test_known_municipality_without_accent():
    result = normalize_geography("medellin")
    assert result["location"] == "medellin"
    assert result["zone"] is None


def test_known_zone_maps_to_parent_city():
    result = normalize_geography("Chapinero")
    assert result["location"] == "bogota"
    assert result["zone"] == "Chapinero"


def test_known_zone_case_insensitive():
    result = normalize_geography("CHAPINERO")
    assert result["location"] == "bogota"
    assert result["zone"] == "CHAPINERO"


def test_known_zone_with_spaces():
    result = normalize_geography("El Poblado")
    assert result["location"] == "medellin"
    assert result["zone"] == "El Poblado"


def test_unknown_string_falls_back_to_location():
    result = normalize_geography("asdfqwerty")
    assert result["location"] == "asdfqwerty"
    assert result["zone"] is None


def test_empty_string_falls_through():
    result = normalize_geography("")
    assert result["location"] == ""
    assert result["zone"] is None


def test_canonical_location_strips_accents_and_case():
    assert canonical_location("BogotÁ ") == "bogota"
    assert canonical_location("Medellín") == "medellin"
    assert canonical_location("MEDELLÍN") == "medellin"


def test_canonical_location_matches_bogota_against_dane_dc_suffix(caplog):
    """DANE lists Bogotá as 'BOGOTÁ, D.C.' — user input 'bogota' must hit it."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.utils.geography"):
        result = canonical_location("bogota")
    assert result == "bogota"
    # If the DANE branch matched, no warning was logged. Regression-guard:
    assert not any(
        "not in DANE list" in rec.message for rec in caplog.records
    ), "canonical_location('bogota') fell through to the degraded path"


def test_canonical_location_accepts_explicit_dc_suffix():
    assert canonical_location("Bogotá D.C.") == "bogota"
    assert canonical_location("BOGOTÁ, D.C.") == "bogota"


def test_canonical_location_resolves_known_zone_to_parent_city():
    assert canonical_location("Chapinero") == "bogota"
    assert canonical_location("El Poblado") == "medellin"


def test_canonical_location_unknown_falls_through_with_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="src.utils.geography"):
        result = canonical_location("Atlantis")
    assert result == "atlantis"
    assert any("Atlantis" in record.message for record in caplog.records)


def test_canonical_location_empty_returns_none():
    assert canonical_location("") is None
    assert canonical_location(None) is None
    assert canonical_location("   ") is None


def test_canonical_zone_lowercases_and_strips_accents():
    assert canonical_zone("Chapinero") == "chapinero"
    assert canonical_zone("EL POBLADO") == "el poblado"
    assert canonical_zone("Las Brisas") == "las brisas"


def test_canonical_zone_empty_returns_none():
    assert canonical_zone("") is None
    assert canonical_zone(None) is None
