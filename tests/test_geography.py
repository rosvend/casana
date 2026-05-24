"""Unit tests for src.utils.geography.normalize_geography."""

from src.utils.geography import normalize_geography


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
