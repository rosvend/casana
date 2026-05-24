"""Standalone smoke test for ``properties_node`` (vertical slice).

Builds a fake ``PropertyFinderState`` with a mix of URL-pushable and
in-memory constraints, drives it through the node, and prints the resulting
``raw_listings``. This hits the live portals — it is slow (a stealthy
Cloudflare-solving fetch per property) and is meant to be run by hand:

    uv run python -m tests.test_properties_node

It verifies the node: extracts params from the constraint list, orchestrates
the discover/enrich tools, applies both filter stages, and returns the
``{"raw_listings": [...]}`` shape.
"""

from __future__ import annotations

import logging
import sys

from src.state import Constraint, PropertyFinderState, StructuredRequirements
from src.agents.properties_agent import _extract_params, properties_node


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    # URL constraints: location + max price 4M. In-memory constraint: 2+ baths.
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location", exact_value="medellin",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="property_type", exact_value="apartment",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="transaction_type", exact_value="rent",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="price", max_value=4_000_000,
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="bathrooms", min_value=2,
                constraint_type="hard", importance="important",
            ),
        ],
        summary="2+ bathroom apartment for rent in Medellín under 4M COP.",
    )

    state: PropertyFinderState = {"requirements": requirements}

    print("=== properties_node ===", file=sys.stderr)
    result = properties_node(state)

    listings = result.get("raw_listings", [])
    print(f"\nreturn keys: {list(result.keys())}")
    print(f"raw_listings: {len(listings)} listing(s) survived both filter stages")
    by_source: dict[str, int] = {}
    for lst in listings:
        by_source[lst.source_site] = by_source.get(lst.source_site, 0) + 1
    print(f"by source: {by_source or '{}'}  "
          "(both portals should appear — they are queried concurrently)\n")
    for lst in listings:
        print(
            f"  - {lst.id}\n"
            f"    price={lst.price}  bathrooms={lst.bathrooms}  "
            f"bedrooms={lst.bedrooms}  area_m2={lst.area_m2}\n"
            f"    {lst.url}"
        )

    if not listings:
        print(
            "\n(no listings survived — could be portal DOM drift or a transient "
            "fetch failure; the return shape is still correct. Re-run to retry.)",
            file=sys.stderr,
        )
    return 0


def test_extract_params_canonicalizes_location() -> None:
    """Accents and case in user input must not leak into URL params."""
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location",
                exact_value="BogotÁ ",
                constraint_type="hard",
                importance="critical",
            ),
        ],
        summary="",
    )
    url_params, _ = _extract_params(requirements)
    assert url_params["location"] == "bogota"


def test_extract_params_resolves_known_zone_to_parent_city() -> None:
    """A 'location' string that's actually a neighborhood resolves to the city."""
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location",
                exact_value="Chapinero",
                constraint_type="hard",
                importance="critical",
            ),
        ],
        summary="",
    )
    url_params, _ = _extract_params(requirements)
    assert url_params["location"] == "bogota"


def test_extract_params_skips_soft_zone() -> None:
    """A soft zone constraint must not narrow the search URL."""
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location",
                exact_value="bogota",
                constraint_type="hard",
                importance="critical",
            ),
            Constraint(
                field="zone",
                exact_value="chapinero",
                constraint_type="soft",
                importance="nice_to_have",
            ),
        ],
        summary="",
    )
    url_params, _ = _extract_params(requirements)
    assert "zone" not in url_params, url_params


def test_extract_params_keeps_hard_zone() -> None:
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="zone",
                exact_value="Chapinero",
                constraint_type="hard",
                importance="critical",
            ),
        ],
        summary="",
    )
    url_params, _ = _extract_params(requirements)
    assert url_params["zone"] == "chapinero"


if __name__ == "__main__":
    raise SystemExit(main())
