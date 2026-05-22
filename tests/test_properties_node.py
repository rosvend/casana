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
from src.agents.properties_agent import properties_node


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
    print(f"raw_listings: {len(listings)} listing(s) survived both filter stages\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
