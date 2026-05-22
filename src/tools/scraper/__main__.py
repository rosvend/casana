"""Live demo entry point — exercises the full discover → enrich pipeline.

Run against live portals with:

    uv run python -m src.tools.scraper
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Bootstrap the project root onto sys.path so `python -m src.tools.scraper`
# resolves the `src` package even when invoked from outside the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tools.scraper import extract_property_details, search_listings  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    print("=== Tool 1: search_listings (filtered) ===", file=sys.stderr)
    hits: list[dict] = search_listings.invoke({
        "location": "medellin",
        "property_type": "apartamentos",
        "transaction": "arriendo",
        "max_price": 2_500_000,
        "bedrooms": 2,
    })
    print(json.dumps(hits[:5], indent=2, ensure_ascii=False))

    if not hits:
        print("no listings discovered — aborting enrichment demo", file=sys.stderr)
        sys.exit(1)

    target = hits[0]
    print(f"\n=== Tool 2: extract_property_details({target['url']!r}) ===", file=sys.stderr)
    listing = extract_property_details.invoke({"url": target["url"]})
    if listing is None:
        print("deep scrape failed", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(json.loads(listing.model_dump_json()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
