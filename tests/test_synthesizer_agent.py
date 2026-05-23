"""Standalone smoke test for ``synthesizer_node`` (vertical slice).

``synthesizer_node`` is pure Python — no network, no LLM — so this script
exercises three scenarios with hand-built ``Listing`` / ``NewsItem`` objects:

- ``test_fuzzy_merge_across_portals`` — two compatible listings from different
  portals (Finca Raíz + Metro Cuadrado) plus an unrelated third must collapse
  to exactly two candidates; the merged one must keep the higher price and
  carry both source URLs.
- ``test_coordinate_handling`` — coordinates ~10m apart still merge; the same
  pair ~4.6 km apart must stay separate even when every other heuristic agrees.
- ``test_intra_portal_dedup`` — two listings sharing the same ``id`` collapse
  to one (last-write-wins by ``id``).

    uv run python -m tests.test_synthesizer_agent
"""

from __future__ import annotations

from src.agents.synthesizer_agent import synthesizer_node
from src.state import Candidate, Listing, NewsItem, PropertyFinderState


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _listing(
    *,
    id: str,
    source_site: str,
    url: str,
    bedrooms: int,
    area_m2: float,
    price: float,
    zone: str = "El Poblado",
    property_type: str = "apartment",
    transaction_type: str = "rent",
    bathrooms: int | None = None,
    parking_lots: int | None = None,
    coordinates: dict[str, float] | None = None,
) -> Listing:
    """Tight factory so each test case fits on one screen."""
    return Listing(
        id=id,
        source_site=source_site,
        url=url,
        price=price,
        area_m2=area_m2,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        zone=zone,
        property_type=property_type,
        transaction_type=transaction_type,
        parking_lots=parking_lots,
        coordinates=coordinates,
    )


def _news_results() -> dict:
    """Single dummy NewsItem so we can assert it propagates to every candidate."""
    item = NewsItem(
        title="Nuevo parque en El Poblado",
        summary="La alcaldía inaugura un parque público en la zona.",
        url="https://example.com/news/1",
        source="example.com",
        zone="El Poblado",
    )
    return {
        "crime_safety": [],
        "transportation": [],
        "infrastructure": [item],
        "events": [],
        "market_trends": [],
    }


def test_fuzzy_merge_across_portals() -> bool:
    """A + B merge; C stays separate; merged keeps the higher price + both URLs."""
    a = _listing(
        id="fincaraiz:A1", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/A1",
        bedrooms=2, area_m2=100, price=3_000_000,
    )
    b = _listing(
        id="metrocuadrado:B1", source_site="metrocuadrado",
        url="https://metrocuadrado.com/inmueble/B1",
        bedrooms=2, area_m2=95, price=3_100_000,
    )
    c = _listing(
        id="fincaraiz:C1", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/C1",
        bedrooms=3, area_m2=150, price=5_000_000,
    )

    news = _news_results()
    state: PropertyFinderState = {"raw_listings": [a, b, c], "news_results": news}
    result = synthesizer_node(state)

    candidates = result.get("candidates")
    passed = _check("result['candidates'] is a list", isinstance(candidates, list))
    if not isinstance(candidates, list):
        return passed

    passed &= _check(
        "exactly 2 candidates produced",
        len(candidates) == 2,
        f"got {len(candidates)}",
    )
    passed &= _check(
        "every candidate is a Candidate",
        all(isinstance(c, Candidate) for c in candidates),
    )

    merged = next(
        (cand for cand in candidates if len(cand.source_urls) > 1), None
    )
    singleton = next(
        (cand for cand in candidates if len(cand.source_urls) == 1), None
    )
    passed &= _check("one merged candidate (2 source_urls)", merged is not None)
    passed &= _check("one singleton candidate (1 source_url)", singleton is not None)

    if merged is not None:
        passed &= _check(
            "merged candidate kept the higher price (3_100_000)",
            merged.listing.price == 3_100_000,
            f"got {merged.listing.price}",
        )
        passed &= _check(
            "merged candidate carries both A and B URLs",
            set(merged.source_urls) == {a.url, b.url},
            f"got {merged.source_urls}",
        )
        passed &= _check(
            "merged candidate has match_notes set",
            bool(merged.match_notes),
            f"got {merged.match_notes!r}",
        )

    if singleton is not None:
        passed &= _check(
            "singleton candidate is C (3 bedrooms)",
            singleton.listing.bedrooms == 3,
            f"got bedrooms={singleton.listing.bedrooms}",
        )

    passed &= _check(
        "news_results propagated to every candidate",
        all(cand.relevant_news == news for cand in candidates),
    )
    return passed


def test_coordinate_handling() -> bool:
    """Close coords (~10m) merge; far coords (~4.6 km) block the merge."""
    # Sub-case 2a — close coords: should still merge.
    close_a = _listing(
        id="fincaraiz:A1", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/A1",
        bedrooms=2, area_m2=100, price=3_000_000,
        coordinates={"lat": 6.2086, "lon": -75.5695},
    )
    close_b = _listing(
        id="metrocuadrado:B1", source_site="metrocuadrado",
        url="https://metrocuadrado.com/inmueble/B1",
        bedrooms=2, area_m2=95, price=3_100_000,
        coordinates={"lat": 6.20861, "lon": -75.5695},
    )
    state_close: PropertyFinderState = {
        "raw_listings": [close_a, close_b],
        "news_results": _news_results(),
    }
    result_close = synthesizer_node(state_close)
    candidates_close = result_close.get("candidates") or []

    passed = _check(
        "close coords (~1m apart): 1 candidate",
        len(candidates_close) == 1,
        f"got {len(candidates_close)}",
    )
    if candidates_close:
        passed &= _check(
            "close coords: candidate carries both URLs",
            set(candidates_close[0].source_urls) == {close_a.url, close_b.url},
        )

    # Sub-case 2b — far coords: should NOT merge.
    far_a = _listing(
        id="fincaraiz:A1", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/A1",
        bedrooms=2, area_m2=100, price=3_000_000,
        coordinates={"lat": 6.2086, "lon": -75.5695},
    )
    far_b = _listing(
        id="metrocuadrado:B1", source_site="metrocuadrado",
        url="https://metrocuadrado.com/inmueble/B1",
        bedrooms=2, area_m2=95, price=3_100_000,
        coordinates={"lat": 6.2500, "lon": -75.5695},
    )
    state_far: PropertyFinderState = {
        "raw_listings": [far_a, far_b],
        "news_results": _news_results(),
    }
    result_far = synthesizer_node(state_far)
    candidates_far = result_far.get("candidates") or []

    passed &= _check(
        "far coords (~4.6 km apart): 2 candidates",
        len(candidates_far) == 2,
        f"got {len(candidates_far)}",
    )
    if len(candidates_far) == 2:
        passed &= _check(
            "far coords: each candidate has exactly one source URL",
            all(len(cand.source_urls) == 1 for cand in candidates_far),
        )

    return passed


def test_intra_portal_dedup() -> bool:
    """Two listings sharing ``id`` collapse to one candidate (last write wins)."""
    a1 = _listing(
        id="fincaraiz:DUP", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/DUP-v1",
        bedrooms=2, area_m2=100, price=3_000_000,
    )
    a2 = _listing(
        id="fincaraiz:DUP", source_site="fincaraiz",
        url="https://fincaraiz.com.co/inmueble/DUP-v2",
        bedrooms=2, area_m2=100, price=3_050_000,
    )
    state: PropertyFinderState = {
        "raw_listings": [a1, a2],
        "news_results": _news_results(),
    }
    result = synthesizer_node(state)
    candidates = result.get("candidates") or []

    passed = _check(
        "duplicate ids: exactly 1 candidate",
        len(candidates) == 1,
        f"got {len(candidates)}",
    )
    if candidates:
        passed &= _check(
            "duplicate ids: candidate's url is the last one written",
            candidates[0].source_urls == [a2.url],
            f"got {candidates[0].source_urls}",
        )
    return passed


def main() -> int:
    print("=== synthesizer_node ===")
    ok = True
    print("\ntest_fuzzy_merge_across_portals:")
    ok &= test_fuzzy_merge_across_portals()
    print("\ntest_coordinate_handling:")
    ok &= test_coordinate_handling()
    print("\ntest_intra_portal_dedup:")
    ok &= test_intra_portal_dedup()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
