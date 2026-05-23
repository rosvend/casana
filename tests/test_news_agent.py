"""Standalone smoke test for ``news_node`` (vertical slice).

Drives the area-news node two ways:

- ``test_no_location_returns_empty`` — offline, no network or LLM. Proves the
  no-location early-return yields a well-formed, all-empty ``news_results``
  (the anti-hallucination contract) for a near-zero cost.
- ``test_news_node_live`` — end-to-end against live DuckDuckGo + the OpenAI
  API. Needs a valid ``OPENAI_API_KEY`` in ``.env`` and network access.

    uv run python -m tests.test_news_agent

It verifies the node: resolves location/zone from the constraints list, runs
the per-category searches, and returns a ``NewsResults`` dict whose five
``NewsCategory`` keys each hold a list of ``NewsItem`` objects.
"""

from __future__ import annotations

from src.agents.news_agent import news_node
from src.state import Constraint, NewsItem, PropertyFinderState, StructuredRequirements

#: The closed NewsCategory set the node must always key its result by.
_CATEGORIES = {
    "crime_safety", "transportation", "infrastructure", "events", "market_trends",
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _check_shape(news: object) -> bool:
    """Assertions shared by both tests: ``news`` is a 5-category dict of lists."""
    if not isinstance(news, dict):
        return _check("news_results is a dict", False, f"got {type(news).__name__}")

    passed = _check("news_results is a dict", True)
    passed &= _check(
        "keys are exactly the 5 NewsCategory values",
        set(news.keys()) == _CATEGORIES,
        f"got {sorted(news.keys())}",
    )
    for category, items in news.items():
        passed &= _check(
            f"news_results[{category!r}] is a list",
            isinstance(items, list),
            f"got {type(items).__name__}",
        )
        if isinstance(items, list):
            bad = [type(i).__name__ for i in items if not isinstance(i, NewsItem)]
            passed &= _check(
                f"every item in {category!r} is a NewsItem",
                not bad,
                f"non-NewsItem entries: {bad}" if bad else "",
            )
    return passed


def test_no_location_returns_empty() -> bool:
    """No location constraint -> a well-formed, all-empty news_results."""
    state: PropertyFinderState = {"requirements": None}
    result = news_node(state)

    news = result.get("news_results")
    passed = _check_shape(news)
    if isinstance(news, dict):
        passed &= _check(
            "every category list is empty",
            all(items == [] for items in news.values()),
            f"non-empty: {[k for k, v in news.items() if v]}",
        )
    return passed


def test_news_node_live() -> bool:
    """A Medellín/Guayabal brief must return categorized NewsItem objects."""
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location", exact_value="Medellin",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="zone", exact_value="Guayabal",
                constraint_type="soft", importance="important",
            ),
        ],
        summary="Apartamento en Guayabal, Medellín.",
    )
    state: PropertyFinderState = {"requirements": requirements}
    result = news_node(state)

    news = result.get("news_results")
    passed = _check_shape(news)

    if not isinstance(news, dict):
        return False

    # Visual verification — print every synthesized item for manual review.
    total = sum(len(items) for items in news.values())
    print(f"  synthesized {total} item(s) across {len(news)} categories:")
    for category, items in news.items():
        print(f"  [{category}] {len(items)} item(s)")
        for item in items:
            print(f"    - title:   {item.title}")
            print(f"      summary: {item.summary}")
            print(f"      url:     {item.url}")
            print(f"      source:  {item.source}  zone: {item.zone}")

    # Live search is non-deterministic — don't hard-fail on counts, but every
    # item that *was* produced must carry a non-empty title and summary.
    for category, items in news.items():
        for item in items:
            passed &= _check(
                f"{category!r} item has non-empty title",
                bool(item.title and item.title.strip()),
            )
            passed &= _check(
                f"{category!r} item has non-empty summary",
                bool(item.summary and item.summary.strip()),
            )
    return passed


def main() -> int:
    print("=== news_node ===")
    ok = True
    print("\ntest_no_location_returns_empty:")
    ok &= test_no_location_returns_empty()
    print("\ntest_news_node_live:")
    ok &= test_news_node_live()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
