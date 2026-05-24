"""Fast, offline checks for properties_node concurrency + source balancing.

The live test (test_properties_node.py) proves the node works end-to-end
against the real portals but takes many minutes. This one fakes the scraper
tools so the *orchestration* — balanced source selection and concurrent
enrichment — can be verified deterministically in well under a second.

    uv run python -m tests.test_properties_node_concurrency
"""

from __future__ import annotations

import importlib
import sys
import threading
import time

# ``src/agents/__init__.py`` re-binds each ``xxx_agent`` submodule name to its
# node *function* (so the graph can do ``from src.agents import properties_agent``
# and get a callable). That rebinding shadows the submodule on the package's
# namespace, so ``import src.agents.properties_agent as pa`` returns the
# *function*, not the module — and ``pa.search_listings = ...`` silently
# assigns onto the function instead of patching the module global the node
# actually reads. Reach for the real module via ``sys.modules`` instead.
from src.agents.properties_agent import _balanced_shortlist, properties_node
from src.state import Listing, PropertyFinderState

pa = importlib.import_module("src.agents.properties_agent")
assert pa is sys.modules["src.agents.properties_agent"], (
    "properties_agent module shadowed — patches will not take effect"
)

ENRICH_SLEEP = 0.5  # simulated per-fetch latency


class _FakeTool:
    """Stand-in for a @tool object — exposes only the .invoke the node uses."""

    def __init__(self, fn):
        self._fn = fn

    def invoke(self, arg):
        return self._fn(arg)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def test_balanced_shortlist() -> bool:
    """8 Finca Raíz + 5 Metro Cuadrado stubs, cap 5 -> 3 FR + 2 MC, interleaved."""
    stubs = (
        [{"id": f"fincaraiz:{i}", "url": f"https://fr/{i}"} for i in range(8)]
        + [{"id": f"metrocuadrado:{i}", "url": f"https://mc/{i}"} for i in range(5)]
    )
    short = _balanced_shortlist(stubs, 5)
    sources = [s["id"].split(":")[0] for s in short]
    fr = sources.count("fincaraiz")
    mc = sources.count("metrocuadrado")
    ok = len(short) == 5 and fr == 3 and mc == 2
    passed = _check("balanced shortlist draws from both portals", ok,
                    f"got {fr} fincaraiz + {mc} metrocuadrado")
    # A flat slice would have been 5 fincaraiz / 0 metrocuadrado.
    passed &= _check("interleaved, not front-loaded", sources[0] != sources[1],
                     f"order={sources}")
    return passed


def test_concurrent_enrichment() -> bool:
    """Node must fan enrichment out across threads, not serialize the fetches."""
    threads_seen: set[str] = set()

    def fake_search(_arg):
        return (
            [{"id": f"fincaraiz:{i}", "url": f"https://fr/{i}", "price": None}
             for i in range(8)]
            + [{"id": f"metrocuadrado:{i}", "url": f"https://mc/{i}", "price": None}
               for i in range(5)]
        )

    def fake_extract(arg):
        threads_seen.add(threading.current_thread().name)
        time.sleep(ENRICH_SLEEP)  # simulate a slow stealthy fetch
        url = arg["url"]
        src = "fincaraiz" if "fr/" in url else "metrocuadrado"
        return Listing(id=f"{src}:{url.rsplit('/', 1)[-1]}", source_site=src, url=url,
                       bathrooms=2)

    pa.search_listings = _FakeTool(fake_search)
    pa.extract_property_details = _FakeTool(fake_extract)

    state: PropertyFinderState = {"requirements": None}
    start = time.monotonic()
    result = properties_node(state)
    elapsed = time.monotonic() - start

    listings = result.get("raw_listings", [])
    by_source = {s: sum(1 for l in listings if l.source_site == s)
                 for s in ("fincaraiz", "metrocuadrado")}

    passed = _check("enriched 5 listings (MAX_ENRICH)", len(listings) == 5,
                    f"got {len(listings)}")
    passed &= _check("both portals present in raw_listings",
                     by_source["fincaraiz"] > 0 and by_source["metrocuadrado"] > 0,
                     f"{by_source}")
    # 5 fetches x 0.5s = 2.5s sequential; concurrent should be ~0.5s.
    passed &= _check("enrichment ran concurrently", elapsed < 1.5,
                     f"elapsed {elapsed:.2f}s (sequential would be "
                     f"~{5 * ENRICH_SLEEP:.1f}s)")
    passed &= _check("work spread across multiple threads", len(threads_seen) >= 2,
                     f"{len(threads_seen)} thread(s)")
    return passed


def main() -> int:
    print("=== properties_node concurrency + balancing ===")
    ok = True
    print("\ntest_balanced_shortlist:")
    ok &= test_balanced_shortlist()
    print("\ntest_concurrent_enrichment:")
    ok &= test_concurrent_enrichment()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
