"""Tavily search backend.

Wraps :class:`langchain_community.tools.TavilySearchResults` with
``include_raw_content=True`` so the LLM downstream sees the full page bodies,
not just headline snippets. Needs ``TAVILY_API_KEY`` in the environment.

The benchmark in :mod:`scripts.benchmark_search` selected Tavily as the
``news_agent`` default for two reasons:

- richest summaries (~320 chars avg vs ~250 from snippet-only DuckDuckGo);
- lowest end-to-end latency (~0.7s/query — faster than DDG and ~20× faster
  than SearXNG + Crawl4AI), because Tavily's API does the crawl-and-extract
  step on its own infrastructure.

Cost: free up to 1,000 queries/month; ~$0.005–0.01/query past that.
"""

from __future__ import annotations

import logging

from src.tools.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class TavilyProvider(SearchProvider):
    """Tavily web search with full-page raw content enabled."""

    name = "tavily"

    def __init__(self) -> None:
        # Construction is deferred to ``search()`` so a missing TAVILY_API_KEY
        # or a missing tavily-python wheel surfaces as an empty result (the
        # SearchProvider contract), not as an exception at provider-resolve
        # time.
        self._tool = None

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run one Tavily query; never raises (see SearchProvider contract)."""
        if self._tool is None:
            try:
                from langchain_community.tools import TavilySearchResults

                self._tool = TavilySearchResults(
                    max_results=5, include_raw_content=True
                )
            except Exception as e:  # noqa: BLE001 — missing key, missing SDK, etc.
                logger.warning("tavily init failed: %s", e)
                return []

        try:
            raw = self._tool.invoke({"query": query})
        except Exception as e:  # noqa: BLE001 — network, rate limit, transient.
            logger.warning("tavily search failed for %r: %s", query, e)
            return []
        if isinstance(raw, str):
            # Some versions of TavilySearchResults return a stringified dump
            # when raw_content parsing fails. Surface it as one fallback hit.
            return [SearchResult(title="tavily-string-result", url="", snippet=raw)]

        out: list[SearchResult] = []
        for hit in (raw or [])[:max_results]:
            try:
                # 'content' is the LLM-curated snippet; 'raw_content' is the
                # full extracted page text. Concatenate so the downstream
                # extractor sees both layers.
                snippet = hit.get("content", "") or ""
                body = hit.get("raw_content") or ""
                full = snippet if not body else f"{snippet}\n\n{body[:4000]}"
                out.append(
                    SearchResult(
                        title=hit.get("title", "") or "",
                        url=hit.get("url", "") or "",
                        snippet=full,
                    )
                )
            except Exception:  # noqa: BLE001 — skip a malformed hit.
                continue
        return out
