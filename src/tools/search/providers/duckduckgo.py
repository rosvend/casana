"""DuckDuckGo search backend.

Wraps :class:`langchain_community.tools.DuckDuckGoSearchResults` (backed by the
``ddgs`` package) — a completely free, zero-config web search. Constructed with
``output_format="list"`` so the tool yields structured ``list[dict]`` payloads
(keys ``title`` / ``link`` / ``snippet``) instead of one joined string.
"""

from __future__ import annotations

import logging

from langchain_community.tools import DuckDuckGoSearchResults

from src.tools.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(SearchProvider):
    """Free, zero-config web search via DuckDuckGo."""

    name = "duckduckgo"

    def __init__(self) -> None:
        # output_format="list" -> invoke() returns list[dict] with keys
        # 'title', 'link', 'snippet' rather than a single joined string.
        self._tool = DuckDuckGoSearchResults(output_format="list")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run one DuckDuckGo query; never raises (see SearchProvider contract)."""
        try:
            raw = self._tool.invoke(query)
        except Exception as e:  # noqa: BLE001 — backend may rate-limit; degrade, don't crash.
            logger.warning("duckduckgo search failed for %r: %s", query, e)
            return []

        out: list[SearchResult] = []
        for item in (raw or [])[:max_results]:
            try:
                out.append(
                    SearchResult(
                        title=item.get("title", "") or "",
                        url=item.get("link", "") or "",
                        snippet=item.get("snippet", "") or "",
                    )
                )
            except Exception:  # noqa: BLE001 — skip a malformed hit, keep the rest.
                continue
        return out
