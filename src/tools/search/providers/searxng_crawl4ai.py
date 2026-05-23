"""Two-stage SearXNG + Crawl4AI provider — the free, self-hosted fallback.

Pipeline:

1. Hit a local SearXNG instance (``SEARXNG_URL``, default
   ``http://localhost:8080``) for the top-K result URLs. SearXNG's JSON
   endpoint is disabled by default (returns 403); we parse the HTML results
   page with BeautifulSoup instead, which works against any vanilla SearXNG
   container without config changes. The URL list is identical to what JSON
   would return — relevance ranking is format-independent.
2. Pass those URLs to a sandboxed Crawl4AI invocation that fetches each one
   in a Playwright-backed browser and extracts clean main-content markdown.

The Crawl4AI step runs in a **separate ``uv``-managed venv** declared by the
PEP 723 inline metadata in :mod:`scripts._crawl4ai_extract`. This is the
only way to use real Crawl4AI from this project: Crawl4AI pins
``lxml~=5.3`` and the project pins ``lxml>=6.1`` via ``scrapling[ai]``, so
the two libraries cannot coexist in one venv.

This provider is **disabled by default** — Tavily is the production default
because it's ~20× faster and yields richer summaries. Enable this one for
zero-recurring-cost deployments by setting ``LOCAL_SEARCH=yes`` (see the
:func:`src.tools.search.get_search_provider` factory).
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from src.tools.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

#: Resolve the helper script path relative to this file so the provider
#: works regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CRAWL4AI_SCRIPT = _REPO_ROOT / "scripts" / "_crawl4ai_extract.py"
_CRAWL4AI_SENTINEL = "\n<<<CRAWL4AI_URL_BOUNDARY>>>\n"

#: ``uv`` invocation. Pins are duplicated in the script's PEP 723 metadata —
#: keep them in sync. ``playwright==1.59.0`` matches the chromium revision
#: (rev 1217) already cached at ``~/.cache/ms-playwright``.
_UV_CMD_PREFIX = (
    "uv", "run", "--no-project", "--quiet",
    "--python", "3.12",
    "--with", "lxml>=5.3,<5.4",
    "--with", "crawl4ai>=0.8",
    "--with", "playwright==1.59.0",
)


class SearxngCrawl4aiProvider(SearchProvider):
    """SearXNG-discovers + Crawl4AI-extracts. Free, self-hosted, ~14s/query."""

    name = "searxng_crawl4ai"

    #: Whole-batch ceiling for one ``search()`` call (sum of N Crawl4AI fetches).
    TOTAL_TIMEOUT_S = 90

    def __init__(self) -> None:
        self.searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Discover via SearXNG, extract via Crawl4AI sandbox; never raises."""
        urls = self._searxng_urls(query, max_results)
        if not urls:
            return []
        chunks = self._crawl4ai_extract(urls)
        return [
            SearchResult(title=url, url=url, snippet=md)
            for url, md in chunks
            if md and not md.startswith("ERROR:")
        ]

    # --------------------------------------------------------------- helpers

    def _searxng_urls(self, query: str, max_results: int) -> list[str]:
        """Hit SearXNG's HTML results page; return top-K result URLs.

        Works against any vanilla SearXNG container — does not require the
        ``search.formats: [json]`` config flag.
        """
        try:
            resp = httpx.get(
                f"{self.searxng_url}/search",
                params={"q": query, "language": "es", "safesearch": "0"},
                timeout=10.0,
                headers={"User-Agent": "Mozilla/5.0 (estatia)"},
            )
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001 — SearXNG down -> empty result.
            logger.warning("SearXNG unreachable at %s: %s", self.searxng_url, e)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls: list[str] = []
        seen: set[str] = set()
        for article in soup.select("article.result"):
            a = article.select_one("a.url_header") or article.select_one("a[href]")
            href = a.get("href") if a else None
            if href and href.startswith("http") and href not in seen:
                seen.add(href)
                urls.append(href)
                if len(urls) >= max_results:
                    break
        return urls

    def _crawl4ai_extract(self, urls: list[str]) -> list[tuple[str, str]]:
        """Drive Crawl4AI in its PEP 723 sandbox; return ``[(url, markdown)]``."""
        if not _CRAWL4AI_SCRIPT.is_file():
            logger.warning("crawl4ai helper script missing at %s", _CRAWL4AI_SCRIPT)
            return []
        try:
            proc = subprocess.run(
                [*_UV_CMD_PREFIX, str(_CRAWL4AI_SCRIPT), *urls],
                capture_output=True, text=True, timeout=self.TOTAL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            logger.warning("crawl4ai sandbox timed out after %ss", self.TOTAL_TIMEOUT_S)
            return []
        except FileNotFoundError:
            logger.warning("uv binary not found — cannot launch Crawl4AI sandbox")
            return []

        # The helper emits blocks of the form: SENTINEL + "URL: <url>\n<md>\n"
        out: list[tuple[str, str]] = []
        for block in (proc.stdout or "").split(_CRAWL4AI_SENTINEL):
            block = block.strip()
            if not block.startswith("URL:"):
                continue
            header, _, body = block.partition("\n")
            url = header[len("URL:"):].strip()
            out.append((url, body.strip()))
        return out
