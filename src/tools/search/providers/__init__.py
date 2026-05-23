"""Search-provider registry.

The single source of truth the :func:`src.tools.search.get_search_provider`
factory reads. Adding a backend is one new module here plus one entry in
:data:`PROVIDERS` — nothing else changes.
"""

from src.tools.search.base import SearchProvider
from src.tools.search.providers.duckduckgo import DuckDuckGoProvider
from src.tools.search.providers.searxng_crawl4ai import SearxngCrawl4aiProvider
from src.tools.search.providers.tavily import TavilyProvider

#: Provider slug -> provider class. Mirrors the scraper's ADAPTERS registry.
PROVIDERS: dict[str, type[SearchProvider]] = {
    TavilyProvider.name: TavilyProvider,
    DuckDuckGoProvider.name: DuckDuckGoProvider,
    SearxngCrawl4aiProvider.name: SearxngCrawl4aiProvider,
}

__all__ = ["PROVIDERS"]
