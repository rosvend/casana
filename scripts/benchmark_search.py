"""Benchmark three web-search backends against the same area-news queries.

For each provider we measure:

1. **Latency** — wall-clock seconds for the full search pipeline.
2. **Data volume** — total characters returned.
3. **Extraction quality** — whether ``gpt-4o-mini`` with the same
   ``NewsExtraction`` schema used by :func:`src.agents.news_agent.news_node`
   recovers ``NewsItem`` objects with substantive summaries (not just titles).

Three pipelines compared:

- **DuckDuckGo** (baseline) — ``DuckDuckGoSearchResults`` from
  ``langchain_community``. Free, no key.
- **Tavily** — ``TavilySearchResults`` from ``langchain_community`` with
  ``include_raw_content=True``. Needs ``TAVILY_API_KEY``.
- **SearXNG + Crawl4AI** — two-stage. First a request to the local SearXNG
  instance (``SEARXNG_URL``, default ``http://localhost:8080``) to get top
  result URLs; second, those URLs are crawled by **real Crawl4AI** running
  in a PEP 723 ``uv``-managed sandbox (``scripts/_crawl4ai_extract.py``).
  The sandbox is needed because Crawl4AI pins ``lxml~=5.3`` and the project
  pins ``lxml>=6.1`` via ``scrapling[ai]`` — they cannot coexist in one venv.

  SearXNG's JSON endpoint is disabled by default config (returns 403); this
  script parses the HTML results page instead, which works out of the box.

Run:

    uv run python -m scripts.benchmark_search

Requires ``OPENAI_API_KEY`` and ``TAVILY_API_KEY`` in ``.env``, SearXNG on
``localhost:8080``, and Crawl4AI's sandbox warmed (one-time) by running
``scripts/_crawl4ai_extract.py`` once via uv.
"""

from __future__ import annotations

import logging
import os
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI

# Reuse the EXACT extraction contract news_node uses — same schema, same prompt.
from src.agents.news_agent import MODEL, SYSTEM_PROMPT, NewsExtraction

load_dotenv()

logging.basicConfig(level=logging.WARNING)
for name in ("httpx", "httpcore", "ddgs", "langchain"):
    logging.getLogger(name).setLevel(logging.ERROR)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080")
SEARXNG_TOP_K = 5

# Sentinel + uv invocation matching scripts/_crawl4ai_extract.py's contract.
CRAWL4AI_SCRIPT = "scripts/_crawl4ai_extract.py"
CRAWL4AI_SENTINEL = "\n<<<CRAWL4AI_URL_BOUNDARY>>>\n"
CRAWL4AI_TOTAL_TIMEOUT_S = 90  # whole batch — crawl4ai per-URL is 20s

#: Minimum summary length (chars) for an item to count as "well-formed".
#: A short title-echo would slip past `bool(summary)` — this bars that.
WELL_FORMED_MIN_SUMMARY = 80

#: The five standardized benchmark queries (mixed zones and categories).
QUERIES: list[str] = [
    "seguridad Guayabal Medellin noticias 2026",
    "crimen Laureles Medellin 2026",
    "transporte El Poblado Medellin 2026",
    "desarrollo urbano e infraestructura Envigado 2026",
    "mercado inmobiliario precios vivienda Medellin 2026",
]


# ------------------------------------------------------------------- Adapters


def search_duckduckgo(query: str) -> str:
    """Baseline — DuckDuckGoSearchResults(output_format='list') concatenated."""
    tool = DuckDuckGoSearchResults(output_format="list")
    raw = tool.invoke(query) or []
    lines: list[str] = []
    for hit in raw:
        lines.append(f"Title: {hit.get('title', '')}")
        lines.append(f"URL:   {hit.get('link', '')}")
        lines.append(f"Snippet: {hit.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines)


def search_tavily(query: str) -> str:
    """Tavily — include_raw_content=True so the LLM sees full page text."""
    from langchain_community.tools import TavilySearchResults

    tool = TavilySearchResults(max_results=5, include_raw_content=True)
    raw = tool.invoke({"query": query})
    if isinstance(raw, str):
        return raw
    lines: list[str] = []
    for hit in raw or []:
        lines.append(f"Title: {hit.get('title', '')}")
        lines.append(f"URL:   {hit.get('url', '')}")
        snippet = hit.get("content", "") or ""
        body = hit.get("raw_content") or ""
        lines.append(f"Snippet: {snippet}")
        if body:
            lines.append(f"Body: {body[:4000]}")
        lines.append("")
    return "\n".join(lines)


def _searxng_top_urls(query: str) -> list[str]:
    """Hit SearXNG's HTML results page and extract the top-K URLs.

    The JSON endpoint requires ``search.formats: [json]`` in SearXNG's
    ``settings.yml`` and is off by default (returns 403). Parsing HTML works
    against any vanilla SearXNG container.
    """
    resp = httpx.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "language": "es", "safesearch": "0"},
        timeout=10.0,
        headers={"User-Agent": "Mozilla/5.0 (estatia-benchmark)"},
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls: list[str] = []
    seen: set[str] = set()
    for article in soup.select("article.result"):
        a = article.select_one("a.url_header") or article.select_one("a[href]")
        href = a.get("href") if a else None
        if href and href.startswith("http") and href not in seen:
            seen.add(href)
            urls.append(href)
            if len(urls) >= SEARXNG_TOP_K:
                break
    return urls


# Cached uv invocation for the Crawl4AI sandbox — saves recomputing args.
# Pins are duplicated in the script's PEP 723 metadata; keep them in sync.
_CRAWL4AI_CMD_PREFIX = [
    "uv", "run", "--no-project", "--quiet",
    "--python", "3.12",
    "--with", "lxml>=5.3,<5.4",
    "--with", "crawl4ai>=0.8",
    "--with", "playwright==1.59.0",  # browser rev 1217 already cached
    CRAWL4AI_SCRIPT,
]


def _crawl4ai_extract(urls: list[str]) -> str:
    """Drive real Crawl4AI in its PEP 723 sandbox; return concatenated markdown.

    Spawns ``uv run --no-project scripts/_crawl4ai_extract.py <urls>``. The
    sandbox keeps Crawl4AI's incompatible ``lxml~=5.3`` away from the project
    venv. Per-URL chunks are emitted with a fixed sentinel so we re-glue the
    output cleanly even if a single fetch errored mid-batch.
    """
    if not urls:
        return ""
    try:
        proc = subprocess.run(
            [*_CRAWL4AI_CMD_PREFIX, *urls],
            capture_output=True, text=True,
            timeout=CRAWL4AI_TOTAL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return ""
    # The sandbox prints non-fatal warnings to stderr; only stdout carries data.
    return proc.stdout or ""


def search_searxng_crawl(query: str) -> str:
    """SearXNG (HTML) → real Crawl4AI markdown extraction → concatenate."""
    urls = _searxng_top_urls(query)
    if not urls:
        return ""
    return _crawl4ai_extract(urls)


# ------------------------------------------------------- Quality (LLM check)


_LLM = ChatOpenAI(model=MODEL, temperature=0).with_structured_output(
    NewsExtraction, method="function_calling"
)


@dataclass
class QualityScore:
    """Per-(provider, query) LLM-extraction outcome."""
    items_total: int
    items_well_formed: int        # title AND summary >= WELL_FORMED_MIN_SUMMARY chars
    avg_summary_len: float        # across the well-formed items (0 if none)
    error: str | None = None


def extract_items(text: str, query: str) -> QualityScore:
    """Run the news-agent LLM extraction over one provider's text blob.

    "Well-formed" requires both a non-empty title AND a summary of at least
    ``WELL_FORMED_MIN_SUMMARY`` characters — content, not just a headline echo.
    """
    if not text.strip():
        return QualityScore(0, 0, 0.0, error="empty input")
    try:
        result: NewsExtraction = _LLM.invoke(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    f"Consulta del usuario: {query}\n\n"
                    f"Resultados crudos de búsqueda:\n{text}",
                ),
            ]
        )
    except Exception as e:  # noqa: BLE001 — degraded extraction is data too.
        return QualityScore(0, 0, 0.0, error=f"{type(e).__name__}: {e}")

    all_items = (
        result.crime_safety + result.transportation + result.infrastructure
        + result.events + result.market_trends
    )
    summary_lens: list[int] = []
    for it in all_items:
        title = (it.title or "").strip()
        summary = (it.summary or "").strip()
        if title and len(summary) >= WELL_FORMED_MIN_SUMMARY:
            summary_lens.append(len(summary))
    return QualityScore(
        items_total=len(all_items),
        items_well_formed=len(summary_lens),
        avg_summary_len=statistics.mean(summary_lens) if summary_lens else 0.0,
    )


# ----------------------------------------------------- Benchmark scaffolding


@dataclass
class QueryResult:
    """One (provider, query) measurement."""
    provider: str
    query: str
    latency_s: float
    chars: int
    items_total: int
    items_well_formed: int
    avg_summary_len: float
    error: str | None = None

    @property
    def quality_label(self) -> str:
        if self.error or self.chars == 0:
            return "No"
        if self.items_well_formed >= 2:
            return "Yes"
        if self.items_well_formed >= 1:
            return "Partial"
        return "No"


@dataclass
class ProviderAgg:
    """Aggregate of one provider across all queries."""
    name: str
    cost_note: str
    results: list[QueryResult] = field(default_factory=list)

    def add(self, r: QueryResult) -> None:
        self.results.append(r)

    @property
    def avg_latency(self) -> float:
        return statistics.mean(r.latency_s for r in self.results) if self.results else 0.0

    @property
    def avg_chars(self) -> int:
        return sum(r.chars for r in self.results) // len(self.results) if self.results else 0

    @property
    def total_well_formed(self) -> int:
        return sum(r.items_well_formed for r in self.results)

    @property
    def avg_summary_len(self) -> float:
        nonzero = [r.avg_summary_len for r in self.results if r.avg_summary_len > 0]
        return statistics.mean(nonzero) if nonzero else 0.0

    @property
    def success_label(self) -> str:
        yes = sum(1 for r in self.results if r.quality_label == "Yes")
        part = sum(1 for r in self.results if r.quality_label == "Partial")
        no = sum(1 for r in self.results if r.quality_label == "No")
        return f"{yes} Yes / {part} Partial / {no} No"


# --------------------------------------------------------------------- Run


def run_one(provider: str, fn: Callable[[str], str], query: str) -> QueryResult:
    """Time ``fn(query)``, extract items, and package a QueryResult."""
    print(f"  [{provider}] {query!r}")
    t0 = time.monotonic()
    error: str | None = None
    text = ""
    try:
        text = fn(query)
    except Exception as e:  # noqa: BLE001 — captured as a benchmark datapoint.
        error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    if error:
        score = QualityScore(0, 0, 0.0, error=error)
    else:
        score = extract_items(text, query)

    qr = QueryResult(
        provider=provider, query=query, latency_s=latency, chars=len(text),
        items_total=score.items_total, items_well_formed=score.items_well_formed,
        avg_summary_len=score.avg_summary_len, error=error or score.error,
    )
    if qr.error:
        suffix = f" — ERROR: {qr.error}"
    else:
        suffix = (
            f" — {latency:.2f}s, {len(text):,} chars, "
            f"{qr.items_well_formed}/{qr.items_total} well-formed "
            f"(avg summary {qr.avg_summary_len:.0f} chars)"
        )
    print(f"    {qr.quality_label}{suffix}")
    return qr


def render_markdown(aggs: list[ProviderAgg]) -> str:
    """Format the final per-provider comparison as a Markdown table."""
    header = (
        "| Provider | Avg latency (s) | Avg chars / query | Well-formed items "
        "| Avg summary chars | Successful extractions | Cost / complexity |\n"
        "|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {a.name} | {a.avg_latency:.2f} | {a.avg_chars:,} | {a.total_well_formed} "
        f"| {a.avg_summary_len:.0f} | {a.success_label} | {a.cost_note} |"
        for a in aggs
    ]
    return "\n".join([header, *rows])


def _warm_crawl4ai_sandbox() -> None:
    """One-off ``uv run`` so the resolver/install/playwright cost is excluded
    from the first SearXNG+Crawl4AI measurement."""
    print("  warming Crawl4AI sandbox (uv resolve + playwright launch) ...")
    t0 = time.monotonic()
    try:
        subprocess.run(
            [*_CRAWL4AI_CMD_PREFIX, "https://example.com"],
            capture_output=True, text=True, timeout=180,
        )
        print(f"  warmed in {time.monotonic() - t0:.1f}s.")
    except subprocess.TimeoutExpired:
        print("  WARN: warmup timed out; the benchmark may still work but the first call will be slow.")


def main() -> int:
    print("=== search-backend benchmark ===")
    print(f"queries: {len(QUERIES)}")
    print(f"SearXNG: {SEARXNG_URL}")
    print(f"uv:      {shlex.join(_CRAWL4AI_CMD_PREFIX[:1])}\n")
    _warm_crawl4ai_sandbox()

    providers: list[tuple[str, Callable[[str], str], str]] = [
        (
            "DuckDuckGo",
            search_duckduckgo,
            "Free, no key. Snippet-only text; subject to DDG rate limits.",
        ),
        (
            "Tavily",
            search_tavily,
            "API key. ~$0.005-0.01/query past 1k/mo free tier. Full page bodies.",
        ),
        (
            "SearXNG + Crawl4AI",
            search_searxng_crawl,
            "Free. Self-hosted SearXNG (Docker) + Crawl4AI sandbox (extra venv via uv).",
        ),
    ]

    aggs: list[ProviderAgg] = []
    for name, fn, cost in providers:
        print(f"\n--- {name} ---")
        agg = ProviderAgg(name=name, cost_note=cost)
        for q in QUERIES:
            agg.add(run_one(name, fn, q))
        aggs.append(agg)

    table = render_markdown(aggs)
    print("\n\n=== RESULTS ===\n")
    print(table)

    # Pick a winner: most well-formed items wins; ties broken by avg summary
    # length, then by speed. Content depth matters as much as count.
    rankable = [a for a in aggs if a.total_well_formed > 0]
    if rankable:
        winner = max(
            rankable,
            key=lambda a: (a.total_well_formed, a.avg_summary_len, -a.avg_latency),
        )
        recommendation = (
            f"\n**Recommendation:** wire `{winner.name}` into `news_agent` — "
            f"it yielded {winner.total_well_formed} well-formed items "
            f"(avg summary {winner.avg_summary_len:.0f} chars; "
            f"{winner.success_label}) at {winner.avg_latency:.2f}s/query."
        )
    else:
        recommendation = (
            "\n**Recommendation:** no provider returned usable items in this "
            "run. Re-check keys / SearXNG / Crawl4AI sandbox; stay on "
            "DuckDuckGo as the zero-config baseline."
        )
    print(recommendation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
