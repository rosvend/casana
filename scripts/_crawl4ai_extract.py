#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "crawl4ai>=0.8",
#     # Pinned: chromium-headless-shell rev 1217 (cached in ~/.cache/ms-playwright)
#     # ships with playwright 1.59.x. Newer playwright versions expect rev 1223
#     # which our cache doesn't have and uv can't install without --with-deps.
#     "playwright==1.59.0",
# ]
# ///
"""Sandboxed Crawl4AI extractor (PEP 723 inline-metadata script).

The main benchmark cannot install crawl4ai in its venv: crawl4ai pins
``lxml~=5.3`` and the project's ``scrapling[ai]`` requires ``lxml>=6.1``.
PEP 723 inline metadata lets ``uv`` build an ephemeral, fully isolated venv
for *this script alone* so we can drive the real Crawl4AI without touching
the project's lockfile.

Invocation (from the main benchmark):

    uv run --no-project --quiet scripts/_crawl4ai_extract.py <url1> <url2> ...

Reads URLs from argv; for each, fetches via :class:`AsyncWebCrawler` and
writes a ``SENTINEL``-separated block to stdout, so the parent process can
slice the per-URL markdown back apart. Errors are recorded inline rather
than raising — one bad URL doesn't kill the whole batch.
"""

from __future__ import annotations

import asyncio
import sys

SENTINEL = "\n<<<CRAWL4AI_URL_BOUNDARY>>>\n"
PER_URL_TIMEOUT_S = 20.0
MAX_CHARS_PER_URL = 6000  # parent caps prompt budget; keep raw text honest.


async def _fetch_all(urls: list[str]) -> None:
    # Imported inside the function so a `--help`/no-arg run doesn't pay the
    # crawl4ai (and Playwright) import cost.
    from crawl4ai import AsyncWebCrawler, BrowserConfig

    # crawl4ai's BrowserConfig has TWO channel attributes (``chrome_channel``
    # AND ``channel``) that both default to ``'chromium'`` and both feed into
    # Playwright's launch as ``channel='chromium'`` — which makes Playwright
    # hunt for the chromium-1223 stable-channel binary we don't have.
    # BrowserConfig's validator rewrites empty/None back to the default at
    # construction, so we override BOTH attributes after construction to fall
    # back to Playwright's bundled chromium-1217 (already in ms-playwright).
    cfg = BrowserConfig(headless=True, browser_type="chromium")
    cfg.chrome_channel = ""
    cfg.channel = ""
    async with AsyncWebCrawler(config=cfg, verbose=False) as crawler:
        for url in urls:
            sys.stdout.write(SENTINEL)
            sys.stdout.write(f"URL: {url}\n")
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=url), timeout=PER_URL_TIMEOUT_S
                )
                md = (result.markdown or "")
                if hasattr(md, "raw_markdown"):  # crawl4ai 0.5+: MarkdownGenerationResult
                    md = md.raw_markdown or ""
                md = str(md)[:MAX_CHARS_PER_URL]
                sys.stdout.write(md)
            except asyncio.TimeoutError:
                sys.stdout.write(f"ERROR: timeout after {PER_URL_TIMEOUT_S}s")
            except Exception as e:  # noqa: BLE001
                sys.stdout.write(f"ERROR: {type(e).__name__}: {e}")
            sys.stdout.write("\n")
            sys.stdout.flush()


def main() -> int:
    urls = sys.argv[1:]
    if not urls:
        print("usage: _crawl4ai_extract.py <url> [<url> ...]", file=sys.stderr)
        return 2
    asyncio.run(_fetch_all(urls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
