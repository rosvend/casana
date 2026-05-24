"""Shared, portal-agnostic scraping infrastructure.

Everything in here is reused by *every* portal adapter — low-level fetching,
scalar parsers, regex helpers, coordinate sniffing, WhatsApp link building, and
the canonical property-type / transaction maps. Portal-specific parsing lives
in :mod:`src.tools.scraper.adapters`; this module deliberately knows nothing
about any one site.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)


# Canonical lookup tables (shared across portals).

# user-facing slug -> (fincaraiz_plural, metrocuadrado_singular, canonical_en)
_PROPERTY_TYPE_MAP: dict[str, tuple[str, str, str]] = {
    "apartamentos": ("apartamentos", "apartamento", "apartment"),
    "casas": ("casas", "casa", "house"),
    "locales": ("locales", "local", "commercial"),
    "oficinas": ("oficinas", "oficina", "office"),
    "fincas": ("fincas", "finca", "country_house"),
}

# user-facing slug -> canonical English transaction
_TRANSACTION_MAP: dict[str, str] = {
    "arriendo": "rent",
    "venta": "sale",
}


# Shared low-level helpers (mostly lifted from the original PoC).


def _fetch_page(url: str):
    """Stealthy fetch. Lazy import so the missing-binary error only fires when
    scraping is actually attempted.

    ``solve_cloudflare`` is disabled because the target portals (Finca Raíz,
    Metro Cuadrado) don't gate behind Turnstile; leaving it on caused 60-90 s
    of wasted retries per fetch hunting for a challenge that never appears.
    """
    try:
        from scrapling.fetchers import StealthyFetcher
    except ImportError as e:
        raise RuntimeError("scrapling import failed — run `uv sync` first") from e

    StealthyFetcher.adaptive = True
    return StealthyFetcher.fetch(
        url,
        headless=True,
        network_idle=True,
        solve_cloudflare=False,
        timeout=90_000,
    )


def _page_html(page) -> str:
    """Return the page's raw HTML as a string, regardless of Scrapling version."""
    for attr in ("html_content", "body", "text"):
        val = getattr(page, attr, None)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, (bytes, bytearray)) and val:
            try:
                return val.decode("utf-8", errors="replace")
            except Exception:
                continue
    return ""


def _parse_cop_price(text: str | None) -> float | None:
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    return float(digits) if digits else None


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _parse_area(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mt)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(m.group(1 if m.lastindex else 0).replace(",", ".")) if m else None


def _slug_from_url(url: str) -> str:
    # Slug from the path's last segment only — drop any ?query/#fragment so
    # tracking params (e.g. MC's ?src_url=...) don't leak into the id.
    path = urlparse(url).path.rstrip("/")
    tail = path.rsplit("/", 1)[-1] or path or url
    return re.sub(r"[^\w\-]", "_", tail)[:80]


def _safe(fn: Callable[..., Any], *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Run ``fn(*args, **kwargs)`` and swallow any exception, returning ``default``.

    Used to keep one missing DOM node from killing the whole deep-scrape.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — by design
        logger.debug("safe-call failed: %s(%r) → %s", getattr(fn, "__name__", fn), args, e)
        return default


# Coordinate + contact-link extraction (shared across sites).

# Tuple of (pattern, lat_group, lon_group). Patterns are ordered most-specific
# first; the first hit wins.
_COORD_PATTERNS: tuple[tuple[re.Pattern[str], int, int], ...] = (
    # Metro Cuadrado / Next.js __NEXT_DATA__: "latitude":"6.123","longitude":"-75.4"
    (
        re.compile(
            r'"latitude"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?\s*,\s*"longitude"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?'
        ),
        1,
        2,
    ),
    # Reverse order: "longitude":...,"latitude":...
    (
        re.compile(
            r'"longitude"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?\s*,\s*"latitude"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?'
        ),
        2,
        1,
    ),
    # Finca Raíz / generic: "lat":6.123,"lng":-75.4  (or "lon")
    (
        re.compile(
            r'"lat"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?\s*,\s*"l(?:ng|on)"\s*:\s*"?(-?\d{1,3}(?:\.\d+)?)"?'
        ),
        1,
        2,
    ),
    # DOM-attribute fallback (some embed map widgets this way)
    (
        re.compile(
            r'data-lat=["\'](-?\d{1,3}(?:\.\d+)?)["\']\s+data-l(?:ng|on)=["\'](-?\d{1,3}(?:\.\d+)?)["\']'
        ),
        1,
        2,
    ),
)


def _extract_coordinates(html: str) -> dict[str, float] | None:
    """Sniff lat/lon out of inline JSON / data-attributes in the page HTML.

    These portals don't render maps from textual addresses; the coordinates
    are injected by JS, typically inside ``__NEXT_DATA__`` (Next.js) or an
    Angular state blob. Returns ``None`` if nothing plausible is found.
    """
    if not html:
        return None
    for pattern, lat_g, lon_g in _COORD_PATTERNS:
        m = pattern.search(html)
        if not m:
            continue
        try:
            lat = float(m.group(lat_g))
            lon = float(m.group(lon_g))
        except (TypeError, ValueError):
            continue
        # Sanity check: Colombia roughly spans lat 4-12, lon -79 to -67.
        # Allow a wider envelope to stay portable, but reject obvious junk.
        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat != 0 or lon != 0):
            return {"lat": lat, "lon": lon}
    return None


def _format_whatsapp_link(raw_phone: str | None, message: str | None = None) -> str | None:
    """Build an ``api.whatsapp.com`` deep-link for a Colombian mobile number.

    Strips non-digits, drops a leading ``57`` country code if present, and
    validates that the remaining 10 digits start with ``3`` (the CO mobile
    prefix). Returns ``None`` on any failure so callers can skip cleanly.
    """
    if not raw_phone:
        return None
    digits = re.sub(r"\D", "", raw_phone)
    if digits.startswith("57") and len(digits) > 10:
        digits = digits[2:]
    if len(digits) != 10 or not digits.startswith("3"):
        return None
    base = f"https://api.whatsapp.com/send/?phone=57{digits}"
    if message:
        base += f"&text={quote(message, safe='')}"
    return base


def _extract_contact_links(page) -> list[str]:
    """Harvest WhatsApp deep-links (api.whatsapp.com / wa.me) from the page."""
    try:
        hrefs = page.css(
            "a[href*='api.whatsapp.com']::attr(href), a[href*='wa.me']::attr(href)"
        ).getall()
    except Exception as e:
        logger.debug("contact-link harvest failed: %s", e)
        return []
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hrefs:
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


# DOM text helpers (shared by the per-portal detail parsers).


def _first_text(page, selector: str) -> str | None:
    """css(...)::text first-hit, trimmed, ``None`` if empty."""
    try:
        val = page.css(selector).get()
    except Exception:
        return None
    if not val:
        return None
    val = val.strip()
    return val or None


def _joined_text(page, selector: str) -> str | None:
    """css(...)::text -> all hits joined with whitespace, collapsed."""
    try:
        parts = page.css(selector).getall()
    except Exception:
        return None
    if not parts:
        return None
    text = " ".join(p.strip() for p in parts if p and p.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _find_labeled_int(html_text: str, label_pattern: str) -> int | None:
    """Find ``label: N`` style integers in unstructured page text.

    ``label_pattern`` is wrapped in a non-capturing group so callers can pass
    alternations like ``"parqueader[oa]s?|garaj[ea]s?"`` without ``|``
    binding looser than the surrounding context and dropping ``\\d+`` out of
    the match.
    """
    if not html_text:
        return None
    m = re.search(
        rf"(?:{label_pattern})\s*[:\-]?\s*(\d+)",
        html_text,
        re.IGNORECASE,
    )
    if not m:
        return None
    captured = m.group(1)
    return int(captured) if captured is not None else None


def _find_labeled_area(html_text: str, label_pattern: str) -> float | None:
    """Find ``label: N m2`` style areas in unstructured page text."""
    if not html_text:
        return None
    m = re.search(
        rf"(?:{label_pattern})\s*[:\-]?\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mt)",
        html_text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def _infer_types_from_url(url: str) -> tuple[str | None, str | None]:
    """Best-effort recovery of (property_type, transaction_type) from the URL path."""
    path = urlparse(url).path.lower()
    prop = None
    for slug, (_, _, canonical) in _PROPERTY_TYPE_MAP.items():
        if slug in path or slug.rstrip("s") in path:
            prop = canonical
            break
    trans = None
    for slug, canonical in _TRANSACTION_MAP.items():
        if slug in path:
            trans = canonical
            break
    return prop, trans


# Filter helpers (shared by the discovery orchestrator).


def _collect_filters(**kwargs: int | None) -> dict[str, int]:
    """Drop None values and return a plain dict keyed by canonical filter name.

    Canonical keys: ``min_price``, ``max_price``, ``bedrooms``, ``bathrooms``,
    ``estrato``, ``min_area_m2``, ``max_area_m2``, ``parking_lots``, ``longevity``.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _passes_filters(record: dict, filters: dict[str, int]) -> bool:
    """Best-effort post-filter on the shallow ``{id, url, price}`` record.

    Currently only ``price`` is reliably present at the shallow stage, so this
    enforces ``min_price``/``max_price`` and lets unknown fields pass through.
    Records with ``price = None`` are kept (we can't disprove the constraint)
    — the deep enricher will revisit.
    """
    price = record.get("price")
    if price is None:
        return True
    if (mn := filters.get("min_price")) is not None and price < mn:
        return False
    if (mx := filters.get("max_price")) is not None and price > mx:
        return False
    return True


__all__ = [
    "logger",
    "_PROPERTY_TYPE_MAP",
    "_TRANSACTION_MAP",
    "_fetch_page",
    "_page_html",
    "_parse_cop_price",
    "_parse_int",
    "_parse_area",
    "_slug_from_url",
    "_safe",
    "_extract_coordinates",
    "_format_whatsapp_link",
    "_extract_contact_links",
    "_first_text",
    "_joined_text",
    "_find_labeled_int",
    "_find_labeled_area",
    "_infer_types_from_url",
    "_collect_filters",
    "_passes_filters",
]
