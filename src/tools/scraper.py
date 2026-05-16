"""Discover & Enrich scraping tools for Finca Raíz and Metro Cuadrado.

Exposes two ``@tool``-decorated callables:

- :func:`search_listings` — *discoverer*. Hits a portal's search-results page
  and returns lightweight ``{"id", "url", "price"}`` dicts. Cheap.
- :func:`extract_property_details` — *enricher*. Routes a single property URL
  to the correct site-specific parser, deep-scrapes the detail page, and
  returns a fully validated :class:`Listing`.

The two-stage split lets an agent shortlist before paying the stealthy-fetch
cost per property. Both tools degrade gracefully: missing DOM nodes yield
``None`` fields, never exceptions.

Run the file directly to exercise the full pipeline against live sites:

    uv run python -m src.tools.scraper
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urljoin, urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.tools import tool  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from src.state.listings import Listing  # noqa: E402

logger = logging.getLogger(__name__)

MAX_LISTINGS_PER_SOURCE = 10

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


# ---------------------------------------------------------------------------
# Shared low-level helpers (mostly lifted from the original PoC).
# ---------------------------------------------------------------------------


def _fetch_page(url: str):
    """Stealthy fetch with Cloudflare bypass. Lazy import so the missing-binary
    error only fires when scraping is actually attempted."""
    try:
        from scrapling.fetchers import StealthyFetcher
    except ImportError as e:
        raise RuntimeError("scrapling import failed — run `uv sync` first") from e

    StealthyFetcher.adaptive = True
    return StealthyFetcher.fetch(
        url,
        headless=True,
        network_idle=True,
        solve_cloudflare=True,
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
    tail = url.rstrip("/").rsplit("/", 1)[-1] or url
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


# ---------------------------------------------------------------------------
# Coordinate + contact-link extraction (shared across sites).
# ---------------------------------------------------------------------------

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


# Finca Raíz strips the broker phone from the rendered HTML and reveals it
# only through a public GraphQL mutation on the parent (Infocasas) backend.
# Recon (playwright network capture) showed: no auth, no CSRF, no cookies —
# just Content-Type + an x-origin header gate. So we replay the mutation
# from stdlib urllib instead of dragging the broker through a fake form fill.
_FR_LEAD_URL = "https://graph.infocasas.com.uy/graphql"
_FR_LEAD_QUERY = (
    "mutation request_wpp_mutation("
    "$property_id: Int!, $type: PropEntityType, $email: String, "
    "$name: String, $phone: String) {"
    "  requestPhoneAgent("
    "    property_id: $property_id, isWpp: true, type: $type, "
    "    email: $email, name: $name, phone: $phone"
    "  ) { name phone __typename }"
    "}"
)
# Deliberately bot-flagged dummy values (.invalid TLD + obvious phone) so
# brokers reading their CRM can identify the lead as automated rather than
# being misled by realistic-looking fakes.
_FR_LEAD_DUMMY = {
    "name": "Estatia Verification Bot",
    "email": "noreply@example.invalid",
    "phone": "+573000000000",
}


def _fincaraiz_property_id(url: str) -> int | None:
    """Extract the trailing int id from a Finca Raíz detail URL.

    e.g. ``/apartamento-en-arriendo-en-el-poblado-medellin/193388258`` -> 193388258.
    """
    m = re.search(r"/(\d{4,})(?:[/?#]|$)", url)
    return int(m.group(1)) if m else None


def _fincaraiz_lookup_whatsapp(url: str) -> str | None:
    """Resolve the broker's WhatsApp number via the public lead-gen GraphQL.

    Returns the first 10-digit CO mobile from the response (preferring the
    one starting with ``57 3...``, i.e. a mobile not a landline). Returns
    ``None`` on any failure — network error, missing property id, malformed
    response, or a response that only contains landlines.
    """
    import urllib.error
    import urllib.request

    property_id = _fincaraiz_property_id(url)
    if property_id is None:
        return None

    payload = json.dumps([{
        "operationName": "request_wpp_mutation",
        "variables": {
            "property_id": property_id,
            "type": "PROPERTY",
            **_FR_LEAD_DUMMY,
        },
        "query": _FR_LEAD_QUERY,
    }]).encode("utf-8")

    req = urllib.request.Request(
        _FR_LEAD_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-origin": "www.fincaraiz.com.co",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.warning("FR lead lookup failed for %s: %s", url, e)
        return None

    # Response is a batched GraphQL list of {data: {requestPhoneAgent: [...]}}.
    if not isinstance(body, list) or not body:
        return None
    phones_raw = (body[0].get("data") or {}).get("requestPhoneAgent") or []
    phones: list[str] = [
        p.get("phone") for p in phones_raw
        if isinstance(p, dict) and isinstance(p.get("phone"), str)
    ]
    # Prefer a CO mobile (57 + 3xxxxxxxxx); fall back to whatever's there.
    for p in phones:
        digits = re.sub(r"\D", "", p)
        if digits.startswith("57") and len(digits) == 12 and digits[2] == "3":
            return digits[2:]
    return re.sub(r"\D", "", phones[0]) if phones else None


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


# ---------------------------------------------------------------------------
# Search-results parsers (shallow — produce {id, url, price} dicts only).
# ---------------------------------------------------------------------------


def _shallow_fincaraiz(card, base_url: str) -> dict | None:
    href = card.css("a.lc-data::attr(href)").get() or card.css("a::attr(href)").get()
    if not href:
        return None
    url = urljoin(base_url, href)
    price = _parse_cop_price(card.css(".lc-price .main-price::text").get())
    return {"id": f"fincaraiz:{_slug_from_url(url)}", "url": url, "price": price}


def _shallow_metrocuadrado(card, base_url: str) -> dict | None:
    href = card.attrib.get("href") if hasattr(card, "attrib") else None
    if not href:
        href = card.css("::attr(href)").get()
    if not href:
        return None
    url = urljoin(base_url, href)
    price = _parse_cop_price(card.css(".property-card__detail-price::text").get())
    return {"id": f"metrocuadrado:{_slug_from_url(url)}", "url": url, "price": price}


_SEARCH_ADAPTERS: list[dict[str, Any]] = [
    {
        "name": "fincaraiz",
        "url_template": "https://www.fincaraiz.com.co/{transaction}/{slug}/{location}",
        "slug_field": 0,  # index into _PROPERTY_TYPE_MAP value tuple
        "card_selector": ".listingCard",
        "shallow_parser": _shallow_fincaraiz,
    },
    {
        "name": "metrocuadrado",
        "url_template": "https://www.metrocuadrado.com/{slug}/{transaction}/{location}/",
        "slug_field": 1,
        "card_selector": "a[href*='/inmueble/']",
        "shallow_parser": _shallow_metrocuadrado,
    },
]


def _discover_one(adapter: dict[str, Any], location: str, property_type: str, transaction: str) -> list[dict]:
    name = adapter["name"]
    slug = _PROPERTY_TYPE_MAP[property_type][adapter["slug_field"]]
    url = adapter["url_template"].format(slug=slug, transaction=transaction, location=location)
    logger.info("[%s] discovering: %s", name, url)

    try:
        page = _fetch_page(url)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] fetch failed: %s", name, e)
        return []

    status = getattr(page, "status", None)
    if isinstance(status, int) and status >= 400:
        logger.warning("[%s] HTTP %s — skipping", name, status)
        return []

    try:
        cards = page.css(adapter["card_selector"])
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] selector %r failed: %s", name, adapter["card_selector"], e)
        return []
    logger.info("[%s] %d card(s) matched", name, len(cards))

    parser = adapter["shallow_parser"]
    out: list[dict] = []
    seen_urls: set[str] = set()
    for card in cards:
        if len(out) >= MAX_LISTINGS_PER_SOURCE:
            break
        rec = _safe(parser, card, url)
        if not rec or rec["url"] in seen_urls:
            continue
        seen_urls.add(rec["url"])
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Detail-page parsers (deep — produce a full Listing).
# ---------------------------------------------------------------------------


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


def _parse_fincaraiz_detail(page, url: str) -> Listing | None:
    html = _page_html(page)

    # Finca Raíz is an Angular SPA — class names are hashed and rarely stable.
    # The reliable surface is the rendered "Detalles de la Propiedad" panel,
    # which lays each fact out as ``Label  Value``; we extract by label.
    page_text = _joined_text(page, "body *::text") or html
    # Restrict numeric extraction to the canonical details table — the page
    # summary uses Spanish order (``3 Baños``, ``4 Habs.``) which would
    # otherwise let the regex pick up neighbouring digits like ``Baños 265 m²``.
    details_marker = re.search(r"Detalles de la Propiedad", page_text, re.IGNORECASE)
    details_text = page_text[details_marker.end():] if details_marker else page_text

    title = _first_text(page, "h1::text")
    price = _parse_cop_price(
        _first_text(page, ".price-wrapper .price::text")
        or _first_text(page, "[class*='price'] *::text")
    )
    area = (
        _find_labeled_area(details_text, r"(?:área|area)\s+construida")
        or _find_labeled_area(details_text, r"(?:área|area)\s+privada")
        or _find_labeled_area(details_text, r"(?:área|area)")
    )
    bedrooms = _find_labeled_int(details_text, r"habitaciones?|alcobas?")
    bathrooms = _find_labeled_int(details_text, r"baños?")
    # "Ubicación Principal <neighborhood>, <city>, <state>" — capture the line.
    zone = None
    zm = re.search(
        r"Ubicaci[oó]n\s+Principal\s+([^•\n]+?)(?:\s+Ubicaciones\s+asociadas|\s+Destacado|\s+Favorito|$)",
        page_text,
        re.IGNORECASE,
    )
    if zm:
        zone = zm.group(1).strip(" ,") or None
    description = _joined_text(
        page, "section[data-testid='description'] *::text"
    ) or _joined_text(page, "[class*='description'] *::text")

    estrato = _find_labeled_int(details_text, r"estrato")
    parking = _find_labeled_int(details_text, r"parqueader[oa]s?|garaj[ea]s?")

    coordinates = _extract_coordinates(html)
    contact_links = _extract_contact_links(page)
    phone_numbers: list[str] = []

    # Finca Raíz scrubs the broker phone before render but signals existence
    # via `"has_whatsapp": true` in __NEXT_DATA__. Skip the lead-API call when
    # the flag is false to avoid generating spurious leads on the broker side.
    has_whatsapp = bool(re.search(r'"has_whatsapp"\s*:\s*true', html))
    if has_whatsapp:
        phone_raw = _safe(_fincaraiz_lookup_whatsapp, url)
        if phone_raw:
            link = _format_whatsapp_link(phone_raw)
            if link and link not in contact_links:
                contact_links.append(link)
            if phone_raw not in phone_numbers:
                phone_numbers.append(phone_raw)

    prop_type, trans_type = _infer_types_from_url(url)
    raw_payload: dict[str, Any] = {}
    if title:
        raw_payload["title"] = title
    if has_whatsapp:
        raw_payload["has_whatsapp"] = True

    return Listing(
        id=f"fincaraiz:{_slug_from_url(url)}",
        source_site="fincaraiz",
        url=url,
        price=price,
        area_m2=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        zone=zone,
        phone_numbers=phone_numbers,
        property_type=prop_type,
        transaction_type=trans_type,
        estrato=estrato,
        parking_lots=parking,
        contact_links=contact_links,
        coordinates=coordinates,
        description=description,
        raw_payload=raw_payload,
    )


def _parse_metrocuadrado_detail(page, url: str) -> Listing | None:
    html = _page_html(page)

    # Metro Cuadrado is a Next.js app — most of the structured data lives in
    # __NEXT_DATA__. We pull what's easy from JSON and fall back to DOM.
    next_data: dict[str, Any] = {}
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if m:
        try:
            next_data = json.loads(m.group(1))
        except json.JSONDecodeError:
            next_data = {}

    def _dig(d: Any, *keys: str) -> Any:
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    propsroot = _dig(next_data, "props", "pageProps") or {}
    propdata = (
        _dig(propsroot, "propertyDetail")
        or _dig(propsroot, "property")
        or _dig(propsroot, "data")
        or {}
    )

    title = _first_text(page, "h1::text") or (propdata.get("title") if isinstance(propdata, dict) else None)
    price = _parse_cop_price(_first_text(page, ".property-card__detail-price::text"))
    if price is None and isinstance(propdata, dict):
        for k in ("salePrice", "rentPrice", "price"):
            v = propdata.get(k)
            if isinstance(v, (int, float)) and v > 0:
                price = float(v)
                break

    area = None
    if isinstance(propdata, dict):
        for k in ("area", "areaPrivada", "areaConstruida"):
            v = propdata.get(k)
            if isinstance(v, (int, float)) and v > 0:
                area = float(v)
                break

    bedrooms = None
    bathrooms = None
    if isinstance(propdata, dict):
        bedrooms = propdata.get("rooms") or propdata.get("bedrooms")
        bathrooms = propdata.get("bathrooms") or propdata.get("baths")
        bedrooms = int(bedrooms) if isinstance(bedrooms, (int, float, str)) and str(bedrooms).isdigit() else None
        bathrooms = int(bathrooms) if isinstance(bathrooms, (int, float, str)) and str(bathrooms).isdigit() else None

    zone = None
    if isinstance(propdata, dict):
        zone = (
            propdata.get("neighborhood")
            or propdata.get("zone")
            or propdata.get("sector")
            or propdata.get("location")
        )
        if isinstance(zone, dict):
            zone = zone.get("name") or zone.get("label")
    if not zone:
        zone = _first_text(page, ".property-card__detail-top__left div::text")
        if zone:
            zone = zone.split("|")[0].strip()

    description = None
    if isinstance(propdata, dict):
        description = propdata.get("description") or propdata.get("comment")
    if not description:
        description = _joined_text(page, ".detail-description *::text") or _joined_text(
            page, "[class*='description'] *::text"
        )

    estrato = None
    parking = None
    if isinstance(propdata, dict):
        for k in ("estrato", "stratum"):
            v = propdata.get(k)
            if isinstance(v, (int, float)):
                estrato = int(v)
                break
            if isinstance(v, str) and v.isdigit():
                estrato = int(v)
                break
        for k in ("garages", "parkingLots", "parkingSpaces", "parqueaderos"):
            v = propdata.get(k)
            if isinstance(v, (int, float)):
                parking = int(v)
                break
            if isinstance(v, str) and v.isdigit():
                parking = int(v)
                break

    page_text = _joined_text(page, "body *::text") or html
    if estrato is None:
        estrato = _find_labeled_int(page_text, r"estrato")
    if parking is None:
        parking = _find_labeled_int(page_text, r"parqueader[oa]s?|garaj[ea]s?")

    coordinates = None
    if isinstance(propdata, dict):
        lat = propdata.get("latitude") or _dig(propdata, "coordinates", "lat")
        lon = (
            propdata.get("longitude")
            or _dig(propdata, "coordinates", "lon")
            or _dig(propdata, "coordinates", "lng")
        )
        try:
            if lat is not None and lon is not None:
                coordinates = {"lat": float(lat), "lon": float(lon)}
        except (TypeError, ValueError):
            coordinates = None
    if coordinates is None:
        coordinates = _extract_coordinates(html)

    contact_links = _extract_contact_links(page)
    phone_numbers: list[str] = []

    # Metro Cuadrado streams its detail data as escaped JSON inside
    # <script>self.__next_f.push(...)</script> chunks. The broker phone is
    # therefore reachable with plain regex on the raw HTML — no JSON parse
    # needed (and the chunks aren't valid JSON in isolation anyway).
    # ``whatsappBot`` is the portal's own automated bot, not the broker; we
    # ignore it on purpose so verification messages reach a human.
    wa_m = re.search(r'\\"whatsapp\\"\s*:\s*\\"(\d{7,15})\\"', html)
    phone_m = re.search(r'\\"contactPhone\\"\s*:\s*\\"(\d{7,15})\\"', html)
    msg_m = re.search(r'\\"whatsappMessage\\"\s*:\s*\\"((?:[^\\"]|\\.)*?)\\"', html)
    phone_raw = (wa_m.group(1) if wa_m else None) or (phone_m.group(1) if phone_m else None)
    wa_message = None
    if msg_m:
        # The captured chunk is a JSON-string body without surrounding quotes;
        # let json.loads handle escapes correctly (and without mangling UTF-8
        # the way codecs.unicode_escape does).
        try:
            wa_message = json.loads('"' + msg_m.group(1) + '"')
        except json.JSONDecodeError:
            wa_message = msg_m.group(1)
    if phone_raw:
        link = _format_whatsapp_link(phone_raw, wa_message)
        if link and link not in contact_links:
            contact_links.append(link)
        if phone_raw not in phone_numbers:
            phone_numbers.append(phone_raw)

    prop_type, trans_type = _infer_types_from_url(url)

    return Listing(
        id=f"metrocuadrado:{_slug_from_url(url)}",
        source_site="metrocuadrado",
        url=url,
        price=price,
        area_m2=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        zone=zone,
        phone_numbers=phone_numbers,
        property_type=prop_type,
        transaction_type=trans_type,
        estrato=estrato,
        parking_lots=parking,
        contact_links=contact_links,
        coordinates=coordinates,
        description=description,
        raw_payload={"title": title} if title else {},
    )


# ---------------------------------------------------------------------------
# Public tools.
# ---------------------------------------------------------------------------


@tool
def search_listings(
    location: str = "medellin",
    property_type: str = "apartamentos",
    transaction: str = "arriendo",
) -> list[dict]:
    """Discover candidate property URLs across Finca Raíz and Metro Cuadrado.

    Returns a lightweight list of ``{"id", "url", "price"}`` dicts — enough to
    deduplicate and shortlist before paying the deep-scrape cost.

    Args:
        location: City slug, e.g. ``"medellin"``, ``"bogota"``.
        property_type: One of ``"apartamentos"``, ``"casas"``, ``"locales"``,
            ``"oficinas"``, ``"fincas"``. Unknown values fall back to apartamentos.
        transaction: ``"arriendo"`` (rent) or ``"venta"`` (sale).
    """
    if property_type not in _PROPERTY_TYPE_MAP:
        logger.warning("unknown property_type %r — falling back to 'apartamentos'", property_type)
        property_type = "apartamentos"
    if transaction not in _TRANSACTION_MAP:
        logger.warning("unknown transaction %r — falling back to 'arriendo'", transaction)
        transaction = "arriendo"

    results: list[dict] = []
    for adapter in _SEARCH_ADAPTERS:
        results.extend(_discover_one(adapter, location, property_type, transaction))
    logger.info("discovered %d total listing stub(s) across %d source(s)",
                len(results), len(_SEARCH_ADAPTERS))
    return results


@tool
def extract_property_details(url: str) -> Listing | None:
    """Deep-scrape a single property page and return a validated Listing.

    Routes by URL host: Finca Raíz vs Metro Cuadrado. Returns ``None`` if the
    page can't be fetched, the URL is from an unsupported host, or validation
    fails. Individual missing fields surface as ``None`` on the Listing, not
    as exceptions.

    Args:
        url: A property detail URL produced by :func:`search_listings`.
    """
    if not url:
        logger.warning("extract_property_details called with empty url")
        return None

    host = urlparse(url).netloc.lower()
    if "fincaraiz.com.co" in host:
        parser: Callable[[Any, str], Listing | None] = _parse_fincaraiz_detail
        source = "fincaraiz"
    elif "metrocuadrado.com" in host:
        parser = _parse_metrocuadrado_detail
        source = "metrocuadrado"
    else:
        logger.warning("unsupported host %r — cannot route deep scrape", host)
        return None

    logger.info("[%s] enriching: %s", source, url)
    try:
        page = _fetch_page(url)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] fetch failed: %s", source, e)
        return None

    status = getattr(page, "status", None)
    if isinstance(status, int) and status >= 400:
        logger.warning("[%s] HTTP %s — aborting", source, status)
        return None

    try:
        listing = parser(page, url)
    except ValidationError as ve:
        logger.warning("[%s] Listing validation failed: %s", source, ve)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] deep-scrape parser raised: %s", source, e)
        return None

    if listing is None:
        logger.warning("[%s] parser returned None", source)
    return listing


# ---------------------------------------------------------------------------
# End-to-end demo: discover → enrich.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    print("=== Tool 1: search_listings ===", file=sys.stderr)
    hits: list[dict] = search_listings.invoke({})  # all defaults
    print(json.dumps(hits[:5], indent=2, ensure_ascii=False))

    if not hits:
        print("no listings discovered — aborting enrichment demo", file=sys.stderr)
        sys.exit(1)

    target = hits[0]
    print(f"\n=== Tool 2: extract_property_details({target['url']!r}) ===", file=sys.stderr)
    listing = extract_property_details.invoke({"url": target["url"]})
    if listing is None:
        print("deep scrape failed", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(json.loads(listing.model_dump_json()), indent=2, ensure_ascii=False))
