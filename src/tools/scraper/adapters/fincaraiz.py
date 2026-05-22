"""Finca Raíz adapter — strictly Finca Raíz scraping logic.

Finca Raíz is an Angular SPA: class names are hashed and unstable, so the
detail parser leans on the rendered "Detalles de la Propiedad" panel and
label-anchored regex. The broker phone is scrubbed from the HTML and recovered
through a public GraphQL lead-gen mutation (see :func:`_fincaraiz_lookup_whatsapp`).
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urljoin

from src.state.listings import Listing
from src.tools.scraper.adapters.base import PortalAdapter
from src.tools.scraper.core import (
    _extract_contact_links,
    _extract_coordinates,
    _find_labeled_area,
    _find_labeled_int,
    _first_text,
    _format_whatsapp_link,
    _infer_types_from_url,
    _joined_text,
    _page_html,
    _parse_cop_price,
    _safe,
    _slug_from_url,
    logger,
)

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


class FincaRaizAdapter(PortalAdapter):
    """Strategy implementation for ``fincaraiz.com.co``."""

    name = "fincaraiz"
    slug_field = 0  # index into _PROPERTY_TYPE_MAP value tuple
    card_selector = ".listingCard"
    hosts = ("fincaraiz.com.co",)

    def build_search_url(
        self, slug: str, transaction: str, location: str, filters: dict[str, int]
    ) -> str:
        """Finca Raíz uses additive path slugs, one per segment. Order is stable
        (price → rooms → bath → estrato → area → parking) so URLs are
        deterministic and reproducible."""
        base = f"https://www.fincaraiz.com.co/{transaction}/{slug}/{location}"
        parts: list[str] = []
        if (v := filters.get("min_price")) is not None:
            parts.append(f"desde-{v}")
        if (v := filters.get("max_price")) is not None:
            parts.append(f"hasta-{v}")
        if (v := filters.get("bedrooms")) is not None:
            parts.append(f"{v}-habitaciones")
        if (v := filters.get("bathrooms")) is not None:
            parts.append(quote(f"{v}-baños", safe="-"))
        if (v := filters.get("estrato")) is not None:
            parts.append(f"estrato-{v}")
        if (v := filters.get("min_area_m2")) is not None:
            parts.append(f"desde-area-{v}")
        if (v := filters.get("max_area_m2")) is not None:
            parts.append(f"hasta-area-{v}")
        if (v := filters.get("parking_lots")) is not None:
            parts.append(f"{v}-parqueaderos")
        # longevity has no stable FR slug — handled by the post-filter.
        if parts:
            return base + "/" + "/".join(parts)
        return base

    def parse_card(self, card: Any, base_url: str) -> dict | None:
        href = card.css("a.lc-data::attr(href)").get() or card.css("a::attr(href)").get()
        if not href:
            return None
        url = urljoin(base_url, href)
        price = _parse_cop_price(card.css(".lc-price .main-price::text").get())
        return {"id": f"fincaraiz:{_slug_from_url(url)}", "url": url, "price": price}

    def parse_detail(self, page: Any, url: str) -> Listing | None:
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
