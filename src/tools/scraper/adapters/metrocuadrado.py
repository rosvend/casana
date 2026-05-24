"""Metro Cuadrado adapter — strictly Metro Cuadrado scraping logic.

Metro Cuadrado retired its ``__NEXT_DATA__`` blob; the detail object now ships
as escaped JSON inside ``self.__next_f.push([1,"..."])`` RSC-stream chunks.
:func:`_mc_property_data` isolates and decodes that object; DOM and page-text
fallbacks cover whatever the stream omits.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin

from src.state.listings import Listing
from src.tools.scraper.adapters.base import PortalAdapter
from src.tools.scraper.core import (
    _extract_contact_links,
    _extract_coordinates,
    _find_labeled_int,
    _first_text,
    _format_whatsapp_link,
    _infer_types_from_url,
    _joined_text,
    _page_html,
    _parse_cop_price,
    _slug_from_url,
    _zone_slug,
)


def _mc_property_data(html: str) -> dict[str, Any]:
    """Extract Metro Cuadrado's main property object from its RSC stream.

    MC retired the ``__NEXT_DATA__`` blob; the detail object now ships as
    escaped JSON inside a ``self.__next_f.push([1,"..."])`` chunk, anchored by
    the ``"data":{"propertyId":...}`` key. We isolate that object's leading
    slice, decode the escaped quotes, and pull each scalar field with plain
    regex — returning a flat dict keyed by MC's own field names. Returns ``{}``
    when the anchor is absent so the caller falls back to the DOM cleanly.
    """
    idx = html.find('\\"data\\":{\\"propertyId\\"')
    if idx == -1:
        return {}
    # 8 KB of raw (escaped) chars comfortably spans every scalar field; the
    # full object runs much longer (the free-text description sits ~17 KB in).
    blob = html[idx:idx + 8000].replace('\\"', '"')

    def _num(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"?(-?\d+(?:\.\d+)?)"?', blob)
        return m.group(1) if m else None

    def _txt(key: str) -> str | None:
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', blob)
        if not m:
            return None
        try:
            return json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            return m.group(1)

    data: dict[str, Any] = {}
    # Prices, areas and coordinates → float.
    for key in ("salePrice", "rentPrice", "area", "areac", "lat", "lon"):
        raw = _num(key)
        if raw is not None:
            data[key] = float(raw)
    # Room/bath/parking counts ship as quoted strings ("3") → int.
    for key in ("rooms", "bathrooms", "garages"):
        raw = _num(key)
        if raw is not None:
            try:
                data[key] = int(float(raw))
            except ValueError:
                pass
    for key in ("neighborhood", "commonNeighborhood", "comment"):
        val = _txt(key)
        if val:
            data[key] = val
    return data


class MetroCuadradoAdapter(PortalAdapter):
    """Strategy implementation for ``metrocuadrado.com``."""

    name = "metrocuadrado"
    slug_field = 1  # index into _PROPERTY_TYPE_MAP value tuple
    card_selector = "a[href*='/inmueble/']"
    hosts = ("metrocuadrado.com",)

    def build_search_url(
        self,
        slug: str,
        transaction: str,
        location: str,
        filters: dict[str, int],
        zone: str | None = None,
        page: int = 1,
    ) -> str | None:
        """Metro Cuadrado packs all filters into a single hyphen-joined slug
        segment and requires the ``?search=form`` suffix to parse it. Prices are
        expressed in *millions* (integer), not raw COP. Ranges work for price
        and area; bedrooms/bath/estrato/parking only accept a single value.

        Zone slot: live-verified to sit *after* the city, e.g.
        ``/apartamento/arriendo/bogota/chapinero/``.

        Pagination: live recon confirmed MC has no URL-routable pagination
        (results render via a JS-driven ``rc-pagination`` widget that hits
        ``/rest-search/search?from=N`` via XHR; query params like ``?p=2``,
        ``?page=2`` and ``?pagina=2`` all silently render page 1). The MC
        page already returns ~50 cards on first load — well above our
        per-page cap — so a single fetch is enough. Return ``None`` for
        ``page > 1`` to signal "no further pages reachable" and let the
        orchestrator stop iterating.
        """
        if page > 1:
            return None
        z = _zone_slug(zone)
        if z:
            base = f"https://www.metrocuadrado.com/{slug}/{transaction}/{location}/{z}/"
        else:
            base = f"https://www.metrocuadrado.com/{slug}/{transaction}/{location}/"
        tokens: list[str] = []

        min_p = filters.get("min_price")
        max_p = filters.get("max_price")
        min_p_m = int(round(min_p / 1_000_000)) if min_p is not None else None
        max_p_m = int(round(max_p / 1_000_000)) if max_p is not None else None
        if min_p_m is not None and max_p_m is not None and min_p_m < max_p_m:
            tokens.append(f"{min_p_m}-{max_p_m}-millones")
        elif max_p_m is not None:
            tokens.append(f"{max_p_m}-millones")
        # min-only price has no clean MC slug — post-filter catches it.

        if (v := filters.get("bedrooms")) is not None:
            tokens.append(f"{v}-habitaciones")
        if (v := filters.get("bathrooms")) is not None:
            tokens.append(f"{v}-banos")
        if (v := filters.get("estrato")) is not None:
            tokens.append(f"estrato-{v}")
        if (v := filters.get("parking_lots")) is not None:
            tokens.append(f"{v}-parqueaderos")

        min_a = filters.get("min_area_m2")
        max_a = filters.get("max_area_m2")
        if min_a is not None and max_a is not None and min_a < max_a:
            tokens.append(f"{min_a}-{max_a}-m2")
        # min-only / max-only area: post-filter only.

        if tokens:
            return base + "-".join(tokens) + "/?search=form"
        return base

    def parse_card(self, card: Any, base_url: str) -> dict | None:
        href = card.attrib.get("href") if hasattr(card, "attrib") else None
        if not href:
            href = card.css("::attr(href)").get()
        if not href:
            return None
        url = urljoin(base_url, href)
        price = _parse_cop_price(card.css(".property-card__detail-price::text").get())
        return {"id": f"metrocuadrado:{_slug_from_url(url)}", "url": url, "price": price}

    def parse_detail(self, page: Any, url: str) -> Listing | None:
        html = _page_html(page)

        # Metro Cuadrado retired __NEXT_DATA__; the detail object now streams as
        # escaped JSON inside a self.__next_f.push(...) chunk. _mc_property_data
        # isolates and decodes it; DOM/page-text fallbacks below cover the rest.
        propdata = _mc_property_data(html)

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
        lat, lon = propdata.get("lat"), propdata.get("lon")
        if lat is not None and lon is not None:
            try:
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
