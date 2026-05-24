"""`news_node` — the area-news LangGraph node.

This node fetches neighborhood/zone context so the downstream synthesizer and
evaluator can score soft constraints like "security". It runs in parallel with
the properties/whatsapp branch and converges at the synthesizer.

The work is three stages:

1. Resolve the target place from the structured brief — a ``zone`` constraint
   when present, otherwise just the ``location``.
2. Issue one Spanish, year-2026 web search per :data:`NewsCategory` (safety,
   transport, infrastructure, events, market trends) through a swappable
   :class:`~src.tools.search.SearchProvider`. Searches run concurrently.
3. A single ``gpt-4o-mini`` structured-output call classifies and synthesizes
   the raw search snippets into :class:`NewsItem` objects grouped by category.

The search backend is decoupled behind :func:`src.tools.search.get_search_provider`,
so DuckDuckGo can later be swapped for SearXNG / Tavily / Crawl4AI without
touching this node. When search yields nothing the node returns empty category
lists — it must not let the LLM hallucinate news.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import NewsItem, NewsResults, PropertyFinderState, StructuredRequirements
from src.tools.search import get_search_provider

load_dotenv()

logger = logging.getLogger(__name__)
MODEL = "gpt-5-nano"

#: Year stamped into every query so search engines surface fresh context.
SEARCH_YEAR = 2026
#: Hits requested per category query.
MAX_RESULTS_PER_QUERY = 5

#: NewsCategory -> Spanish search-query template. ``{place}`` is filled per run.
#: Keys are exactly the NewsCategory literals so dict assembly stays type-safe.
_CATEGORY_QUERIES: dict[str, str] = {
    "crime_safety": f"noticias de seguridad y delincuencia en {{place}} {SEARCH_YEAR}",
    "transportation": f"noticias de movilidad y transporte en {{place}} {SEARCH_YEAR}",
    "infrastructure": f"noticias de desarrollo urbano e infraestructura en {{place}} {SEARCH_YEAR}",
    "events": f"eventos y noticias locales en {{place}} {SEARCH_YEAR}",
    "market_trends": f"mercado inmobiliario y precios de vivienda en {{place}} {SEARCH_YEAR}",
}


class NewsExtraction(BaseModel):
    """The schema the LLM fills in for one area-news synthesis turn.

    One ``list[NewsItem]`` per :data:`NewsCategory`. An empty list is valid and
    expected when the raw snippets carry nothing relevant for that category —
    this is the structured-output stand-in for the ``NewsResults`` dict alias,
    which ``with_structured_output`` cannot target directly.
    """

    crime_safety: list[NewsItem] = Field(
        default_factory=list, description="Items about crime, safety, policing."
    )
    transportation: list[NewsItem] = Field(
        default_factory=list, description="Items about mobility, transit, roads."
    )
    infrastructure: list[NewsItem] = Field(
        default_factory=list,
        description="Items about urban development, construction, public works.",
    )
    events: list[NewsItem] = Field(
        default_factory=list, description="Items about local events and general area news."
    )
    market_trends: list[NewsItem] = Field(
        default_factory=list,
        description="Items about real-estate prices, demand, and new developments.",
    )


SYSTEM_PROMPT = """\
Eres un analista de noticias locales para un buscador inmobiliario en Colombia.
Recibes resultados crudos de búsqueda web (título, url, fragmento) agrupados por
categoría. Tu trabajo es CLASIFICAR y SINTETIZAR esos resultados en objetos
NewsItem, sin inventar nada.

REGLAS:
- Usa EXCLUSIVAMENTE la información de los fragmentos provistos. Si para una
  categoría no hay fragmentos relevantes, devuelve una lista VACÍA. NUNCA
  inventes noticias, titulares ni URLs.
- Descarta fragmentos irrelevantes (publicidad, páginas genéricas, contenido
  que no sea noticia del lugar consultado).
- title: el titular tal como aparece en el fragmento.
- summary: un párrafo en español que sintetice el hallazgo y su sentimiento
  (¿mejora o empeora la zona para alguien que busca vivienda?).
- url y source: cópialos del fragmento; si no hay, déjalos en null.
- published_at: déjalo SIEMPRE en null; no inventes fechas.
- zone: usa la zona indicada en el contexto del usuario.
- Categorías: crime_safety (seguridad y delincuencia), transportation
  (movilidad y transporte), infrastructure (desarrollo urbano e
  infraestructura), events (eventos y noticias locales), market_trends
  (mercado inmobiliario: precios, demanda, nuevos proyectos).
"""


def _resolve_place(
    requirements: StructuredRequirements | None,
) -> tuple[str | None, str | None]:
    """Return ``(location, zone)`` extracted from the constraints list.

    Location and zone are not direct fields on :class:`StructuredRequirements`;
    they are :class:`~src.state.Constraint` objects with ``field == "location"``
    / ``field == "zone"``. The first match of each wins.
    """
    if requirements is None:
        return None, None
    location = zone = None
    for c in requirements.constraints:
        if c.field == "location" and c.exact_value is not None and location is None:
            location = str(c.exact_value)
        elif c.field == "zone" and c.exact_value is not None and zone is None:
            zone = str(c.exact_value)
    return location, zone


def _run_searches(place: str) -> dict[str, list]:
    """Run the per-category queries concurrently; return ``category -> hits``.

    Each ``provider.search`` call is an independent blocking network request,
    so they are fanned across a thread pool — mirroring ``search_listings`` in
    the scraper. ``executor.map`` preserves order, keeping the result keys
    aligned with the category list.
    """
    provider = get_search_provider()
    categories = list(_CATEGORY_QUERIES.keys())

    def _one(category: str) -> list:
        query = _CATEGORY_QUERIES[category].format(place=place)
        logger.info("news_node: searching [%s] %r", category, query)
        return provider.search(query, max_results=MAX_RESULTS_PER_QUERY)

    with ThreadPoolExecutor(max_workers=len(categories)) as executor:
        hits = list(executor.map(_one, categories))
    return dict(zip(categories, hits))


def _format_snippets(per_category: dict[str, list], zone: str | None) -> str:
    """Render raw search hits into the human message for the LLM."""
    lines: list[str] = []
    if zone:
        lines.append(f"Zona objetivo del usuario: {zone}\n")
    for category, hits in per_category.items():
        lines.append(f"### Categoría: {category} ({len(hits)} resultado(s))")
        if not hits:
            lines.append("(sin resultados)")
        for hit in hits:
            lines.append(f"- título: {hit.title}")
            lines.append(f"  url: {hit.url}")
            lines.append(f"  fragmento: {hit.snippet}")
        lines.append("")
    return "\n".join(lines)


def _empty_results() -> NewsResults:
    """An all-empty NewsResults — one empty list per NewsCategory."""
    return {
        "crime_safety": [],
        "transportation": [],
        "infrastructure": [],
        "events": [],
        "market_trends": [],
    }


def news_node(state: PropertyFinderState) -> dict:
    """Fetch and categorize area news for the requested location/zone.

    Reads ``state["requirements"]`` to resolve the target place, web-searches
    each :data:`NewsCategory`, and returns ``{"news_results": NewsResults}``.
    Returns all-empty category lists — never hallucinated news — when no
    location is known or search yields nothing.
    """
    # News is invariant under softening — the zone/location don't change, only
    # the property constraints (price, size, bedrooms). Re-running would burn
    # Tavily credits for identical queries, so skip if we've already produced
    # a NewsResults dict in a prior superstep.
    if state.get("news_results") is not None:
        logger.info("news_node: news_results already populated, skipping refetch")
        return {}

    requirements = state.get("requirements")
    location, zone = _resolve_place(requirements)

    if not location:
        logger.warning("news_node: no location constraint — returning empty news_results")
        return {"news_results": _empty_results()}

    # Prefer "zone, location" for precision; fall back to the location alone.
    place = f"{zone}, {location}" if zone else location
    logger.info("news_node: researching %r", place)

    per_category = _run_searches(place)
    total_hits = sum(len(hits) for hits in per_category.values())
    if total_hits == 0:
        logger.warning("news_node: search yielded 0 results for %r", place)
        return {"news_results": _empty_results()}

    llm = ChatOpenAI(model=MODEL, temperature=0)
    # function_calling (not strict json_schema) — NewsExtraction nests NewsItem
    # with an optional datetime, which strict mode handles poorly.
    structured_llm = llm.with_structured_output(NewsExtraction, method="function_calling")
    messages: list[tuple[str, str]] = [
        ("system", SYSTEM_PROMPT),
        ("human", _format_snippets(per_category, zone)),
    ]
    extraction: NewsExtraction = structured_llm.invoke(messages)

    news_results: NewsResults = {
        "crime_safety": extraction.crime_safety,
        "transportation": extraction.transportation,
        "infrastructure": extraction.infrastructure,
        "events": extraction.events,
        "market_trends": extraction.market_trends,
    }

    # Stamp the zone onto every item so the synthesizer can join on it.
    join_key = zone or location
    for items in news_results.values():
        for item in items:
            if item.zone is None:
                item.zone = join_key

    logger.info(
        "news_node: synthesized %d item(s) across %d categories from %d search hit(s)",
        sum(len(v) for v in news_results.values()), len(news_results), total_hits,
    )
    return {"news_results": news_results}
