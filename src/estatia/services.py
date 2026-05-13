from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from openai import OpenAI
from pydantic import BaseModel

from estatia.config import Settings
from estatia.listing_sources import PlaywrightListingClient
from estatia.models import EvalResult, Listing, NewsInsight, SellerReport, UserRequest

logger = logging.getLogger("estatia.services")
ModelT = TypeVar("ModelT", bound=BaseModel)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").strip().lower()


class IntakeService(Protocol):
    def parse_request(self, raw_text: str) -> UserRequest: ...

    def chill_request(self, request: UserRequest, feedback: str) -> UserRequest: ...


class EvaluationService(Protocol):
    def evaluate(
        self,
        request: UserRequest,
        listings: list[Listing],
        news: list[NewsInsight],
        threshold: float,
    ) -> EvalResult: ...


class SellerService(Protocol):
    def build_report(
        self,
        request: UserRequest,
        listings: list[Listing],
        news: list[NewsInsight],
        evaluation: EvalResult,
        language: str,
    ) -> SellerReport: ...


class ListingService(Protocol):
    def search(self, request: UserRequest) -> list[Listing]: ...


class NewsService(Protocol):
    def search(self, request: UserRequest, listings: list[Listing]) -> list[NewsInsight]: ...


class WhatsAppService(Protocol):
    def validate(self, listings: list[Listing]) -> list[str]: ...


class OpenAIWorkflowService(IntakeService, EvaluationService, SellerService):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.llm_provider
        if self.provider == "nvidia":
            if not settings.nvidia_api_key:
                raise ValueError("NVIDIA_API_KEY is required when ESTATIA_LLM_PROVIDER=nvidia.")
            self.client = OpenAI(
                api_key=settings.nvidia_api_key,
                base_url=settings.nvidia_base_url or "https://integrate.api.nvidia.com/v1",
            )
        else:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required to run the workflow.")
            self.client = OpenAI(api_key=settings.openai_api_key)

    def _parse_structured(
        self,
        model: str,
        schema: type[ModelT],
        system_prompt: str,
        user_content: str,
    ) -> ModelT:
        if self.provider == "nvidia":
            return self._parse_with_chat_json(model, schema, system_prompt, user_content)
        response = self.client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            text_format=schema,
        )
        return response.output_parsed

    def _parse_with_chat_json(
        self,
        model: str,
        schema: type[ModelT],
        system_prompt: str,
        user_content: str,
    ) -> ModelT:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n"
                        "Return only a valid JSON object. Do not use markdown fences.\n"
                        f"JSON schema:\n{schema_json}"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        content = completion.choices[0].message.content or ""
        payload = self._extract_json_object(content)
        return schema.model_validate_json(payload)

    def _extract_json_object(self, content: str) -> str:
        candidate = content.strip()
        if candidate.startswith("```"):
            parts = [part for part in candidate.split("```") if part.strip()]
            if parts:
                candidate = parts[-1].strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Model response did not contain a valid JSON object.")
        return candidate[start : end + 1]

    def parse_request(self, raw_text: str) -> UserRequest:
        logger.info("%s parse_request:start", self.provider)
        parsed = self._parse_structured(
            model=self.settings.fast_model,
            schema=UserRequest,
            system_prompt=(
                "Extract a real-estate search request into the provided schema. "
                "Prefer explicit values from the user. Use null when unknown. "
                "Keep the summary short and factual."
            ),
            user_content=raw_text,
        )
        logger.info("%s parse_request:done", self.provider)
        return parsed

    def chill_request(self, request: UserRequest, feedback: str) -> UserRequest:
        logger.info("%s chill_request:start", self.provider)
        parsed = self._parse_structured(
            model=self.settings.fast_model,
            schema=UserRequest,
            system_prompt=(
                "Relax a real-estate search request so it becomes searchable. "
                "Do not invent new priorities. Preserve the user intent. "
                "You may relax any blocking constraint: budget, neighborhood scope, property type, "
                "bedroom/bathroom/area targets, and strict must-have filters. "
                "If the requested budget is too low for the target area, raise the budget ceiling "
                "to the nearest viable market range and mark the budget as flexible. "
                "If the area is too narrow, widen it to nearby neighborhoods. "
                "If the area is too broad and noisy, narrow it to the most promising zone. "
                "If constraints are too strict, move secondary preferences into nice_to_have. "
                "Keep the same city unless the failure feedback clearly says the city itself has no matches."
            ),
            user_content=(
                f"Original request:\n{request.model_dump_json(indent=2)}\n\n"
                f"Why it failed:\n{feedback}\n\n"
                "Important rules:\n"
                "- Relax the smallest number of constraints needed to make the search viable.\n"
                "- Budget can be raised.\n"
                "- Area can be widened or narrowed.\n"
                "- Room count, area, property type, and must-have filters can be relaxed.\n"
                "- Do not invent new preferences that were never implied by the user."
            ),
        )
        logger.info("%s chill_request:done", self.provider)
        return parsed

    def evaluate(
        self,
        request: UserRequest,
        listings: list[Listing],
        news: list[NewsInsight],
        threshold: float,
    ) -> EvalResult:
        logger.info("%s evaluate:start listings=%s news=%s", self.provider, len(listings), len(news))
        listing_blob = [listing.model_dump(mode="json") for listing in listings]
        news_blob = [item.model_dump(mode="json") for item in news]
        result = self._parse_structured(
            model=self.settings.quality_model,
            schema=EvalResult,
            system_prompt=(
                "Evaluate whether the candidate properties fit the request. "
                "Be strict about budget, location fit, and must-have constraints."
            ),
            user_content=(
                f"Threshold: {threshold}\n"
                f"Request:\n{request.model_dump_json(indent=2)}\n\n"
                f"Listings:\n{listing_blob}\n\n"
                f"News:\n{news_blob}"
            ),
        )
        logger.info("%s evaluate:done score=%.2f", self.provider, result.score)
        return result.model_copy(update={"threshold": threshold, "passed": result.score >= threshold})

    def build_report(
        self,
        request: UserRequest,
        listings: list[Listing],
        news: list[NewsInsight],
        evaluation: EvalResult,
        language: str,
    ) -> SellerReport:
        logger.info("%s build_report:start listings=%s", self.provider, len(listings))
        language_name = "Spanish" if language == "es" else "English"
        parsed = self._parse_structured(
            model=self.settings.quality_model,
            schema=SellerReport,
            system_prompt=(
                "Prepare a concise sales report for shortlisted properties. "
                "Be specific, practical, and grounded in the provided data. "
                f"Write the report in {language_name}."
            ),
            user_content=(
                f"Request:\n{request.model_dump_json(indent=2)}\n\n"
                f"Listings:\n{[item.model_dump(mode='json') for item in listings]}\n\n"
                f"News:\n{[item.model_dump(mode='json') for item in news]}\n\n"
                f"Evaluation:\n{evaluation.model_dump_json(indent=2)}"
            ),
        )
        logger.info("%s build_report:done", self.provider)
        return parsed.model_copy(update={"language": language})


class PlaywrightListingService(ListingService):
    def __init__(self, settings: Settings, fallback: ListingService | None = None) -> None:
        self.client = PlaywrightListingClient(settings)
        self.fallback = fallback

    def search(self, request: UserRequest) -> list[Listing]:
        logger.info("Playwright listing search:start")
        listings = self.client.search(request)
        if listings:
            logger.info("Playwright listing search:done listings=%s", len(listings))
            return listings
        if self.fallback is not None:
            logger.warning("Playwright listing search returned no listings, using configured fallback service")
            return self.fallback.search(request)
        logger.warning("Playwright listing search returned no listings")
        return []


CITY_NEIGHBORHOODS: dict[str, list[str]] = {
    "bogota": ["Chico Norte", "Cedritos", "Teusaquillo", "Rosales", "Chapinero", "Usaquen"],
    "medellin": ["Laureles", "El Poblado", "Envigado", "Sabaneta", "Belen", "Los Colores"],
    "cali": ["Ciudad Jardin", "Granada", "San Fernando", "El Ingenio", "Pance"],
}


class TavilyNewsService(NewsService):
    def __init__(self, settings: Settings, fallback: NewsService | None = None) -> None:
        self.settings = settings
        self.api_key = settings.tavily_api_key
        self.fallback = fallback

    def search(self, request: UserRequest, listings: list[Listing]) -> list[NewsInsight]:
        if not self.api_key:
            logger.warning("Tavily news search skipped because TAVILY_API_KEY is not configured")
            return self._fallback(request, listings)

        neighborhoods = self._candidate_neighborhoods(request, listings)
        query = self._build_query(request, neighborhoods)
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "time_range": "month",
            "max_results": self.settings.news_results_limit,
            "include_answer": False,
            "include_raw_content": False,
        }
        logger.info(
            "Tavily news search:start city=%s neighborhoods=%s",
            request.location.city,
            neighborhoods,
        )
        try:
            response = self._post_search(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Tavily news search failed: %s", exc)
            return self._fallback(request, listings)

        insights = self._build_insights(response, neighborhoods, request)
        logger.info("Tavily news search:done insights=%s", len(insights))
        if insights:
            return insights
        return self._fallback(request, listings)

    def _post_search(self, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = UrlRequest(
            "https://api.tavily.com/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def _candidate_neighborhoods(
        self,
        request: UserRequest,
        listings: list[Listing],
    ) -> list[str]:
        if request.location.neighborhood:
            return [request.location.neighborhood]

        candidates: list[str] = []
        for area in request.location.alternate_areas:
            if area and area not in candidates:
                candidates.append(area)

        for listing in listings:
            neighborhood = listing.location.neighborhood
            if neighborhood and neighborhood not in candidates:
                candidates.append(neighborhood)

        city_key = normalize_text(request.location.city)
        for area in CITY_NEIGHBORHOODS.get(city_key, []):
            if area not in candidates:
                candidates.append(area)

        return candidates[:6]

    def _build_query(self, request: UserRequest, neighborhoods: list[str]) -> str:
        city = request.location.city or "the city"
        intent = "rental" if request.intent.value == "rent" else "property"
        focus = ", ".join(neighborhoods[:4]) if neighborhoods else city
        return (
            f"{city} Colombia neighborhood news {focus} "
            f"safety transport development walkability demand {intent}"
        )

    def _build_insights(
        self,
        response: dict[str, object],
        neighborhoods: list[str],
        request: UserRequest,
    ) -> list[NewsInsight]:
        results = response.get("results", [])
        if not isinstance(results, list):
            return []

        insights: list[NewsInsight] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            summary = str(item.get("content") or item.get("snippet") or "").strip()
            source = str(item.get("source") or item.get("domain") or "Tavily").strip()
            if not title or not url or not summary:
                continue
            neighborhood = self._detect_neighborhood(title, summary, neighborhoods, request)
            if not neighborhood:
                continue
            try:
                insights.append(
                    NewsInsight(
                        neighborhood=neighborhood,
                        title=title,
                        summary=summary[:420],
                        source=source,
                        url=url,
                    )
                )
            except Exception:
                continue
        return insights[: self.settings.news_results_limit]

    def _detect_neighborhood(
        self,
        title: str,
        summary: str,
        neighborhoods: list[str],
        request: UserRequest,
    ) -> str | None:
        haystack = normalize_text(f"{title} {summary}")
        for neighborhood in neighborhoods:
            if normalize_text(neighborhood) and normalize_text(neighborhood) in haystack:
                return neighborhood
        if request.location.neighborhood:
            return request.location.neighborhood
        return request.location.city

    def _fallback(self, request: UserRequest, listings: list[Listing]) -> list[NewsInsight]:
        if self.fallback is None:
            return []
        logger.info("Tavily news search:fallback")
        return self.fallback.search(request, listings)


class StandbyWhatsAppService(WhatsAppService):
    def validate(self, listings: list[Listing]) -> list[str]:
        return [f"{listing.id}: standby validation disabled" for listing in listings]


@dataclass(slots=True)
class Services:
    intake: IntakeService
    evaluation: EvaluationService
    seller: SellerService
    listing: ListingService
    news: NewsService
    whatsapp: WhatsAppService
