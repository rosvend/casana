import concurrent.futures
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
from estatia.models import EvalResult, Requirement, Property, NewsItem, Proposal, RequirementList

logger = logging.getLogger("estatia.services")
ModelT = TypeVar("ModelT", bound=BaseModel)


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").strip().lower()


class IntakeService(Protocol):
    def parse_request(self, raw_text: str) -> list[Requirement]: ...

    def chill_request(self, requirement: Requirement, feedback: str) -> Requirement: ...


class EvaluationService(Protocol):
    def evaluate(
        self,
        request: Requirement | None,
        listings: list[Property],
        news: list[NewsItem],
        threshold: float,
    ) -> EvalResult:
        ...


class SellerService(Protocol):
    def build_report(
        self,
        request: Requirement | None,
        listings: list[Property],
        news: list[NewsItem],
        evaluation: EvalResult,
        language: str,
    ) -> Proposal:
        ...


class ListingService(Protocol):
    def search(self, request: Requirement) -> list[Property]: ...


class NewsService(Protocol):
    def search(self, request: Requirement, listings: list[Property]) -> list[NewsItem]: ...


class WhatsAppService(Protocol):
    def validate(self, listings: list[Property]) -> list[str]: ...


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

    def parse_request(self, raw_text: str) -> list[Requirement]:
        logger.info("%s parse_request:start", self.provider)
        parsed = self._parse_structured(
            model=self.settings.fast_model,
            schema=RequirementList,
            system_prompt=(
                "Extract real-estate search requirements into the provided schema. "
                "Prefer explicit values from the user."
            ),
            user_content=raw_text,
        )
        logger.info("%s parse_request:done", self.provider)
        return parsed.requirements

    def chill_request(self, requirement: Requirement, feedback: str) -> Requirement:
        logger.info("%s chill_request:start", self.provider)
        parsed = self._parse_structured(
            model=self.settings.fast_model,
            schema=Requirement,
            system_prompt=(
                "Relax a real-estate search requirement so it becomes searchable. "
                "Do not invent new priorities. Preserve the user intent."
            ),
            user_content=(
                f"Original request:\n{requirement.model_dump_json(indent=2)}\n\n"
                f"Why it failed:\n{feedback}\n\n"
                "Important rules:\n"
                "- Relax the smallest number of constraints needed to make the search viable."
            ),
        )
        logger.info("%s chill_request:done", self.provider)
        return parsed

    def evaluate(
        self,
        request: Requirement | None,
        listings: list[Property],
        news: list[NewsItem],
        threshold: float,
    ) -> EvalResult:
        logger.info("%s evaluate:start listings=%s news=%s", self.provider, len(listings), len(news))
        listing_blob = [listing.model_dump(mode="json") for listing in listings]
        news_blob = [item.model_dump(mode="json") for item in news]
        request_blob = request.model_dump_json(indent=2) if request else "None"
        result = self._parse_structured(
            model=self.settings.quality_model,
            schema=EvalResult,
            system_prompt=(
                "Evaluate whether the candidate properties fit the request."
            ),
            user_content=(
                f"Threshold: {threshold}\n"
                f"Request:\n{request_blob}\n\n"
                f"Listings:\n{listing_blob}\n\n"
                f"News:\n{news_blob}"
            ),
        )
        logger.info("%s evaluate:done score=%.2f", self.provider, result.score)
        return result.model_copy(update={"threshold": threshold, "passed": result.score >= threshold})

    def build_report(
        self,
        request: Requirement | None,
        listings: list[Property],
        news: list[NewsItem],
        evaluation: EvalResult,
        language: str,
    ) -> Proposal:
        logger.info("%s build_report:start listings=%s", self.provider, len(listings))
        language_name = "Spanish" if language == "es" else "English"
        request_blob = request.model_dump_json(indent=2) if request else "None"
        parsed = self._parse_structured(
            model=self.settings.quality_model,
            schema=Proposal,
            system_prompt=(
                "Prepare a concise sales proposal for shortlisted properties. "
                "Be specific, practical, and grounded in the provided data. "
                f"Write the report in {language_name}."
            ),
            user_content=(
                f"Request:\n{request_blob}\n\n"
                f"Listings:\n{[item.model_dump(mode='json') for item in listings]}\n\n"
                f"News:\n{[item.model_dump(mode='json') for item in news]}\n\n"
                f"Evaluation:\n{evaluation.model_dump_json(indent=2)}"
            ),
        )
        logger.info("%s build_report:done", self.provider)
        return parsed


class DummyListingService(ListingService):
    def search(self, request: Requirement) -> list[Property]:
        return []


class DummyNewsService(NewsService):
    def search(self, request: Requirement, listings: list[Property]) -> list[NewsItem]:
        return []


class StandbyWhatsAppService(WhatsAppService):
    def validate(self, listings: list[Property]) -> list[str]:
        return [f"{listing.location}: standby validation disabled" for listing in listings]


@dataclass(slots=True)
class Services:
    intake: IntakeService
    evaluation: EvaluationService
    seller: SellerService
    listing: ListingService
    news: NewsService
    whatsapp: WhatsAppService
class TavilyNewsService(NewsService):
    def __init__(self, settings: Settings, fallback: NewsService | None = None) -> None:
        self.settings = settings
        self.api_key = settings.tavily_api_key
        self.fallback = fallback
        self.topics = [
            "movilidad transporte publico trafico",
            "seguridad crimen policia",
            "comercializacion comercio tiendas",
            "vida nocturna bares restaurantes",
            "riesgos ambientales inundaciones clima"
        ]

    def search(self, request: Requirement, listings: list[Property]) -> list[NewsItem]:
        if not self.api_key:
            logger.warning("Tavily skipped (no API key). Using fallback.")
            return self._fallback(request, listings)

        locations = self._candidate_locations(request, listings)
        focus = ", ".join(locations) if locations else "Colombia"
        
        queries = [f"{focus} {topic}" for topic in self.topics]
        
        logger.info("Tavily composite search:start locations=%s topics=%s", locations, len(queries))
        
        all_news_items = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
            future_to_query = {executor.submit(self._fetch_single_query, query): query for query in queries}
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    items = future.result()
                    all_news_items.extend(items)
                except Exception as exc:
                    logger.warning("Tavily sub-query '%s' failed: %s", query, exc)
                    
        logger.info("Tavily composite search:done total_insights=%s", len(all_news_items))
        
        # Deduplicate by title to avoid noise
        seen_titles = set()
        unique_items = []
        for item in all_news_items:
            if item.text not in seen_titles:
                unique_items.append(item)
                seen_titles.add(item.text)
                
        if unique_items:
            return unique_items[:self.settings.news_results_limit * len(self.topics)]
            
        return self._fallback(request, listings)

    def _fetch_single_query(self, query: str) -> list[NewsItem]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": "news",
            "search_depth": "basic",
            "time_range": "month",
            "max_results": max(1, self.settings.news_results_limit),
            "include_answer": False,
        }
        response = self._post_search(payload)
        return self._build_insights(response)

    def _post_search(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = UrlRequest(
            "https://api.tavily.com/search",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def _candidate_locations(self, request: Requirement, listings: list[Property]) -> list[str]:
        candidates = []
        if request.location and request.location not in candidates:
            candidates.append(request.location)
            
        for listing in listings:
            if listing.location and listing.location not in candidates:
                candidates.append(listing.location)
                
        return candidates[:3] 

    def _build_insights(self, response: dict) -> list[NewsItem]:
        results = response.get("results", [])
        news_items = []
        for item in results:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("content") or item.get("snippet") or "").strip()
            source = str(item.get("source") or item.get("domain") or "Tavily").strip()
            
            if not title or not summary:
                continue
                
            news_items.append(
                NewsItem(
                    source=source,
                    text=title,
                    summary=summary[:420] 
                )
            )
        return news_items

    def _fallback(self, request: Requirement, listings: list[Property]) -> list[NewsItem]:
        if self.fallback is None:
            return []
        return self.fallback.search(request, listings)
