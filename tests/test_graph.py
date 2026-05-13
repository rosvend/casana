from estatia.config import Settings
from estatia.graph import build_graph
from estatia.models import (
    Budget,
    EvalResult,
    Listing,
    ListingLocation,
    ListingProperty,
    Location,
    Recommendation,
    NewsInsight,
    SellerReport,
    PropertyType,
    UserRequest,
)
from estatia.services import Services, StandbyWhatsAppService


class StaticListingService:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings

    def search(self, request: UserRequest) -> list[Listing]:
        if request.location.neighborhood == "Atlantis":
            return []
        return self.listings


class StaticNewsService:
    def __init__(self, insights: list[NewsInsight]) -> None:
        self.insights = insights

    def search(self, request: UserRequest, listings: list[Listing]) -> list[NewsInsight]:
        return self.insights


class DummyWorkflow:
    def parse_request(self, raw_text: str) -> UserRequest:
        return UserRequest(
            raw_text=raw_text,
            search_summary="Need an apartment in Bogota",
            location=Location(city="Bogota"),
            budget=Budget(max=3000000),
        )

    def chill_request(self, request: UserRequest, feedback: str) -> UserRequest:
        return request.model_copy(update={"location": Location(city="Bogota", neighborhood="Teusaquillo")})

    def evaluate(self, request, listings, news, threshold):
        return EvalResult(
            score=0.85 if listings else 0.4,
            threshold=threshold,
            passed=bool(listings),
            reasons=["Budget fit"],
            required_fixes=[] if listings else ["Need more options"],
        )

    def build_report(self, request, listings, news, evaluation, language):
        top = listings[0]
        return SellerReport(
            title="Top match",
            summary="A concise match.",
            recommendations=[
                Recommendation(
                    listing_id=top.id,
                    title=top.title,
                    neighborhood=top.location.neighborhood,
                    price=top.price,
                    currency=top.currency,
                    why_it_fits=["Within budget"],
                    tradeoffs=["Limited inventory"],
                )
            ],
            budget_fit=["Stays under max budget."],
            market_notes=["Inventory is thin but usable."],
            next_steps=["Book a visit."],
            language="en",
        )


def test_graph_reaches_seller_node_with_seed_services():
    workflow = DummyWorkflow()
    listings = [
        Listing(
            id="bog-apt-001",
            source="test",
            url="https://example.com/listings/bog-apt-001",
            title="Modern apartment near Parque 93",
            price=2800000,
            location=ListingLocation(city="Bogota", neighborhood="Teusaquillo"),
            property=ListingProperty(type=PropertyType.APARTMENT, bedrooms=2, bathrooms=2),
        )
    ]
    services = Services(
        intake=workflow,
        evaluation=workflow,
        seller=workflow,
        listing=StaticListingService(listings),
        news=StaticNewsService([]),
        whatsapp=StandbyWhatsAppService(),
    )
    settings = Settings(openai_api_key="test", max_retries=1)

    graph = build_graph(services, settings)
    state = graph.invoke({"raw_text": "find me a place", "retries": 0, "trace": []})

    assert "html" in state
    assert state["evaluation"].passed is True


def test_graph_no_results_feedback_mentions_constraint_relaxation():
    class NoResultsWorkflow:
        def parse_request(self, raw_text: str) -> UserRequest:
            return UserRequest(
                raw_text=raw_text,
                search_summary="Strict budget in exact zone",
                location=Location(city="Bogota", neighborhood="Atlantis"),
                budget=Budget(max=3000000),
            )

        def chill_request(self, request: UserRequest, feedback: str) -> UserRequest:
            return request

        def evaluate(self, request, listings, news, threshold):
            return EvalResult(
                score=0.0,
                threshold=threshold,
                passed=False,
                reasons=["No results"],
                required_fixes=["Raise budget"],
            )

        def build_report(self, request, listings, news, evaluation, language):
            return SellerReport(
                title="unused",
                summary="unused",
                recommendations=[],
                budget_fit=[],
                market_notes=[],
                next_steps=[],
            )

    workflow = NoResultsWorkflow()
    services = Services(
        intake=workflow,
        evaluation=workflow,
        seller=workflow,
        listing=StaticListingService([]),
        news=StaticNewsService([]),
        whatsapp=StandbyWhatsAppService(),
    )
    settings = Settings(openai_api_key="test", max_retries=0)

    graph = build_graph(services, settings)
    state = graph.invoke({"raw_text": "find me a place", "retries": 0, "trace": []})

    feedback = state["feedback"].lower()
    assert "increase budget.max" in feedback
    assert "widen or narrow the search area" in feedback
    assert "relax property type" in feedback
