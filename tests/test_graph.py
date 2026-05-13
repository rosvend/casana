from estatia.config import Settings
from estatia.graph import build_graph
from estatia.models import (
    EvalResult,
    Requirement,
    Property,
    NewsItem,
    Proposal,
)
from estatia.services import Services, StandbyWhatsAppService


class StaticListingService:
    def __init__(self, listings: list[Property]) -> None:
        self.listings = listings

    def search(self, request: Requirement) -> list[Property]:
        if request.location == "Atlantis":
            return []
        return self.listings


class StaticNewsService:
    def __init__(self, insights: list[NewsItem]) -> None:
        self.insights = insights

    def search(self, request: Requirement, listings: list[Property]) -> list[NewsItem]:
        return self.insights


class DummyWorkflow:
    def parse_request(self, raw_text: str) -> list[Requirement]:
        return [
            Requirement(
                location="Bogota",
                price=3000000,
                area=70.0,
                bedrooms=2,
                parking_spaces=1,
                admin_fee=200000,
                bathrooms=1,
                property_type="apartment"
            )
        ]

    def chill_request(self, request: Requirement, feedback: str) -> Requirement:
        return request.model_copy(update={"location": "Teusaquillo"})

    def evaluate(self, request, listings, news, threshold):
        return EvalResult(
            score=0.85 if listings else 0.4,
            threshold=threshold,
            passed=bool(listings),
            reasons=["Budget fit"],
            required_fixes=[] if listings else ["Need more options"],
        )

    def build_report(self, request, listings, news, evaluation, language):
        return Proposal(
            properties=listings,
            score=0.85
        )


def test_graph_reaches_seller_node_with_seed_services():
    workflow = DummyWorkflow()
    listings = [
        Property(
            location="Teusaquillo",
            price=2800000,
            area=65.0,
            bedrooms=2,
            parking_spaces=1,
            admin_fee=150000,
            bathrooms=2,
            property_type="apartment",
            score=0.9
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
    state = graph.invoke(
        {
            "user_text": "find me a place",
            "retries": 0,
        }
    )

    assert "feedback" in state
    assert state["feedback"] == "" # Because it passed evaluation


def test_graph_no_results_feedback_mentions_constraint_relaxation():
    class NoResultsWorkflow:
        def parse_request(self, raw_text: str) -> list[Requirement]:
            return [
                Requirement(
                    location="Atlantis",
                    price=3000000,
                    area=70.0,
                    bedrooms=2,
                    parking_spaces=1,
                    admin_fee=200000,
                    bathrooms=1,
                    property_type="apartment"
                )
            ]

        def chill_request(self, request: Requirement, feedback: str) -> Requirement:
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
            return Proposal(
                properties=[],
                score=0.0
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
    state = graph.invoke(
        {
            "user_text": "find me a place",
            "retries": 0,
        }
    )

    feedback = state["feedback"].lower()
    assert "no properties matched" in feedback