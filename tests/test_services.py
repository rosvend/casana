from types import SimpleNamespace

from estatia.models import Listing, ListingLocation, ListingProperty, Location, PropertyType, UserRequest
from estatia.services import TavilyNewsService, normalize_text


def test_normalize_text_strips_accents_and_spacing():
    assert normalize_text("  Bogotá Norte  ") == "bogota norte"


def test_tavily_neighborhood_helpers_use_request_and_listings():
    request = UserRequest(
        raw_text="need options",
        search_summary="broad request",
        location=Location(city="Bogota"),
    )
    listings = [
        Listing(
            id="med-001",
            source="test",
            url="https://example.com/med-001",
            title="Laureles apartment",
            price=3200000,
            location=ListingLocation(city="Medellin", neighborhood="Laureles"),
            property=ListingProperty(type=PropertyType.APARTMENT, bedrooms=2, bathrooms=2),
        )
    ]
    service = TavilyNewsService(settings=SimpleNamespace(tavily_api_key=None, news_results_limit=5))

    neighborhoods = service._candidate_neighborhoods(request, listings)
    query = service._build_query(request, neighborhoods)

    assert "Laureles" in neighborhoods
    assert "Bogota" in query
    assert "neighborhood news" in query
