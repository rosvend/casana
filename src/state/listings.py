"""Property listings — raw, verified, and finalized as candidates.

The pipeline produces three list shapes, in order:
    `Listing` (scraped) → `VerifiedListing` (WhatsApp-confirmed) →
    `Candidate` (synthesizer's merged + news-enriched view).
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.state.news import NewsCategory, NewsItem


class Listing(BaseModel):
    """A raw property listing as scraped from a portal.

    Written by: `properties_agent` (from Finca Raíz, Metro Cuadrado, etc.).
    Read by: `whatsapp_agent` (uses `phone_numbers` to verify availability)
    and the synthesizer.
    """

    id: str = Field(..., description="Stable identifier — typically '{source_site}:{listing_id}'.")
    source_site: str = Field(..., description="Portal slug, e.g. 'fincaraiz', 'metrocuadrado'.")
    url: str = Field(..., description="Canonical listing URL.")
    price: float | None = Field(default=None, description="Asking price in COP.")
    area_m2: float | None = Field(default=None, description="Built area in square meters.")
    bedrooms: int | None = Field(default=None, description="Number of bedrooms.")
    bathrooms: int | None = Field(default=None, description="Number of bathrooms.")
    zone: str | None = Field(
        default=None,
        description="Neighborhood / locality string used to join against news by zone.",
    )
    phone_numbers: list[str] = Field(
        default_factory=list,
        description="Contact numbers extracted from the listing; consumed by whatsapp_agent.",
    )
    property_type: str | None = Field(
        default=None,
        description="Property kind in canonical English, e.g. 'apartment', 'house'.",
    )
    transaction_type: str | None = Field(
        default=None,
        description="Transaction kind in canonical English, e.g. 'rent', 'sale'.",
    )
    estrato: int | None = Field(
        default=None,
        description="Colombian socio-economic stratum (1-6) when published by the portal.",
    )
    parking_lots: int | None = Field(
        default=None,
        description="Parking spaces included with the unit.",
    )
    contact_links: list[str] = Field(
        default_factory=list,
        description="Direct contact URLs found on the listing page (api.whatsapp.com / wa.me).",
    )
    coordinates: dict[str, float] | None = Field(
        default=None,
        description="Map coordinates as {'lat': float, 'lon': float} when extractable.",
    )
    description: str | None = Field(
        default=None,
        description="Full free-text description scraped from the detail page.",
    )
    raw_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific extra fields preserved verbatim for traceability.",
    )


class VerifiedListing(Listing):
    """A `Listing` after `whatsapp_agent` has attempted to confirm availability.

    Inherits every field of `Listing` so callers can treat it polymorphically.
    Written by: `whatsapp_agent`. Read by: `synthesizer_agent`.
    """

    availability_confirmed: bool = Field(
        ...,
        description="True if the contact party confirmed the unit is still available.",
    )
    verification_timestamp: datetime = Field(
        ..., description="When the verification was completed."
    )
    verification_notes: str | None = Field(
        default=None,
        description="Free-text observations from the WhatsApp exchange (no reply, agent unsure, etc.).",
    )


class Candidate(BaseModel):
    """A synthesized, news-enriched property the evaluator will score.

    Written by: `synthesizer_agent` (merges + dedupes verified listings and
    attaches the news slice matching the listing's zone).
    Read by: `evaluator_agent`, `done_node`, `best_effort_node`.
    """

    listing: VerifiedListing = Field(..., description="The underlying verified listing.")
    relevant_news: dict[NewsCategory, list[NewsItem]] = Field(
        default_factory=dict,
        description="News items already filtered to this candidate's zone, grouped by category.",
    )
    match_notes: str | None = Field(
        default=None,
        description="Synthesizer's annotation about why this candidate was included or merged.",
    )
    match_score: float = Field(
        default=0.0,
        description=(
            "Final score in [0.0, 1.0] computed by evaluator_agent by combining "
            "per-Constraint importance with StructuredRequirements.priority_weights. "
            "0.0 means unscored or total mismatch; 1.0 means perfect match."
        ),
    )
