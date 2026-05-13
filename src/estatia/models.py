from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    RENT = "rent"
    BUY = "buy"


class PropertyType(str, Enum):
    ANY = "any"
    APARTMENT = "apartment"
    HOUSE = "house"
    STUDIO = "studio"
    LOFT = "loft"
    OFFICE = "office"
    LAND = "land"


class Budget(BaseModel):
    min: float | None = None
    max: float | None = None
    currency: str = "COP"
    flexible: bool = False

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: str | None) -> str:
        return (value or "COP").upper()


class Location(BaseModel):
    city: str | None = None
    neighborhood: str | None = None
    alternate_areas: list[str] = Field(default_factory=list)
    radius_km: float | None = None


class PropertyPreferences(BaseModel):
    type: PropertyType = PropertyType.ANY
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_min_m2: float | None = None
    area_max_m2: float | None = None


class Constraints(BaseModel):
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)


class UserRequest(BaseModel):
    raw_text: str
    search_summary: str | None = None
    intent: Intent = Intent.RENT
    location: Location = Field(default_factory=Location)
    budget: Budget = Field(default_factory=Budget)
    property: PropertyPreferences = Field(default_factory=PropertyPreferences)
    constraints: Constraints = Field(default_factory=Constraints)


class ListingLocation(BaseModel):
    city: str
    neighborhood: str | None = None
    address: str | None = None


class ListingProperty(BaseModel):
    type: PropertyType = PropertyType.ANY
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_m2: float | None = None


class Listing(BaseModel):
    id: str
    source: str
    url: str
    title: str
    price: float
    currency: str = "COP"
    location: ListingLocation
    property: ListingProperty = Field(default_factory=ListingProperty)
    highlights: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    score: float = 0.0

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: str | None) -> str:
        return (value or "COP").upper()


class NewsInsight(BaseModel):
    neighborhood: str
    title: str
    summary: str
    source: str
    url: str


class TraceEvent(BaseModel):
    node: str
    message: str


class EvalResult(BaseModel):
    score: float
    threshold: float
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    required_fixes: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    listing_id: str
    title: str
    neighborhood: str | None = None
    price: float
    currency: str = "COP"
    why_it_fits: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: str | None) -> str:
        return (value or "COP").upper()


class SellerReport(BaseModel):
    title: str
    summary: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    budget_fit: list[str] = Field(default_factory=list)
    market_notes: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    language: str = "en"


# Legacy aliases kept for compatibility with older tests and notebooks.
class Requirement(BaseModel):
    location: str
    price: int
    area: float
    bedrooms: int
    parking_spaces: int
    admin_fee: int
    bathrooms: int
    property_type: str


class Property(BaseModel):
    location: str
    price: int
    area: float
    bedrooms: int
    parking_spaces: int
    admin_fee: int
    bathrooms: int
    property_type: str
    score: float


class NewsItem(BaseModel):
    source: str
    text: str
    summary: str


class Proposal(BaseModel):
    properties: list[Property]
    score: float


class AgentState(BaseModel):
    raw_text: str | None = None
    user_text: str | None = None
    properties: list[Property] | None = None
    requirements: list[Requirement] | None = None
    news_items: list[NewsItem] | None = None
    proposals: list[Proposal] | None = None
    evaluation: EvalResult | None = None
    html: str = ""
    run_news: bool = False
    retries: int = 0
    feedback: str = ""
from pydantic import BaseModel
class EvalResult(BaseModel):
    score: float
    threshold: float
    passed: bool
    reasons: list[str] = []
    required_fixes: list[str] = []
from pydantic import BaseModel
from estatia.models import Requirement
class RequirementList(BaseModel):
    requirements: list[Requirement]
