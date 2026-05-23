"""Area news fetched by `news_agent`, grouped by sub-query category.

`news_agent` issues several sub-queries per zone (safety, transport,
infrastructure, events, real-estate market) and returns them already
categorized so the synthesizer can attach the right slice to each
candidate's zone.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

NewsCategory = Literal[
    "crime_safety", "transportation", "infrastructure", "events", "market_trends"
]
"""Closed set of news sub-queries `news_agent` issues — crime/safety, transport,
urban infrastructure, local events, and real-estate market trends (prices,
demand, new developments). Keeping this a Literal means routing logic and dict
keys stay in sync via the type checker."""


class NewsItem(BaseModel):
    """A single news article or post relevant to a candidate's area.

    Written by: `news_agent`.
    Read by: `synthesizer_agent` (attached to candidates), evaluator (only
    indirectly, via candidate context).
    """

    title: str = Field(..., description="Headline as published.")
    summary: str = Field(..., description="One-paragraph distillation for the synthesizer.")
    url: str | None = Field(default=None, description="Canonical source URL if available.")
    published_at: datetime | None = Field(
        default=None, description="Publication timestamp; used to filter stale items."
    )
    source: str | None = Field(default=None, description="Outlet or domain.")
    zone: str | None = Field(
        default=None,
        description="Zone this item is associated with; the synthesizer joins on this field.",
    )


NewsResults = dict[NewsCategory, list[NewsItem]]
"""Category-keyed news bucket stored on the state. A plain dict alias rather
than a Pydantic model because LangGraph TypedDict-field updates work most
naturally with raw mappings."""
