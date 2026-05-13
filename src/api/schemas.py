from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    raw_text: str = Field(..., min_length=1, max_length=4000)
    language: Literal["en", "es"] = "en"


class RunResponse(BaseModel):
    final_results: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    error: str | None = None


class TraceEvent(BaseModel):
    type: Literal["node_start", "node_end", "log", "error", "done"]
    node: str | None = None
    payload: dict[str, Any] | None = None
    ts: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    graph_ready: bool
