"""Pydantic request/response schemas for the chat-based Estatia API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    user_message: str = Field(..., min_length=1, max_length=4000)


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    resume_payload: str = Field(..., min_length=1, max_length=4000)


class InterruptPayload(BaseModel):
    clarification_question: str


class MessageDict(BaseModel):
    type: Literal["human", "ai", "system", "tool"]
    content: str
    id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    messages: list[MessageDict]
    interrupt: InterruptPayload | None = None
    final_results: list[dict[str, Any]] = Field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    is_best_effort: bool = False


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[MessageDict]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    graph_ready: bool


class StatusEvent(BaseModel):
    """Schema for SSE `status` events emitted while a node is about to run."""

    node: str
    message: str
