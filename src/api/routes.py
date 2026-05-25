"""HTTP endpoints for the chat-based Estatia API.

- ``GET /health`` — liveness + graph readiness flag.
- ``POST /chat`` — fresh user turn. Wraps ``graph.invoke(...)`` in a
  threadpool (graph nodes call blocking I/O — Scrapling, Playwright,
  SQLite). Returns the final state, or the pending interrupt payload
  when the graph paused for clarification.
- ``POST /resume`` — same shape, but invokes ``Command(resume=...)`` to
  continue a paused thread.
- ``GET /history/{thread_id}`` — replays the conversation from the
  checkpointer for the given thread.
- ``POST /chat/stream`` — SSE variant of ``/chat`` that emits ``status``
  events before each node runs so the user sees what the agent is doing.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.api.deps import get_graph
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    InterruptPayload,
    MessageDict,
    ResumeRequest,
)
from src.api.streaming import message_to_dict, stream_chat

logger = logging.getLogger("estatia.api.routes")

router = APIRouter()


def _thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _build_chat_response(graph: Any, thread_id: str) -> ChatResponse:
    """Snapshot the thread's checkpoint and serialize it for the client."""
    snapshot = graph.get_state(_thread_config(thread_id))
    values = snapshot.values if snapshot else {}

    pending_interrupts = list(snapshot.interrupts) if snapshot else []
    interrupt_obj = None
    if pending_interrupts:
        raw = pending_interrupts[0].value or {}
        question = (
            raw.get("clarification_question")
            if isinstance(raw, dict)
            else str(raw)
        )
        if question:
            interrupt_obj = InterruptPayload(clarification_question=question)

    messages = [MessageDict(**message_to_dict(m)) for m in (values.get("messages") or [])]

    evaluation = values.get("evaluation")
    evaluation_dump = None
    if evaluation is not None:
        dump = getattr(evaluation, "model_dump", None)
        evaluation_dump = dump(mode="json") if callable(dump) else dict(evaluation)

    final_results_raw = values.get("final_results") or []
    final_results: list[dict[str, Any]] = []
    for candidate in final_results_raw:
        dump = getattr(candidate, "model_dump", None)
        final_results.append(dump(mode="json") if callable(dump) else dict(candidate))

    return ChatResponse(
        thread_id=thread_id,
        messages=messages,
        interrupt=interrupt_obj,
        final_results=final_results,
        evaluation=evaluation_dump,
        is_best_effort=bool(values.get("is_best_effort")),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        get_graph()
        ready = True
    except Exception:
        logger.exception("graph build failed")
        ready = False
    return HealthResponse(graph_ready=ready)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, graph: Any = Depends(get_graph)) -> ChatResponse:
    config = _thread_config(req.thread_id)
    try:
        await run_in_threadpool(
            graph.invoke,
            {"messages": [HumanMessage(content=req.user_message)]},
            config,
        )
    except Exception as exc:
        logger.exception("graph.invoke failed for thread=%s", req.thread_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _build_chat_response(graph, req.thread_id)


@router.post("/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest, graph: Any = Depends(get_graph)) -> ChatResponse:
    config = _thread_config(req.thread_id)
    try:
        await run_in_threadpool(
            graph.invoke,
            Command(resume=req.resume_payload),
            config,
        )
    except Exception as exc:
        logger.exception("graph.invoke (resume) failed for thread=%s", req.thread_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _build_chat_response(graph, req.thread_id)


@router.get("/history/{thread_id}", response_model=HistoryResponse)
async def history(thread_id: str, graph: Any = Depends(get_graph)) -> HistoryResponse:
    snapshot = graph.get_state(_thread_config(thread_id))
    if snapshot is None:
        return HistoryResponse(thread_id=thread_id, messages=[])
    values = snapshot.values or {}
    messages = [MessageDict(**message_to_dict(m)) for m in (values.get("messages") or [])]
    return HistoryResponse(thread_id=thread_id, messages=messages)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, graph: Any = Depends(get_graph)) -> StreamingResponse:
    config = _thread_config(req.thread_id)
    invocation_input = {"messages": [HumanMessage(content=req.user_message)]}
    return StreamingResponse(
        stream_chat(graph, invocation_input, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
