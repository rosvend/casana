from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import HealthResponse, RunRequest, RunResponse
from src.api.streaming import stream_workflow

logging.basicConfig(level=os.getenv("ESTATIA_LOG_LEVEL", "INFO"))
logger = logging.getLogger("estatia.api.app")

app = FastAPI(title="Estatia API", version="0.1.0")

_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ESTATIA_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _load_graph() -> Any | None:
    """Lazy-load the compiled graph. Returns None when the agent modules
    on main are still empty stubs so the API can still serve /health.
    """
    try:
        from src.graph.graph import build_graph
        return build_graph()
    except Exception as exc:
        logger.warning("build_graph() unavailable: %s", exc)
        return None


def _build_initial_state(req: RunRequest) -> dict[str, Any]:
    return {
        "raw_text": req.raw_text,
        "language": req.language,
        "retries": 0,
        "trace": [],
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(graph_ready=_load_graph() is not None)


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    logger.info("POST /run language=%s text_len=%d", req.language, len(req.raw_text))
    graph = _load_graph()
    if graph is None:
        return RunResponse(error="Graph not ready: agent modules on main are not implemented yet.")
    try:
        state = await run_in_threadpool(graph.invoke, _build_initial_state(req))
    except Exception as exc:
        logger.exception("graph.invoke failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunResponse(
        final_results=state.get("final_results", []) or [],
        trace=state.get("trace", []) or [],
        evaluation=state.get("evaluation"),
    )


@app.post("/run/stream")
async def run_stream(req: RunRequest) -> StreamingResponse:
    logger.info("POST /run/stream language=%s text_len=%d", req.language, len(req.raw_text))
    graph = _load_graph()
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not ready: agent modules on main are not implemented yet.",
        )
    return StreamingResponse(
        stream_workflow(graph, _build_initial_state(req)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
