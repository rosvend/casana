from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger("estatia.api.streaming")


def format_sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def trace_event(event_type: str, node: str | None = None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": event_type, "node": node, "payload": payload, "ts": time.time()}


async def stream_workflow(graph: Any, initial_state: dict[str, Any]) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def producer() -> None:
        try:
            for chunk in graph.stream(initial_state):
                loop.call_soon_threadsafe(queue.put_nowait, ("update", chunk))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:
            logger.exception("graph.stream failed")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    asyncio.create_task(asyncio.to_thread(producer))

    yield format_sse("node_start", trace_event("node_start", node="__start__"))
    while True:
        kind, payload = await queue.get()
        if kind == "update":
            for node_name, node_state in payload.items():
                yield format_sse(
                    "node_end",
                    trace_event("node_end", node=node_name, payload=_safe_payload(node_state)),
                )
        elif kind == "done":
            yield format_sse("done", trace_event("done"))
            return
        elif kind == "error":
            yield format_sse("error", trace_event("error", payload={"message": payload}))
            return


def _safe_payload(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return {"value": _to_jsonable(value)}


def _to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
