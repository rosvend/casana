"""Server-Sent Events streaming for the chat endpoint.

Drives ``graph.astream_events(..., version="v2")`` and emits three
SSE event types:

- ``status`` — a one-line Colombian-Spanish status message emitted **before**
  each node runs (on ``on_chain_start``), sourced from
  ``NODE_STATUS_MESSAGES`` and templated with live state (location/zone,
  news category). The user sees "Estoy buscando propiedades…" *while* the
  multi-minute scrape is running, not after it finishes.
- ``message`` — emitted on ``on_chain_end`` so a streaming frontend can
  flush a "node done" marker (carries no payload — fetch the full state
  via ``done`` or ``/history``).
- ``done`` — terminal event carrying the final aggregated state
  (``messages``, ``final_results``, ``evaluation``, and any pending
  ``interrupt`` payload).

Why ``astream_events`` and not ``astream(stream_mode="updates")``: the
``updates`` stream only fires when a node finishes, so any "I'm about
to do X" status arrives after X is already done. ``astream_events``
exposes both start and end of every chain step, which is what the live
status UX needs.

The state-templated status strings use ``str.format_map`` with a
``defaultdict(lambda: "tu zona")`` fallback, so missing keys quietly
collapse to a generic phrase rather than raising ``KeyError``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger("estatia.api.streaming")


NODE_STATUS_MESSAGES: dict[str, str] = {
    "requirements_agent": "Estoy entendiendo lo que buscas…",
    "properties_agent": "Estoy buscando propiedades en {location_str}…",
    "news_agent": "Estoy analizando noticias relevantes sobre {location_str}…",
    "synthesizer_agent": "Estoy uniendo propiedades y noticias…",
    "evaluator_agent": "Estoy evaluando si las propiedades encontradas son suficientes…",
    "softener_agent": "Estoy relajando algunos criterios para encontrar más opciones…",
    "whatsapp_agent": "Estoy verificando disponibilidad por WhatsApp…",
    "responder_agent": "Estoy redactando tu respuesta…",
}


def _format_sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


def _location_str(state: dict[str, Any] | None) -> str:
    """Pull a human zone/city string from the current requirements snapshot."""
    if not state:
        return "tu zona"
    requirements = state.get("requirements")
    if requirements is None:
        return "tu zona"
    constraints = getattr(requirements, "constraints", None) or []
    zone = next(
        (
            c.exact_value
            for c in constraints
            if c.field == "zone" and isinstance(c.exact_value, str)
        ),
        None,
    )
    location = next(
        (
            c.exact_value
            for c in constraints
            if c.field == "location" and isinstance(c.exact_value, str)
        ),
        None,
    )
    if zone and location:
        return f"{zone.title()}, {location.title()}"
    if zone:
        return zone.title()
    if location:
        return location.title()
    return "tu zona"


def _status_for(node: str, state: dict[str, Any] | None) -> str | None:
    template = NODE_STATUS_MESSAGES.get(node)
    if template is None:
        return None
    fields: dict[str, str] = defaultdict(lambda: "tu zona")
    fields["location_str"] = _location_str(state)
    return template.format_map(fields)


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Serialize a LangChain message into the {type, content, id} shape the UI expects."""
    type_map = {
        HumanMessage: "human",
        AIMessage: "ai",
        SystemMessage: "system",
        ToolMessage: "tool",
    }
    type_name = next(
        (label for cls, label in type_map.items() if isinstance(message, cls)),
        message.__class__.__name__.lower(),
    )
    content = message.content if isinstance(message.content, str) else str(message.content)
    return {"type": type_name, "content": content, "id": getattr(message, "id", None)}


async def stream_chat(
    graph: Any,
    invocation_input: Any,
    config: dict[str, Any],
) -> AsyncIterator[bytes]:
    """Yield SSE frames for one chat or resume invocation.

    ``invocation_input`` is either a ``{"messages": [...]}`` dict (fresh
    turn) or a ``Command(resume=...)`` (continuation). The terminal
    ``done`` frame carries final state + any pending interrupt payload.

    ``astream_events`` fires events for the parent graph and every nested
    chain (LLM calls, sub-runnables); we filter on ``name in
    NODE_STATUS_MESSAGES`` so only top-level node starts/ends emit
    status/message frames.
    """
    sent_status: set[str] = set()
    sent_done: set[str] = set()

    try:
        async for event in graph.astream_events(
            invocation_input, config=config, version="v2"
        ):
            event_type = event.get("event")
            node = event.get("name")
            if node not in NODE_STATUS_MESSAGES:
                continue

            if event_type == "on_chain_start" and node not in sent_status:
                sent_status.add(node)
                # Snapshot the state *as it is right before this node runs*
                # so the status message can be templated with the latest
                # requirements (location/zone).
                snapshot = graph.get_state(config)
                status = _status_for(node, snapshot.values if snapshot else None)
                if status:
                    yield _format_sse(
                        "status",
                        {"node": node, "message": status, "ts": time.time()},
                    )
            elif event_type == "on_chain_end" and node not in sent_done:
                sent_done.add(node)
                yield _format_sse(
                    "message",
                    {"node": node, "ts": time.time()},
                )
    except Exception as exc:
        logger.exception("graph.astream_events failed")
        yield _format_sse("error", {"message": str(exc)})
        return

    snapshot = graph.get_state(config)
    values = snapshot.values if snapshot else {}
    messages = [message_to_dict(m) for m in (values.get("messages") or [])]
    pending_interrupts = list(snapshot.interrupts) if snapshot else []
    interrupt_payload = (
        pending_interrupts[0].value if pending_interrupts else None
    )

    yield _format_sse(
        "done",
        {
            "messages": messages,
            "final_results": values.get("final_results") or [],
            "evaluation": _safe_dump(values.get("evaluation")),
            "is_best_effort": bool(values.get("is_best_effort")),
            "interrupt": interrupt_payload,
        },
    )


def _safe_dump(value: Any) -> Any:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return dump()
    return value
