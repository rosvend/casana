"""API smoke tests for the chat-based endpoints.

Each test uses an isolated graph (its own checkpointer) by overriding
the ``get_graph`` dependency. LLM-hitting tests skip cleanly when
``OPENAI_API_KEY`` is missing.

Coverage:

- ``/health`` returns ok + graph_ready.
- ``/chat`` triggers an ``interrupt()`` with a clarification_question
  payload when the user query is too vague to extract requirements.
- ``/resume`` continues the paused thread without raising.
- ``/history/{thread_id}`` returns the conversation messages list.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from src.api import deps as deps_module
from src.api.main import app
from src.graph.graph import build_graph, make_memory_checkpointer


@pytest.fixture
def client():
    """Fresh graph per test so checkpointer state doesn't leak between tests."""
    fresh_graph = build_graph(checkpointer=make_memory_checkpointer())
    app.dependency_overrides[deps_module.get_graph] = lambda: fresh_graph
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph_ready"] is True


def test_chat_triggers_interrupt_on_ambiguous_query(client: TestClient) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — needs a real LLM call")

    thread_id = f"t-{uuid.uuid4()}"
    resp = client.post(
        "/chat",
        json={"thread_id": thread_id, "user_message": "Hola, busco un lugar para vivir."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == thread_id
    assert body["interrupt"] is not None, body
    assert isinstance(body["interrupt"]["clarification_question"], str)
    assert body["interrupt"]["clarification_question"].strip() != ""


def test_resume_continues_after_interrupt(client: TestClient) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — needs a real LLM call")

    thread_id = f"t-{uuid.uuid4()}"
    chat_resp = client.post(
        "/chat",
        json={"thread_id": thread_id, "user_message": "Busco apartamento."},
    )
    assert chat_resp.status_code == 200
    # The thread should be paused — otherwise resume has nothing to resume from.
    assert chat_resp.json()["interrupt"] is not None

    resume_resp = client.post(
        "/resume",
        json={
            "thread_id": thread_id,
            "resume_payload": "En Chapinero, Bogotá, presupuesto 3 millones de pesos para arriendo.",
        },
    )
    # The key invariant: resume did not 500. The graph may have finished or
    # paused again for further clarification — both are acceptable here.
    assert resume_resp.status_code == 200


def test_history_returns_messages_for_thread(client: TestClient) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — needs a real LLM call")

    thread_id = f"t-{uuid.uuid4()}"
    client.post("/chat", json={"thread_id": thread_id, "user_message": "Hola"})
    resp = client.get(f"/history/{thread_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"] == thread_id
    assert isinstance(body["messages"], list)


def test_history_for_unknown_thread_is_empty(client: TestClient) -> None:
    resp = client.get(f"/history/never-used-{uuid.uuid4()}")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_chat_validates_empty_message(client: TestClient) -> None:
    resp = client.post("/chat", json={"thread_id": "t1", "user_message": ""})
    assert resp.status_code == 422
