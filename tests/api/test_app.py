from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


class _FakeGraph:
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "final_results": [{"id": "p1", "title": "Test listing"}],
            "trace": [{"node": "fake", "ok": True}],
            "evaluation": {"passed": True},
        }

    def stream(self, state: dict[str, Any]):
        yield {"intake": {"requirements_complete": True}}
        yield {"properties_agent": {"candidates": [{"id": "p1"}]}}


def test_health_reports_graph_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_load_graph", lambda: None)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph_ready"] is False


def test_run_returns_results_when_graph_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_load_graph", lambda: _FakeGraph())
    resp = client.post("/run", json={"raw_text": "2-bed in Bogota", "language": "en"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_results"] == [{"id": "p1", "title": "Test listing"}]
    assert body["evaluation"] == {"passed": True}
    assert body["error"] is None


def test_run_returns_friendly_error_when_graph_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_load_graph", lambda: None)
    resp = client.post("/run", json={"raw_text": "hi", "language": "en"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_results"] == []
    assert "not ready" in (body["error"] or "").lower()


def test_run_validates_empty_text(client: TestClient) -> None:
    resp = client.post("/run", json={"raw_text": "", "language": "en"})
    assert resp.status_code == 422


def test_run_stream_emits_sse_events(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_load_graph", lambda: _FakeGraph())
    with client.stream("POST", "/run/stream", json={"raw_text": "hi", "language": "en"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = b"".join(resp.iter_bytes()).decode()
    assert "event: node_start" in body
    assert "event: node_end" in body
    assert "event: done" in body
