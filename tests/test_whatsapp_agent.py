"""Standalone-runnable test for ``whatsapp_node``.

Network I/O against EvolutionAPI is mocked end-to-end (``requests.post``,
``requests.get``, ``time.sleep``, ``random.uniform``) so the suite never
actually messages a broker. Mirrors the shape of
``tests/test_softener_agent.py`` and ``tests/test_evaluator_agent.py``:
private ``_scenario_*()`` helpers return bool and print via ``_check``;
``main()`` aggregates them; thin ``def test_*()`` wrappers assert at the
bottom.

    uv run python -m tests.test_whatsapp_agent
    uv run pytest tests/test_whatsapp_agent.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.whatsapp_agent import whatsapp_node  # noqa: E402
from src.state import (  # noqa: E402
    Candidate,
    Listing,
    PropertyFinderState,
    VerifiedListing,
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _listing(*, id: str, phone: str | None) -> Listing:
    return Listing(
        id=id,
        source_site="fincaraiz",
        url=f"https://fincaraiz.com.co/inmueble/{id}",
        price=2_800_000,
        area_m2=70,
        bedrooms=2,
        bathrooms=2,
        zone="Chapinero",
        property_type="apartment",
        transaction_type="rent",
        phone_numbers=[phone] if phone else [],
    )


def _candidate(*, id: str, phone: str | None, score: float) -> Candidate:
    return Candidate(
        listing=_listing(id=id, phone=phone),
        source_urls=[f"https://fincaraiz.com.co/inmueble/{id}"],
        match_score=score,
    )


def _ok_post_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"key": {"id": "MSG_ID"}}
    return resp


def _no_reply_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"messages": {"total": 0, "records": []}}
    return resp


def _reply_response(*, from_number: str) -> MagicMock:
    """Mirror Evolution 2.3.x findMessages shape: messages.records[].key + messageTimestamp."""
    import time as _time

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "messages": {
            "total": 1,
            "records": [
                {
                    "key": {
                        "id": "MOCK_REPLY",
                        "fromMe": False,
                        "remoteJid": f"{from_number}@s.whatsapp.net",
                    },
                    "message": {"conversation": "Sí, aún está disponible."},
                    "messageTimestamp": int(_time.time()) + 60,
                }
            ],
        }
    }
    return resp


def _evolution_env():
    """Patch the three Evolution env vars so the agent treats the gateway as configured."""
    return patch.dict(
        os.environ,
        {
            "EVOLUTION_API_URL": "http://localhost:8080",
            "EVOLUTION_API_KEY": "test-key",
            "EVOLUTION_INSTANCE": "test-instance",
        },
    )


def _post_numbers_called(mock_post: MagicMock) -> list[str]:
    """Extract the ``number`` field sent on /message/sendText calls only."""
    out: list[str] = []
    for call in mock_post.call_args_list:
        url = call.args[0] if call.args else call.kwargs.get("url", "")
        if "sendText" not in url:
            continue
        body = call.kwargs.get("json") or {}
        if "number" in body:
            out.append(str(body["number"]))
    return out


def _post_router(poll_responses: list[MagicMock] | None = None):
    """Side-effect callable that dispatches by URL.

    ``/message/sendText`` calls always return an OK send response;
    ``/chat/findMessages`` (poll) calls consume from ``poll_responses`` in order,
    falling back to ``_no_reply_response()`` once exhausted.
    """
    poll_iter = iter(poll_responses or [])

    def _route(url, *args, **kwargs):
        if "sendText" in url:
            return _ok_post_response()
        try:
            return next(poll_iter)
        except StopIteration:
            return _no_reply_response()

    return _route


def _count_poll_calls(mock_post: MagicMock) -> int:
    return sum(
        1
        for c in mock_post.call_args_list
        if "findMessages" in (c.args[0] if c.args else c.kwargs.get("url", ""))
    )


def _scenario_low_score_not_contacted() -> bool:
    print("\ntest_low_score_not_contacted:")
    state: PropertyFinderState = {
        "whatsapp_enabled": True,
        "candidates": [
            _candidate(id="A", phone="3001111111", score=0.90),
            _candidate(id="B", phone="3002222222", score=0.85),
            _candidate(id="C", phone="3003333333", score=0.80),
            _candidate(id="D_low", phone="3009999999", score=0.50),
        ],
    }
    with _evolution_env(), \
         patch("src.agents.whatsapp_agent.requests.post") as mock_post, \
         patch("src.agents.whatsapp_agent.time.sleep") as _mock_sleep, \
         patch("src.agents.whatsapp_agent.random.uniform", return_value=5.5):
        mock_post.side_effect = _post_router()

        out = whatsapp_node(state)

    ok = True
    candidates = out.get("candidates") or []
    ok &= _check("returns 4 candidates", len(candidates) == 4, f"got {len(candidates)}")
    numbers_called = _post_numbers_called(mock_post)
    ok &= _check(
        "sendText called exactly 3 times (top 3 by score)",
        len(numbers_called) == 3,
        f"got {len(numbers_called)}",
    )
    ok &= _check(
        "low-score candidate's number never sent",
        not any("9999999" in n for n in numbers_called),
        f"numbers={numbers_called}",
    )

    by_id = {c.listing.id: c for c in candidates}
    low = by_id.get("D_low")
    ok &= _check("low-score candidate present in output", low is not None)
    if low is not None:
        ok &= _check(
            "low-score candidate promoted to VerifiedListing",
            isinstance(low.listing, VerifiedListing),
        )
        ok &= _check(
            "low-score candidate has availability_confirmed=False",
            isinstance(low.listing, VerifiedListing) and low.listing.availability_confirmed is False,
        )
        notes = getattr(low.listing, "verification_notes", "") or ""
        ok &= _check(
            "low-score candidate notes mention score / threshold",
            "score" in notes.lower() or "0.4" in notes,
            f"notes={notes!r}",
        )
    return ok


def _scenario_reply_detected_marks_confirmed() -> bool:
    print("\ntest_reply_detected_marks_confirmed:")
    cands = [
        _candidate(id="A", phone="3001111111", score=0.95),
        _candidate(id="B_replies", phone="3002222222", score=0.90),
        _candidate(id="C", phone="3003333333", score=0.80),
    ]
    state: PropertyFinderState = {"whatsapp_enabled": True, "candidates": cands}

    # 6 polls for A (no reply); B replies on poll #2; 6 polls for C (no reply).
    poll_sequence = (
        [_no_reply_response()] * 6
        + [_no_reply_response(), _reply_response(from_number="573002222222")]
        + [_no_reply_response()] * 6
    )

    with _evolution_env(), \
         patch("src.agents.whatsapp_agent.requests.post") as mock_post, \
         patch("src.agents.whatsapp_agent.time.sleep"), \
         patch("src.agents.whatsapp_agent.random.uniform", return_value=5.5), \
         patch(
             "src.agents.whatsapp_agent._is_reply_confirming_availability",
             return_value=True,
         ) as mock_llm_check:
        mock_post.side_effect = _post_router(poll_responses=poll_sequence)

        out = whatsapp_node(state)

    ok = True
    candidates = out.get("candidates") or []
    by_id = {c.listing.id: c for c in candidates}
    b = by_id.get("B_replies")
    a = by_id.get("A")
    c = by_id.get("C")

    ok &= _check("all three candidates returned", all([a, b, c]))
    if a is None or b is None or c is None:
        return ok

    ok &= _check(
        "B (replied) is a VerifiedListing",
        isinstance(b.listing, VerifiedListing),
    )
    ok &= _check(
        "B.availability_confirmed is True",
        isinstance(b.listing, VerifiedListing) and b.listing.availability_confirmed is True,
    )
    ok &= _check(
        "A.availability_confirmed is False (no reply)",
        isinstance(a.listing, VerifiedListing) and a.listing.availability_confirmed is False,
    )
    ok &= _check(
        "C.availability_confirmed is False (no reply)",
        isinstance(c.listing, VerifiedListing) and c.listing.availability_confirmed is False,
    )
    ok &= _check(
        "B's poll stopped early (≤2 polls used for B before short-circuit)",
        _count_poll_calls(mock_post) <= 6 + 2 + 6,
        f"got {_count_poll_calls(mock_post)}",
    )
    ok &= _check(
        "LLM availability check ran exactly once (only B got a reply)",
        mock_llm_check.call_count == 1,
        f"got {mock_llm_check.call_count}",
    )
    ok &= _check(
        "LLM availability check received the reply text",
        mock_llm_check.call_args is not None
        and "disponible" in (mock_llm_check.call_args.args[0] or "").lower(),
        f"args={mock_llm_check.call_args}",
    )
    return ok


def _scenario_jitter_sleep_called() -> bool:
    print("\ntest_jitter_sleep_called:")
    state: PropertyFinderState = {
        "whatsapp_enabled": True,
        "candidates": [
            _candidate(id="A", phone="3001111111", score=0.95),
            _candidate(id="B", phone="3002222222", score=0.90),
        ],
    }
    with _evolution_env(), \
         patch("src.agents.whatsapp_agent.requests.post") as mock_post, \
         patch("src.agents.whatsapp_agent.time.sleep") as mock_sleep, \
         patch("src.agents.whatsapp_agent.random.uniform", return_value=5.5) as mock_uniform:
        mock_post.side_effect = _post_router()

        whatsapp_node(state)

    ok = True
    ok &= _check("time.sleep was called", mock_sleep.called)
    ok &= _check(
        "random.uniform(3, 8) was called for jitter",
        any(args == (3, 8) or args == (3.0, 8.0) for args, _ in mock_uniform.call_args_list),
        f"calls={mock_uniform.call_args_list}",
    )
    ok &= _check(
        "time.sleep received the jitter value 5.5 at least once",
        any(call.args and call.args[0] == 5.5 for call in mock_sleep.call_args_list),
        f"sleep calls={[c.args for c in mock_sleep.call_args_list]}",
    )
    return ok


def _scenario_disabled_via_state_overrides_env() -> bool:
    print("\ntest_disabled_via_state_overrides_env:")
    state: PropertyFinderState = {
        "whatsapp_enabled": False,
        "candidates": [
            _candidate(id="A", phone="3001111111", score=0.95),
            _candidate(id="B", phone="3002222222", score=0.90),
        ],
    }
    saved = os.environ.get("WHATSAPP_ENABLED")
    os.environ["WHATSAPP_ENABLED"] = "true"
    try:
        with patch("src.agents.whatsapp_agent.requests.post") as mock_post, \
             patch("src.agents.whatsapp_agent.time.sleep"), \
             patch("src.agents.whatsapp_agent.random.uniform", return_value=5.5):
            mock_post.side_effect = _post_router()

            out = whatsapp_node(state)
    finally:
        if saved is None:
            os.environ.pop("WHATSAPP_ENABLED", None)
        else:
            os.environ["WHATSAPP_ENABLED"] = saved

    ok = True
    ok &= _check("no POST sent when disabled via state", mock_post.call_count == 0)
    candidates = out.get("candidates") or []
    ok &= _check("returns 2 candidates", len(candidates) == 2)
    ok &= _check(
        "all candidates promoted to VerifiedListing",
        all(isinstance(c.listing, VerifiedListing) for c in candidates),
    )
    ok &= _check(
        "all availability_confirmed == False",
        all(
            isinstance(c.listing, VerifiedListing)
            and c.listing.availability_confirmed is False
            for c in candidates
        ),
    )
    ok &= _check(
        "verification_notes mention 'disabled'",
        all(
            isinstance(c.listing, VerifiedListing)
            and "disabled" in (c.listing.verification_notes or "").lower()
            for c in candidates
        ),
    )
    return ok


def _scenario_missing_phone_numbers_skipped() -> bool:
    print("\ntest_missing_phone_numbers_skipped:")
    state: PropertyFinderState = {
        "whatsapp_enabled": True,
        "candidates": [
            _candidate(id="A", phone="3001111111", score=0.95),
            _candidate(id="B_nophone", phone=None, score=0.90),
            _candidate(id="C", phone="3003333333", score=0.85),
        ],
    }
    with _evolution_env(), \
         patch("src.agents.whatsapp_agent.requests.post") as mock_post, \
         patch("src.agents.whatsapp_agent.time.sleep"), \
         patch("src.agents.whatsapp_agent.random.uniform", return_value=5.5):
        mock_post.side_effect = _post_router()

        out = whatsapp_node(state)

    ok = True
    numbers_called = _post_numbers_called(mock_post)
    ok &= _check(
        "only 2 sendText calls (phone-less candidate skipped)",
        len(numbers_called) == 2,
        f"got {len(numbers_called)}",
    )
    candidates = out.get("candidates") or []
    by_id = {c.listing.id: c for c in candidates}
    b = by_id.get("B_nophone")
    ok &= _check("phone-less candidate still in output", b is not None)
    if b is not None:
        ok &= _check(
            "phone-less candidate promoted to VerifiedListing",
            isinstance(b.listing, VerifiedListing),
        )
        notes = getattr(b.listing, "verification_notes", "") or ""
        ok &= _check(
            "phone-less candidate notes mention missing phone",
            "phone" in notes.lower(),
            f"notes={notes!r}",
        )
    return ok


def main() -> int:
    print("=== whatsapp_node ===")
    ok = True
    ok &= _scenario_low_score_not_contacted()
    ok &= _scenario_reply_detected_marks_confirmed()
    ok &= _scenario_jitter_sleep_called()
    ok &= _scenario_disabled_via_state_overrides_env()
    ok &= _scenario_missing_phone_numbers_skipped()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


def test_low_score_not_contacted() -> None:
    assert _scenario_low_score_not_contacted()


def test_reply_detected_marks_confirmed() -> None:
    assert _scenario_reply_detected_marks_confirmed()


def test_jitter_sleep_called() -> None:
    assert _scenario_jitter_sleep_called()


def test_disabled_via_state_overrides_env() -> None:
    assert _scenario_disabled_via_state_overrides_env()


def test_missing_phone_numbers_skipped() -> None:
    assert _scenario_missing_phone_numbers_skipped()


if __name__ == "__main__":
    raise SystemExit(main())
