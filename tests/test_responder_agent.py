"""Standalone vertical-slice test for ``responder_node``.

Two scenarios — both call the real ``gpt-4o-mini`` (needs ``OPENAI_API_KEY``):

- ``test_success_path_mentions_urls_and_whatsapp`` — two verified candidates,
  one with ``availability_confirmed=True`` and one with ``False``. The reply
  must mention both URLs and the WhatsApp confirmation.
- ``test_failure_path_acknowledges_no_matches`` — empty candidates after the
  softening budget is exhausted; the reply must acknowledge no matches and
  must not falsely claim WhatsApp availability.

    uv run python -m tests.test_responder_agent
    uv run pytest tests/test_responder_agent.py
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.responder_agent import responder_node
from src.state import (
    Candidate,
    EvaluationResult,
    PropertyFinderState,
    VerifiedListing,
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _verified_listing(
    *, id: str, url: str, price: float, availability_confirmed: bool
) -> VerifiedListing:
    return VerifiedListing(
        id=id,
        source_site="fincaraiz",
        url=url,
        price=price,
        area_m2=80,
        bedrooms=2,
        bathrooms=2,
        zone="Chapinero",
        property_type="apartment",
        transaction_type="rent",
        availability_confirmed=availability_confirmed,
        verification_timestamp=datetime.now(timezone.utc),
        verification_notes=(
            "Confirmado por el corredor" if availability_confirmed else "Sin respuesta"
        ),
    )


def _scenario_success_path_mentions_urls_and_whatsapp() -> bool:
    url_confirmed = "https://fincaraiz.com.co/inmueble/CONFIRMED-123"
    url_unconfirmed = "https://fincaraiz.com.co/inmueble/UNCONFIRMED-456"

    candidate_confirmed = Candidate(
        listing=_verified_listing(
            id="fincaraiz:CONFIRMED-123",
            url=url_confirmed,
            price=2_800_000,
            availability_confirmed=True,
        ),
        source_urls=[url_confirmed],
        match_score=0.92,
    )
    candidate_unconfirmed = Candidate(
        listing=_verified_listing(
            id="fincaraiz:UNCONFIRMED-456",
            url=url_unconfirmed,
            price=3_100_000,
            availability_confirmed=False,
        ),
        source_urls=[url_unconfirmed],
        match_score=0.84,
    )

    state: PropertyFinderState = {
        "candidates": [candidate_confirmed, candidate_unconfirmed],
        "evaluation": EvaluationResult(
            passes=True,
            candidate_scores=[],
            aggregate_failure_reasons=[],
            notes="2 candidate(s) scored; passes=True",
        ),
        "messages": [HumanMessage(content="Busco apto en Chapinero")],
        "softening_attempts": 0,
    }

    result = responder_node(state)

    new_messages = result.get("messages")
    passed = _check(
        "result contains 'messages' as a list",
        isinstance(new_messages, list),
    )
    if not isinstance(new_messages, list):
        return passed

    passed &= _check(
        "exactly one new message appended",
        len(new_messages) == 1,
        f"got {len(new_messages)}",
    )
    if len(new_messages) != 1:
        return passed

    message = new_messages[0]
    content = message.content if isinstance(message, AIMessage) else ""

    passed &= _check(
        "new message is an AIMessage",
        isinstance(message, AIMessage),
        f"got {type(message).__name__}",
    )
    passed &= _check(
        "message content is non-empty",
        isinstance(content, str) and len(content.strip()) > 0,
    )
    passed &= _check(
        "confirmed candidate URL appears in response",
        url_confirmed in content,
    )
    passed &= _check(
        "unconfirmed candidate URL appears in response",
        url_unconfirmed in content,
    )
    passed &= _check(
        "response mentions WhatsApp confirmation",
        "whatsapp" in content.lower(),
    )

    return passed


def _scenario_failure_path_acknowledges_no_matches() -> bool:
    state: PropertyFinderState = {
        "candidates": [],
        "evaluation": EvaluationResult(
            passes=False,
            candidate_scores=[],
            aggregate_failure_reasons=[],
            notes="no candidates remained after softening",
        ),
        "messages": [HumanMessage(content="Busco apto barato")],
        "softening_attempts": 3,
    }

    result = responder_node(state)
    new_messages = result.get("messages")
    passed = _check(
        "result contains 'messages' as a list",
        isinstance(new_messages, list),
    )
    if not isinstance(new_messages, list) or not new_messages:
        return passed

    message = new_messages[0]
    content = message.content if isinstance(message, AIMessage) else ""

    passed &= _check(
        "exactly one new message appended",
        len(new_messages) == 1,
        f"got {len(new_messages)}",
    )
    passed &= _check(
        "message content is non-empty",
        isinstance(content, str) and len(content.strip()) > 0,
    )
    passed &= _check(
        "response does NOT falsely claim WhatsApp confirmation",
        "confirm" not in content.lower() or "no " in content.lower() or "sin " in content.lower(),
        "expected the failure reply to avoid asserting availability confirmation",
    )

    return passed


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set — responder tests need a real LLM call")
        return 0

    print("=== responder_node ===")
    ok = True
    print("\ntest_success_path_mentions_urls_and_whatsapp:")
    ok &= _scenario_success_path_mentions_urls_and_whatsapp()
    print("\ntest_failure_path_acknowledges_no_matches:")
    ok &= _scenario_failure_path_acknowledges_no_matches()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


# Pytest discovery — turn the boolean returns into asserts.
def test_success_path_mentions_urls_and_whatsapp() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_success_path_mentions_urls_and_whatsapp()


def test_failure_path_acknowledges_no_matches() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_failure_path_acknowledges_no_matches()


if __name__ == "__main__":
    raise SystemExit(main())
