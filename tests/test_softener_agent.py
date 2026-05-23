"""Standalone-runnable test for softener_node.

Mirrors tests/test_evaluator_agent.py: private _scenario_* helpers return
bool and print via _check; main() aggregates them; thin def test_* wrappers
at the bottom assert the scenarios for pytest. Skips cleanly without
OPENAI_API_KEY.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.softener_agent import softener_node  # noqa: E402
from src.state import (  # noqa: E402
    Constraint,
    EvaluationResult,
    FailureReason,
    PropertyFinderState,
    StructuredRequirements,
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _build_state_price_20pct_over() -> PropertyFinderState:
    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="price",
                exact_value=None,
                min_value=None,
                max_value=3_000_000,
                constraint_type="hard",
                importance="critical",
            ),
        ],
        summary="Arriendo bajo 3M COP en Medellín",
        priority_weights={"price": 1.0},
    )
    evaluation = EvaluationResult(
        passes=False,
        candidate_scores=[],
        aggregate_failure_reasons=[
            FailureReason(
                constraint_field="price",
                expected="<= 3,000,000 COP",
                actual="3,600,000 COP",
                deviation=0.20,
                importance="critical",
            )
        ],
        notes="Todos los candidatos excedieron el presupuesto.",
    )
    return PropertyFinderState(
        user_query="Quiero un arriendo bajo 3M en Medellín",
        chat_history=[{"role": "user", "content": "Quiero un arriendo bajo 3M en Medellín"}],
        requirements=requirements,
        evaluation=evaluation,
        softening_attempts=0,
        softening_history=[],
    )


def _scenario_price_relaxed_capped_at_15pct() -> bool:
    print("\ntest_price_relaxed_capped_at_15pct:")
    state = _build_state_price_20pct_over()
    out = softener_node(state)
    ok = True

    new_req = out.get("requirements")
    ok &= _check("requirements returned", new_req is not None)
    if new_req is None:
        return ok

    price_c = next((c for c in new_req.constraints if c.field == "price"), None)
    ok &= _check("price constraint present", price_c is not None)
    expected_max = 3_000_000 * 1.15
    ok &= _check(
        "price max_value raised by 15% cap (not the full 20% deviation)",
        price_c is not None and abs((price_c.max_value or 0) - expected_max) < 1.0,
        detail=f"got={price_c.max_value if price_c else None} expected={expected_max}",
    )

    chat_delta = out.get("chat_history") or []
    ok &= _check("chat_history append has exactly one entry", len(chat_delta) == 1)
    last = chat_delta[-1] if chat_delta else {}
    ok &= _check(
        "appended message role is 'assistant'",
        last.get("role") == "assistant",
        detail=f"got role={last.get('role')}",
    )
    ok &= _check(
        "appended message content is non-empty string",
        isinstance(last.get("content"), str) and len(last["content"].strip()) > 0,
    )

    ok &= _check(
        "softening_attempts incremented to 1",
        out.get("softening_attempts") == 1,
        detail=f"got {out.get('softening_attempts')}",
    )

    hist_delta = out.get("softening_history") or []
    ok &= _check("softening_history append has one entry", len(hist_delta) == 1)
    if hist_delta:
        entry = hist_delta[0]
        ok &= _check(
            "softening_history entry targets the price constraint",
            entry.relaxed_constraint == "price",
            detail=f"got {entry.relaxed_constraint}",
        )
        ok &= _check(
            "softening_history entry attempt_number == 1",
            entry.attempt_number == 1,
        )

    return ok


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set — softener tests need a real LLM call")
        return 0
    print("=== softener_node ===")
    ok = True
    ok &= _scenario_price_relaxed_capped_at_15pct()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


def test_price_relaxed_capped_at_15pct() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_price_relaxed_capped_at_15pct()


if __name__ == "__main__":
    raise SystemExit(main())
