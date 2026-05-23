"""Standalone vertical-slice test for ``evaluator_node``.

Two scenarios — both call the real ``gpt-4o-mini`` (needs ``OPENAI_API_KEY``):

- ``test_security_priority_beats_price`` — two candidates clear the same hard
  constraints; the pricier one sits in a zone the news praises as safe, while
  the cheaper one's zone news warns of robberies. With
  ``priority_weights={"security": 0.7, ...}`` the safer (pricier) candidate
  must end up with the strictly higher ``match_score``.
- ``test_hard_constraint_violation_sets_passes_false`` — a single candidate
  busts a hard price ceiling; the deterministic Python gate must set
  ``EvaluationResult.passes`` to ``False`` and surface the violation in
  ``aggregate_failure_reasons`` regardless of what the LLM scores.

    uv run python -m tests.test_evaluator_agent
    uv run pytest tests/test_evaluator_agent.py
"""

from __future__ import annotations

import os

from src.agents.evaluator_agent import evaluator_node
from src.state import (
    Candidate,
    Constraint,
    Listing,
    NewsItem,
    PropertyFinderState,
    StructuredRequirements,
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _listing(*, id: str, price: float, zone: str = "Chapinero") -> Listing:
    return Listing(
        id=id,
        source_site="fincaraiz",
        url=f"https://fincaraiz.com.co/inmueble/{id}",
        price=price,
        area_m2=70,
        bedrooms=2,
        bathrooms=2,
        zone=zone,
        property_type="apartment",
        transaction_type="rent",
    )


def _news(*, title: str, summary: str, zone: str = "Chapinero") -> NewsItem:
    return NewsItem(
        title=title,
        summary=summary,
        url="https://example.com/news",
        source="example.com",
        zone=zone,
    )


def _requirements_security_first() -> StructuredRequirements:
    return StructuredRequirements(
        constraints=[
            Constraint(
                field="price",
                max_value=5_000_000,
                constraint_type="hard",
                importance="critical",
            ),
            Constraint(
                field="bedrooms",
                exact_value=2,
                constraint_type="hard",
                importance="important",
            ),
        ],
        summary="Security-first renter looking for a 2BR in Chapinero",
        priority_weights={"security": 0.7, "price": 0.2, "location": 0.1},
    )


def _scenario_security_priority_beats_price() -> bool:
    candidate_a = Candidate(
        listing=_listing(id="fincaraiz:A_cheap_unsafe", price=2_500_000),
        relevant_news={
            "crime_safety": [
                _news(
                    title="Spike in robberies hits Chapinero",
                    summary=(
                        "Police report a 40% rise in armed robberies this month in "
                        "the zone, with several incidents at night near residential "
                        "blocks."
                    ),
                )
            ],
            "transportation": [],
            "infrastructure": [],
            "events": [],
            "market_trends": [],
        },
        source_urls=["https://fincaraiz.com.co/inmueble/A_cheap_unsafe"],
    )
    candidate_b = Candidate(
        listing=_listing(id="fincaraiz:B_pricier_safe", price=3_200_000),
        relevant_news={
            "crime_safety": [
                _news(
                    title="Chapinero ranked safest neighborhood in the city",
                    summary=(
                        "Latest city-wide safety index lists Chapinero as the "
                        "safest zone, citing low crime and visible police presence."
                    ),
                )
            ],
            "transportation": [],
            "infrastructure": [],
            "events": [],
            "market_trends": [],
        },
        source_urls=["https://fincaraiz.com.co/inmueble/B_pricier_safe"],
    )

    state: PropertyFinderState = {
        "candidates": [candidate_a, candidate_b],
        "requirements": _requirements_security_first(),
    }
    result = evaluator_node(state)

    evaluation = result.get("evaluation")
    candidates = result.get("candidates") or []

    passed = _check("result has 'evaluation' key", evaluation is not None)
    passed &= _check("result has 'candidates' key", isinstance(candidates, list))
    if evaluation is None or not candidates:
        return passed

    passed &= _check(
        "evaluation.passes is True (both candidates clear hard constraints)",
        evaluation.passes is True,
        f"got passes={evaluation.passes}",
    )
    passed &= _check(
        "candidate_scores has length 2",
        len(evaluation.candidate_scores) == 2,
        f"got {len(evaluation.candidate_scores)}",
    )
    passed &= _check(
        "every candidate has match_score in [0.0, 1.0]",
        all(0.0 <= c.match_score <= 1.0 for c in candidates),
    )
    passed &= _check(
        "every candidate has non-empty match_notes (LLM reasoning)",
        all(c.match_notes and c.match_notes.strip() for c in candidates),
    )
    passed &= _check(
        "candidates are sorted desc by match_score",
        all(
            candidates[i].match_score >= candidates[i + 1].match_score
            for i in range(len(candidates) - 1)
        ),
    )

    by_id = {c.listing.id: c for c in candidates}
    safe = by_id.get("fincaraiz:B_pricier_safe")
    unsafe = by_id.get("fincaraiz:A_cheap_unsafe")
    passed &= _check(
        "both candidates present in result",
        safe is not None and unsafe is not None,
    )
    if safe is not None and unsafe is not None:
        passed &= _check(
            "safer (pricier) candidate scores STRICTLY higher than cheaper unsafe one",
            safe.match_score > unsafe.match_score,
            f"safe={safe.match_score:.3f} vs unsafe={unsafe.match_score:.3f}",
        )

    return passed


def _scenario_hard_constraint_violation_sets_passes_false() -> bool:
    over_budget = Candidate(
        listing=_listing(id="fincaraiz:OVER", price=8_000_000),
        relevant_news={
            "crime_safety": [],
            "transportation": [],
            "infrastructure": [],
            "events": [],
            "market_trends": [],
        },
        source_urls=["https://fincaraiz.com.co/inmueble/OVER"],
    )

    state: PropertyFinderState = {
        "candidates": [over_budget],
        "requirements": _requirements_security_first(),  # price max_value=5_000_000
    }
    result = evaluator_node(state)
    evaluation = result.get("evaluation")

    passed = _check("result has 'evaluation' key", evaluation is not None)
    if evaluation is None:
        return passed

    passed &= _check(
        "evaluation.passes is False (only candidate violates hard price ceiling)",
        evaluation.passes is False,
        f"got passes={evaluation.passes}",
    )
    fields_in_failures = {r.constraint_field for r in evaluation.aggregate_failure_reasons}
    passed &= _check(
        "'price' violation surfaces in aggregate_failure_reasons",
        "price" in fields_in_failures,
        f"got fields={sorted(fields_in_failures)}",
    )
    passed &= _check(
        "candidate_scores[0].violated_constraints is non-empty",
        len(evaluation.candidate_scores) == 1
        and len(evaluation.candidate_scores[0].violated_constraints) >= 1,
    )
    return passed


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: OPENAI_API_KEY not set — evaluator tests need a real LLM call")
        return 0

    print("=== evaluator_node ===")
    ok = True
    print("\ntest_security_priority_beats_price:")
    ok &= _scenario_security_priority_beats_price()
    print("\ntest_hard_constraint_violation_sets_passes_false:")
    ok &= _scenario_hard_constraint_violation_sets_passes_false()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


# Pytest discovery — turn the boolean returns into asserts.
def test_security_priority_beats_price() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_security_priority_beats_price()


def test_hard_constraint_violation_sets_passes_false() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_hard_constraint_violation_sets_passes_false()


if __name__ == "__main__":
    raise SystemExit(main())
