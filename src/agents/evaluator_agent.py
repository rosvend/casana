"""`evaluator_node` — score candidates and decide whether the search passes.

Two layers in one node:

1. **Subjective scoring (LLM).** ``gpt-4o-mini`` reads the user's
   ``StructuredRequirements`` (especially ``priority_weights``) plus the
   candidate's listing fields and pre-attached ``relevant_news`` slice, and
   returns an int 0–100 with a short reasoning. The highest-weighted axis
   must dominate the score (e.g. when ``security`` is weighted 0.7 and the
   news mentions a robbery spike, the score collapses even with a perfect
   price).

2. **Deterministic hard-constraint gate (Python).** Every ``Constraint`` with
   ``constraint_type == "hard"`` is checked against the listing. The result
   feeds ``EvaluationResult.passes`` (which ``evaluation_router`` in
   ``graph.py`` reads): ``True`` iff at least one candidate violates zero
   hard constraints. Violations also populate ``aggregate_failure_reasons``
   for the softener.

The LLM's int 0–100 is normalized to ``float`` in ``[0.0, 1.0]`` before
being written to ``Candidate.match_score`` and ``CandidateScore.score``.
Candidates are returned sorted descending by ``match_score``.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import (
    Candidate,
    CandidateScore,
    Constraint,
    EvaluationResult,
    FailureReason,
    Listing,
    NewsCategory,
    NewsItem,
    PropertyFinderState,
    StructuredRequirements,
    VerifiedListing,
)

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gpt-5-nano"

_SYSTEM_PROMPT = """\
You are a strict real estate match judge. Given a user's requirements (with
explicit priority weights across thematic axes — price, location, security)
and a single candidate property (listing fields plus news about its zone),
return an integer score from 0 to 100 plus a short reasoning.

Scoring rules — follow them literally:
- The priority weights tell you what the user *cares about*. The axis with
  the highest weight must dominate the score. A 0.7 weight on security
  means: if zone news shows safety problems, the score drops sharply, even
  if price and other fields are perfect.
- Conversely, positive news on a high-weighted axis lifts the score.
- Lower-weighted axes can move the score but cannot override a serious
  miss on a high-weighted one.
- 100 = perfect match on every weighted axis with positive corroborating
  news; 0 = total mismatch.
- The reasoning must cite (a) the highest-weighted axis and how the
  candidate fared on it, and (b) any specific news item that moved the
  score.

GEOGRAPHIC DISCRIMINATION (critical):
- You must evaluate news strictly against the candidate's listed zone
  (neighborhood). The candidate's zone is shown in the CANDIDATE LISTING
  block; each news item shows its own zone in the ZONE NEWS block.
- Do NOT penalize a property for negative news whose zone differs from the
  candidate's zone, or for general city-wide news that does not explicitly
  mention the candidate's zone. Such news is geographically irrelevant.
- Differentiate your scores across candidates based on how news impacts
  each candidate's exact neighborhood — two candidates in different zones
  must not receive the same score unless their zone-specific news is
  equivalent.
- In `reasoning`, explicitly state if you ignored any news item as
  geographically irrelevant, naming the item and the mismatched zone.
"""


class _LLMCandidateEvaluation(BaseModel):
    """Internal LLM output schema. Not part of the public state surface."""

    score: int = Field(
        ..., ge=0, le=100, description="0 = total mismatch, 100 = perfect match."
    )
    reasoning: str = Field(
        ...,
        description=(
            "Short explanation citing the highest-weighted axis and any news that "
            "moved the score."
        ),
    )


def evaluator_node(state: PropertyFinderState) -> dict:
    """Score each candidate; return updated candidates + ``EvaluationResult``.

    Reads ``state["candidates"]`` and ``state["requirements"]``. Returns
    ``{"candidates": sorted_candidates, "evaluation": EvaluationResult(...)}``.
    Empty/missing inputs produce a non-passing evaluation rather than raising.
    """
    candidates: list[Candidate] = state.get("candidates") or []
    requirements: StructuredRequirements | None = state.get("requirements")

    if not candidates:
        logger.info("evaluator_node: no candidates to evaluate")
        return {
            "candidates": [],
            "evaluation": EvaluationResult(
                passes=False,
                candidate_scores=[],
                aggregate_failure_reasons=[],
                notes="no candidates to evaluate",
            ),
        }

    if requirements is None:
        logger.info("evaluator_node: no requirements on state; cannot gate")
        return {
            "candidates": candidates,
            "evaluation": EvaluationResult(
                passes=False,
                candidate_scores=[],
                aggregate_failure_reasons=[],
                notes="no requirements available to evaluate against",
            ),
        }

    hard_constraints = [
        c for c in requirements.constraints if c.constraint_type == "hard"
    ]

    llm = ChatOpenAI(model=MODEL, temperature=0)
    structured_llm = llm.with_structured_output(
        _LLMCandidateEvaluation, method="function_calling"
    )

    scored: list[Candidate] = []
    candidate_scores: list[CandidateScore] = []

    for candidate in candidates:
        try:
            evaluation = structured_llm.invoke(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("human", _build_prompt(candidate, requirements)),
                ]
            )
            llm_score = evaluation.score
            reasoning = evaluation.reasoning
        except Exception as exc:  # noqa: BLE001 — degrade per-candidate, not whole node
            logger.warning(
                "evaluator_node: LLM call failed for %s: %s",
                candidate.listing.id, exc,
            )
            llm_score = 0
            reasoning = f"evaluation failed: {exc}"

        normalized = max(0.0, min(1.0, llm_score / 100.0))
        violations = _check_hard_constraints(candidate.listing, hard_constraints)
        matched_fields = [
            c.field
            for c in hard_constraints
            if c.field not in {v.constraint_field for v in violations}
        ]

        candidate_scores.append(
            CandidateScore(
                candidate_id=candidate.listing.id,
                score=normalized,
                violated_constraints=violations,
                matched_constraint_fields=matched_fields,
            )
        )
        scored.append(
            candidate.model_copy(
                update={"match_score": normalized, "match_notes": reasoning}
            )
        )

    passes = any(len(cs.violated_constraints) == 0 for cs in candidate_scores)
    aggregate = _aggregate_failures(candidate_scores)
    scored.sort(key=lambda c: c.match_score, reverse=True)

    logger.info(
        "evaluator_node: scored %d candidate(s); passes=%s; aggregate_failures=%d",
        len(scored), passes, len(aggregate),
    )

    return {
        "candidates": scored,
        "evaluation": EvaluationResult(
            passes=passes,
            candidate_scores=candidate_scores,
            aggregate_failure_reasons=aggregate,
            notes=f"{len(scored)} candidate(s) scored; passes={passes}",
        ),
    }


def _build_prompt(
    candidate: Candidate, requirements: StructuredRequirements
) -> str:
    """Render the per-candidate human message for the LLM judge."""
    listing = candidate.listing

    weights_lines = "\n".join(
        f"  - {axis}: {weight:.2f}"
        for axis, weight in requirements.priority_weights.items()
    ) or "  (no explicit weights — use defaults equally)"

    summary = requirements.summary or "(no summary)"

    listing_lines = "\n".join(
        f"  - {label}: {value}"
        for label, value in (
            ("id", listing.id),
            ("price (COP)", listing.price),
            ("bedrooms", listing.bedrooms),
            ("bathrooms", listing.bathrooms),
            ("area_m2", listing.area_m2),
            ("zone", listing.zone),
            ("estrato", listing.estrato),
            ("property_type", listing.property_type),
            ("transaction_type", listing.transaction_type),
        )
        if value is not None
    )

    news_lines = _render_news(candidate.relevant_news) or "  (no news attached)"

    return (
        "USER REQUIREMENTS\n"
        f"Summary: {summary}\n"
        f"Priority weights (must dominate scoring):\n{weights_lines}\n\n"
        "CANDIDATE LISTING\n"
        f"{listing_lines}\n\n"
        "ZONE NEWS\n"
        f"{news_lines}\n\n"
        "Return your structured evaluation."
    )


def _render_news(news: dict[NewsCategory, list[NewsItem]]) -> str:
    """Flatten the news dict into bulleted lines, skipping empty categories."""
    blocks: list[str] = []
    for category, items in news.items():
        if not items:
            continue
        for item in items:
            zone_label = item.zone or "(no zone)"
            blocks.append(
                f"  - [{category}] (zone: {zone_label}) {item.title} — {item.summary}"
            )
    return "\n".join(blocks)


def _check_hard_constraints(
    listing: Listing | VerifiedListing, hard_constraints: list[Constraint]
) -> list[FailureReason]:
    """Return a ``FailureReason`` for every hard constraint the listing misses."""
    failures: list[FailureReason] = []
    for constraint in hard_constraints:
        value = getattr(listing, constraint.field, None)

        if value is None:
            failures.append(
                FailureReason(
                    constraint_field=constraint.field,
                    expected=_render_expected(constraint),
                    actual="unknown",
                    deviation=None,
                    importance=constraint.importance,
                )
            )
            continue

        if constraint.exact_value is not None:
            if value != constraint.exact_value:
                failures.append(
                    FailureReason(
                        constraint_field=constraint.field,
                        expected=f"== {constraint.exact_value}",
                        actual=str(value),
                        deviation=None,
                        importance=constraint.importance,
                    )
                )
            continue

        if constraint.max_value is not None and isinstance(value, (int, float)):
            if value > constraint.max_value:
                deviation = (
                    (value - constraint.max_value) / constraint.max_value
                    if constraint.max_value
                    else None
                )
                failures.append(
                    FailureReason(
                        constraint_field=constraint.field,
                        expected=f"<= {constraint.max_value}",
                        actual=str(value),
                        deviation=deviation,
                        importance=constraint.importance,
                    )
                )
                continue

        if constraint.min_value is not None and isinstance(value, (int, float)):
            if value < constraint.min_value:
                deviation = (
                    (constraint.min_value - value) / constraint.min_value
                    if constraint.min_value
                    else None
                )
                failures.append(
                    FailureReason(
                        constraint_field=constraint.field,
                        expected=f">= {constraint.min_value}",
                        actual=str(value),
                        deviation=deviation,
                        importance=constraint.importance,
                    )
                )

    return failures


def _render_expected(constraint: Constraint) -> str:
    """Human-readable rendering of a constraint's expectation for FailureReason."""
    if constraint.exact_value is not None:
        return f"== {constraint.exact_value}"
    bounds = []
    if constraint.min_value is not None:
        bounds.append(f">= {constraint.min_value}")
    if constraint.max_value is not None:
        bounds.append(f"<= {constraint.max_value}")
    return " and ".join(bounds) if bounds else "(no expectation)"


def _aggregate_failures(
    candidate_scores: list[CandidateScore],
) -> list[FailureReason]:
    """One representative FailureReason per field — keep the worst deviation."""
    worst: dict[str, FailureReason] = {}
    for cs in candidate_scores:
        for reason in cs.violated_constraints:
            existing = worst.get(reason.constraint_field)
            if existing is None:
                worst[reason.constraint_field] = reason
                continue
            if _deviation_magnitude(reason) > _deviation_magnitude(existing):
                worst[reason.constraint_field] = reason
    return list(worst.values())


def _deviation_magnitude(reason: FailureReason) -> float:
    """Sort key for picking the worst violation; ``None`` sorts last."""
    return abs(reason.deviation) if reason.deviation is not None else -1.0
