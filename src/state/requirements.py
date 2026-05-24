"""Structured user requirements produced by `requirements_agent`.

A `Constraint` is the atomic unit the evaluator scores against and the softener
relaxes. The graph reads `StructuredRequirements` from
`PropertyFinderState["requirements"]`; routers downstream use it to decide
which branches to activate.
"""

from typing import Literal

from pydantic import BaseModel, Field

ConstraintType = Literal["hard", "soft"]
"""Hard constraints are gating — a violation fails the candidate outright.
Soft constraints contribute to the score but do not gate."""

Importance = Literal["critical", "important", "nice_to_have"]
"""Importance label. Maps to a numeric weight (0.5 / 0.3 / 0.1) elsewhere in
code — the state stores the label only."""


class Constraint(BaseModel):
    """A single user-stated requirement against a property field.

    Written by: `requirements_agent`.
    Read by: `router_agent` (to plan fetches), `evaluator_agent` (to score),
    `softener_agent` (to choose what to relax).

    Exactly one of `exact_value` *or* the pair `(min_value, max_value)` is
    typically populated; using three explicit fields (rather than a single
    `value: Any`) gives the LLM a stable schema to target and Pydantic real
    validation.
    """

    field: str = Field(
        ..., description="Property attribute being constrained, e.g. 'price', 'bedrooms', 'zone'."
    )
    exact_value: int | float | str | None = Field(
        default=None,
        description="Single-value constraint (e.g. zone='Chapinero'). Mutually exclusive with min/max.",
    )
    min_value: int | float | None = Field(
        default=None, description="Inclusive lower bound for range constraints."
    )
    max_value: int | float | None = Field(
        default=None, description="Inclusive upper bound for range constraints."
    )
    constraint_type: ConstraintType = Field(
        default="hard",
        description=(
            "'hard' gates the candidate; 'soft' only affects the score. "
            "Defaults to 'hard': a user-stated requirement is gating unless "
            "the LLM marks it otherwise."
        ),
    )
    importance: Importance = Field(
        default="important",
        description=(
            "Importance label; weight mapping lives outside the state. "
            "Defaults to 'important' so an LLM extraction that omits the "
            "label still validates."
        ),
    )


class StructuredRequirements(BaseModel):
    """The full parsed brief the rest of the pipeline operates on.

    Written by: `requirements_agent` (possibly across multiple clarification
    turns).
    Read by: every downstream agent.
    """

    constraints: list[Constraint] = Field(
        default_factory=list, description="All extracted constraints, hard and soft."
    )
    summary: str | None = Field(
        default=None,
        description="Natural-language recap of the brief, for UI display and agent context.",
    )
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {"price": 0.34, "location": 0.33, "security": 0.33},
        description=(
            "Macro-level user priority weights across thematic axes, applied by "
            "evaluator_agent on top of per-Constraint importance. Values MUST sum "
            "to exactly 1.0. Canonical keys are English: 'price', 'location', "
            "'security'. When the user writes in Spanish, translate before "
            "populating — e.g. 'seguridad' -> 'security', 'precio' -> 'price', "
            "'ubicación'/'ubicacion'/'zona' -> 'location'. Use the default split "
            "only when the user expresses no preference."
        ),
    )
