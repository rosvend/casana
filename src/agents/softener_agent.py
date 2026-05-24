"""Softener agent — relaxes hard constraints after a failed evaluation.

Triggered by ``evaluation_router`` when ``EvaluationResult.passes`` is False
and ``state["softening_attempts"] < max_softening_attempts``. Reads
``aggregate_failure_reasons`` from the evaluator, mutates a copy of
``requirements.constraints`` with bounded relaxations, records each
relaxation in ``softening_history``, increments ``softening_attempts``,
and appends one assistant message (Colombian Spanish) to ``chat_history``
explaining the adjustment to the user.
"""

from __future__ import annotations

import copy

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.state import (
    Constraint,
    FailureReason,
    PropertyFinderState,
    SofteningAttempt,
    StructuredRequirements,
)

MODEL = "gpt-5-nano"
MAX_RELAX_PCT = 0.15

SYSTEM_PROMPT = (
    "Eres un asistente que busca arriendos en Colombia. "
    "Las preferencias originales del usuario no pudieron cumplirse y tuvimos "
    "que ajustarlas para encontrar más opciones. Escribe UN solo mensaje muy "
    "corto (máximo 2 oraciones), amigable y en español colombiano coloquial, "
    "explicando qué se cambió. No saludes, no uses emojis, no te disculpes "
    "excesivamente."
)


def softener_node(state: PropertyFinderState) -> dict:
    """Relax hard constraints based on the evaluator's failure reasons.

    Returns only the keys this node writes (state is ``TypedDict(total=False)``):
    ``requirements`` (mutated copy), ``chat_history`` (one-message append),
    ``softening_attempts`` (incremented), ``softening_history`` (append list).
    """
    load_dotenv()

    original_requirements = state.get("requirements")
    evaluation = state.get("evaluation")
    if original_requirements is None or evaluation is None:
        return {}

    requirements: StructuredRequirements = copy.deepcopy(original_requirements)
    attempt_number = state.get("softening_attempts", 0) + 1

    change_descriptions: list[str] = []
    history_entries: list[SofteningAttempt] = []

    for reason in evaluation.aggregate_failure_reasons:
        constraint = _find_constraint(requirements, reason.constraint_field)
        if constraint is None:
            continue
        change = _relax_one(constraint, reason)
        if change is None:
            continue
        if change.get("remove"):
            # Zone failures broaden the search to the whole city by dropping
            # the constraint outright — _extract_params then omits the zone
            # slot from the search URL and the scraper returns a wider pool.
            requirements.constraints = [
                c for c in requirements.constraints if c is not constraint
            ]
        change_descriptions.append(change["description"])
        history_entries.append(
            SofteningAttempt(
                attempt_number=attempt_number,
                relaxed_constraint=constraint.field,
                relaxation_description=change["description"],
                previous_value=change["previous"],
                new_value=change["new"],
                subsequent_candidate_count=0,
                subsequent_evaluation_passed=False,
                evaluator_feedback="",
            )
        )

    message = _generate_user_message(change_descriptions)

    return {
        "requirements": requirements,
        "chat_history": [{"role": "assistant", "content": message}],
        "softening_attempts": attempt_number,
        "softening_history": history_entries,
    }


def _find_constraint(
    requirements: StructuredRequirements, field_name: str
) -> Constraint | None:
    for c in requirements.constraints:
        if c.field == field_name:
            return c
    return None


def _relax_one(constraint: Constraint, reason: FailureReason) -> dict | None:
    """Mutate ``constraint`` in place per its field type.

    Returns ``{"description", "previous", "new"}`` describing the change, or
    ``None`` if no relaxation was possible (caller skips it).
    """
    field = constraint.field
    deviation = abs(reason.deviation) if reason.deviation is not None else 0.0
    delta = min(deviation, MAX_RELAX_PCT) or MAX_RELAX_PCT

    if field == "price" and constraint.max_value is not None:
        prev = float(constraint.max_value)
        new = prev * (1 + delta)
        constraint.max_value = new
        return {
            "description": (
                f"Aumenté el presupuesto máximo de "
                f"{prev:,.0f} a {new:,.0f} COP (+{delta*100:.0f}%)."
            ),
            "previous": f"{prev:,.0f} COP",
            "new": f"{new:,.0f} COP",
        }

    if field == "area_m2" and constraint.min_value is not None:
        prev = float(constraint.min_value)
        new = prev * (1 - delta)
        constraint.min_value = new
        return {
            "description": (
                f"Reduje el área mínima de {prev:.0f}m² a {new:.0f}m² "
                f"(-{delta*100:.0f}%)."
            ),
            "previous": f"{prev:.0f} m²",
            "new": f"{new:.0f} m²",
        }

    if field == "zone":
        # Already soft (or removed-then-re-added) — nothing left to relax.
        if constraint.constraint_type == "soft":
            return None
        prev = constraint.exact_value if constraint.exact_value is not None else "(zona)"
        return {
            "description": (
                f"Amplié la búsqueda más allá de '{prev}' a toda la ciudad."
            ),
            "previous": str(prev),
            "new": "(sin zona)",
            "remove": True,
        }

    if field == "location":
        # Dropping the city entirely is too aggressive — convert to soft so
        # the URL still targets the right city but the gate releases.
        if constraint.constraint_type == "soft":
            return None
        constraint.constraint_type = "soft"
        return {
            "description": (
                f"Cambié 'location' de requisito obligatorio a deseable "
                f"manteniendo '{constraint.exact_value}' como preferencia."
            ),
            "previous": "hard",
            "new": "soft",
        }

    if field in ("bedrooms", "bathrooms"):
        label = "habitaciones" if field == "bedrooms" else "baños"
        if (
            constraint.exact_value is not None
            and isinstance(constraint.exact_value, (int, float))
            and constraint.exact_value > 1
        ):
            prev_val = float(constraint.exact_value)
            new_val = prev_val - 1
            constraint.exact_value = None
            constraint.min_value = new_val
            return {
                "description": (
                    f"Cambié {label} de exactamente {prev_val:.0f} a "
                    f"mínimo {new_val:.0f}."
                ),
                "previous": f"= {prev_val:.0f}",
                "new": f">= {new_val:.0f}",
            }
        if constraint.min_value is not None and constraint.min_value > 1:
            prev_val = float(constraint.min_value)
            new_val = prev_val - 1
            constraint.min_value = new_val
            return {
                "description": (
                    f"Reduje el mínimo de {label} de {prev_val:.0f} "
                    f"a {new_val:.0f}."
                ),
                "previous": f">= {prev_val:.0f}",
                "new": f">= {new_val:.0f}",
            }
        return None

    if constraint.constraint_type == "hard":
        constraint.constraint_type = "soft"
        return {
            "description": (
                f"Cambié '{field}' de requisito obligatorio a deseable."
            ),
            "previous": "hard",
            "new": "soft",
        }
    return None


def _generate_user_message(change_descriptions: list[str]) -> str:
    if not change_descriptions:
        return "Ajusté tu búsqueda para encontrar más opciones."
    llm = ChatOpenAI(model=MODEL, temperature=0)
    bulleted = "\n".join(f"- {d}" for d in change_descriptions)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Cambios realizados:\n{bulleted}"),
        ]
    )
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) else str(content).strip()
