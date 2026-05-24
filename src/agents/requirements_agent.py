"""`requirements_node` — the front-of-graph LangGraph node.

This node turns a free-text user request into a structured brief. It sits at
the entrypoint of the graph and gates the clarification self-loop:
``requirements_router`` keeps routing back here until
``state["requirements_complete"]`` is true.

The work is a single LLM call with structured output:

1. The LLM judges whether the request carries enough to run a search — at a
   minimum a **location** plus a **budget or transaction type**.
2. If not, it writes a friendly ``clarification_question`` in Colombian
   Spanish asking only for what is missing.
3. If so, it maps the request into a :class:`StructuredRequirements`: a flat
   list of :class:`Constraint` objects keyed by snake_case English ``field``
   names that mirror the :class:`Listing` model, plus ``priority_weights``
   over the ``price`` / ``location`` / ``security`` axes.

``priority_weights`` must sum to exactly 1.0; since LLMs rarely land that
exactly, :func:`_normalize_weights` rescales the dict before it leaves the
node, so the invariant holds regardless of model drift.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.state import Constraint, PropertyFinderState, StructuredRequirements
from src.utils.geography import canonical_location, canonical_zone, normalize_geography


load_dotenv()

logger = logging.getLogger(__name__)
MODEL = "gpt-5-nano"

#: Default macro-priority split, mirroring StructuredRequirements.priority_weights.
#: Used as the fallback when the LLM returns an empty or zero-sum weight dict.
_DEFAULT_WEIGHTS: dict[str, float] = {"price": 0.34, "location": 0.33, "security": 0.33}


class RequirementsExtraction(BaseModel):
    """The schema the LLM fills in for one requirements-parsing turn.

    Exactly one of ``clarification_question`` / ``extracted_requirements`` is
    meaningful, selected by ``is_complete``.
    """

    is_complete: bool = Field(
        ...,
        description=(
            "True only if the request carries enough to run a property search "
            "— at minimum a location plus a budget or transaction type."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "When is_complete is False: a short, friendly question in Colombian "
            "Spanish asking the user for the missing information."
        ),
    )
    extracted_requirements: StructuredRequirements | None = Field(
        default=None,
        description="When is_complete is True: the parsed structured brief.",
    )


SYSTEM_PROMPT = """\
Eres un asistente inmobiliario colombiano, amable y cercano. Tu trabajo es
convertir la solicitud del usuario en un brief estructurado para un motor de
búsqueda de propiedades.

DECIDE SI HAY SUFICIENTE INFORMACIÓN (is_complete):
- Marca is_complete=True solo si la solicitud tiene, como mínimo, una UBICACIÓN
  y además un PRESUPUESTO o un TIPO DE TRANSACCIÓN (arriendo o venta).
- Si falta algo esencial, marca is_complete=False y escribe en
  clarification_question una pregunta corta y amable, en español colombiano,
  pidiendo SOLO lo que falta.

CUANDO is_complete=True, llena extracted_requirements:
- constraints: una lista plana de restricciones. Cada constraint usa un nombre
  de campo en inglés, snake_case, tomado EXCLUSIVAMENTE de esta lista:
  location, property_type, transaction_type, price, bedrooms, bathrooms,
  parking_lots, estrato, area_m2, zone.
  * Usa exact_value para valores únicos (p. ej. location="medellin",
    transaction_type="rent", property_type="apartment").
  * Usa min_value / max_value para rangos (p. ej. "mínimo 2 habitaciones" ->
    field="bedrooms", min_value=2; "máximo 3 millones" -> field="price",
    max_value=3000000).
  * Traduce los valores al inglés cuando aplique: arriendo->rent, venta->sale,
    apartamento/apto->apartment, casa->house.
  * constraint_type: "hard" si su incumplimiento descarta la propiedad;
    "soft" si solo afecta el puntaje.
  * importance: "critical", "important" o "nice_to_have" según el énfasis del
    usuario.
- summary: un resumen de una línea, en español, del brief.
- priority_weights: pesos macro sobre los ejes 'price', 'location' y
  'security'. DEBEN sumar exactamente 1.0. Usa siempre las llaves en inglés
  (seguridad->security, precio->price, ubicación/zona->location).
  Cuando el usuario expresa una prioridad clara DEBES desplazar los pesos de
  forma marcada — NUNCA dejes los tres valores casi iguales en ese caso.
  Ejemplo: para "lo más importante es que el barrio sea muy seguro para mi
  familia, el precio pasa a segundo plano" un reparto correcto es
  security=0.6, location=0.25, price=0.15.
  Solo si el usuario no expresa ninguna preferencia usa el reparto por defecto
  0.34 / 0.33 / 0.33.
"""


def _apply_geo_normalization(requirements: StructuredRequirements) -> None:
    """Split a sub-municipal ``location`` into (``location=parent_city``, ``zone=...``).

    Mutates ``requirements.constraints`` in place. If a ``zone`` constraint
    already exists (the LLM did the split itself), only the location side is
    rewritten — we never append a duplicate zone.

    Every location/zone string that leaves this function is canonical
    (lowercase, accent-stripped, validated against the DANE list) so every
    downstream consumer can compare them by simple equality.
    """
    constraints = requirements.constraints
    loc_constraint = next((c for c in constraints if c.field == "location"), None)
    if loc_constraint is None or not isinstance(loc_constraint.exact_value, str):
        # Still canonicalize any pre-existing zone constraint emitted by the LLM.
        for c in constraints:
            if c.field == "zone" and isinstance(c.exact_value, str):
                canon = canonical_zone(c.exact_value)
                if canon:
                    c.exact_value = canon
        return

    result = normalize_geography(loc_constraint.exact_value)
    canon_location = canonical_location(result["location"]) or canonical_location(
        loc_constraint.exact_value
    )
    if canon_location:
        loc_constraint.exact_value = canon_location

    # Canonicalize a pre-existing zone constraint regardless of whether the
    # location string itself triggered a split.
    for c in constraints:
        if c.field == "zone" and isinstance(c.exact_value, str):
            canon = canonical_zone(c.exact_value)
            if canon:
                c.exact_value = canon

    if result["zone"] is None:
        return

    if any(c.field == "zone" for c in constraints):
        return
    canon_zone = canonical_zone(result["zone"])
    if not canon_zone:
        return
    constraints.append(Constraint(
        field="zone",
        exact_value=canon_zone,
        constraint_type="hard",
        importance="critical",
    ))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Rescale ``weights`` so the values sum to exactly 1.0.

    Falls back to :data:`_DEFAULT_WEIGHTS` if the LLM returned an empty dict or
    one whose values sum to zero or less — guaranteeing the
    ``priority_weights`` invariant downstream agents rely on.
    """
    total = sum(weights.values()) if weights else 0.0
    if total <= 0:
        return dict(_DEFAULT_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def requirements_node(state: PropertyFinderState) -> dict:
    """Parse the user request into a structured brief (or ask for more info).

    Reads ``state["user_query"]`` (and any prior ``chat_history`` turns) and
    returns the state update: ``requirements_complete`` gates the clarification
    loop, ``requirements`` carries the parsed brief when complete, and
    ``clarification_question`` carries the next question when it is not.
    """
    user_query = state.get("user_query", "") or ""

    llm = ChatOpenAI(model=MODEL, temperature=0)
    # function_calling (not strict json_schema) — StructuredRequirements has an
    # open dict[str, float] priority_weights, which strict mode rejects.
    structured_llm = llm.with_structured_output(
        RequirementsExtraction, method="function_calling"
    )

    # Replay prior clarification turns so a looped run keeps multi-turn context.
    messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    for turn in state.get("chat_history") or []:
        messages.append((turn.get("role", "user"), turn.get("content", "")))
    messages.append(("human", user_query))

    result: RequirementsExtraction = structured_llm.invoke(messages)
    logger.info("requirements_node: is_complete=%s", result.is_complete)

    # The user's turn is always recorded; chat_history uses an `add` reducer.
    chat_history: list[dict[str, str]] = [{"role": "user", "content": user_query}]

    if result.is_complete and result.extracted_requirements is not None:
        requirements = result.extracted_requirements
        requirements.priority_weights = _normalize_weights(requirements.priority_weights)
        _apply_geo_normalization(requirements)
        logger.info(
            "requirements_node: extracted %d constraint(s), weights=%s",
            len(requirements.constraints), requirements.priority_weights,
        )
        return {
            "chat_history": chat_history,
            "requirements_complete": True,
            "requirements": requirements,
            "clarification_question": None,
        }

    # Incomplete — ask for more, and record the question for the next loop.
    question = result.clarification_question or (
        "¿Me cuentas un poco más sobre lo que buscas? Por ejemplo, en qué zona "
        "y cuál es tu presupuesto."
    )
    chat_history.append({"role": "assistant", "content": question})
    logger.info("requirements_node: clarification needed — %s", question)
    return {
        "chat_history": chat_history,
        "requirements_complete": False,
        "requirements": None,
        "clarification_question": question,
    }
