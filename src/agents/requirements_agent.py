"""`requirements_node` — the front-of-graph LangGraph node.

This node turns a free-text user request into a structured brief. It sits at
the entrypoint of the graph and gates the clarification loop via LangGraph's
native ``interrupt()``: when the LLM judges the request incomplete, the node
pauses; the API/CLI surfaces the question; resuming with
``Command(resume=user_reply)`` re-enters this node with the reply appended
to ``state["messages"]``. ``route_requirements`` self-loops the node until
``requirements_complete`` flips to True.

The work is a single LLM call with structured output:

1. The LLM judges whether the request carries enough to run a search — at a
   minimum a **location** plus a **budget or transaction type**.
2. If not, the node calls ``interrupt({"clarification_question": ...})``;
   on resume the assistant's question and the user's reply are both
   appended to ``messages`` and the router loops back.
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
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.state import Constraint, PropertyFinderState, StructuredRequirements
from src.utils.geography import (
    KNOWN_ZONES,
    canonical_location,
    canonical_zone,
    normalize_geography,
)


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

    is_property_search: bool = Field(
        ...,
        description=(
            "REQUIRED. Set False for any greeting, thanks, goodbye, or "
            "small-talk turn that doesn't change the search (e.g. 'hola', "
            "'gracias', 'ok perfecto', 'chao', 'ya no', 'no así está bien'). "
            "Set True only when the user is asking to search for properties, "
            "modify filters, or clarify something about an active search."
        ),
    )
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

DECIDE PRIMERO SI ES UNA BÚSQUEDA (is_property_search):
- True cuando el usuario pide buscar propiedades, modificar filtros, o aclarar
  algo sobre su búsqueda activa.
- False cuando el usuario solo saluda, agradece, se despide o hace charla
  casual ("hola", "gracias", "ok perfecto", "chao", "ya no", "no, así está
  bien"). En este caso NO llenes extracted_requirements y deja is_complete
  como estaba; la pregunta de clarificación tampoco aplica.

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

UBICACIÓN VS. ZONA — REGLA CRÍTICA:
- location es SIEMPRE una ciudad o municipio colombiano (Bogotá, Medellín,
  Cali, Barranquilla, Bucaramanga, Cartagena, etc.).
- zone es un barrio, comuna o UPZ dentro de esa ciudad (Chapinero, Usaquén,
  El Poblado, Laureles, Bocagrande…).
- NUNCA combines ciudad y barrio en un solo string. NUNCA uses guiones bajos
  ni comas para unirlos en un mismo valor.

Ejemplos correctos:
  Usuario: "Busco apto en Chapinero, Bogotá, 3M arriendo"
    → constraints incluye DOS entradas separadas:
       { field: "location", exact_value: "bogota", constraint_type: "hard" }
       { field: "zone",     exact_value: "chapinero", constraint_type: "hard" }
  Usuario: "Apartamento en El Poblado de Medellín, venta hasta 500M"
    → { field: "location", exact_value: "medellin" }
       { field: "zone",     exact_value: "el poblado" }
       { field: "transaction_type", exact_value: "sale" }
       { field: "price", max_value: 500000000 }

Ejemplos INCORRECTOS (NO HAGAS ESTO):
  ✗ location="chapinero_bogota"     ← NUNCA unas con _
  ✗ location="chapinero, bogota"    ← NUNCA unas con coma
  ✗ location="chapinero"            ← Chapinero es zona, no ciudad
  ✗ zone="bogota"                   ← Bogotá es ciudad, no zona
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


def _apply_geo_normalization(requirements: StructuredRequirements) -> bool:
    """Normalize / split / synthesize the location and zone constraints.

    Mutates ``requirements.constraints`` in place and returns whether the
    geography is usable downstream:

    - ``True``  — the brief either has a canonical location, or no location
      at all and we couldn't infer one (caller decides what to do next).
    - ``False`` — the LLM extracted a location string that we can't resolve
      to a DANE municipality, a KNOWN_ZONES parent city, or a substring
      match. The caller should ask the user for clarification rather than
      run a scrape on garbage.

    Behaviors:

    - Sub-municipal ``location`` (e.g. ``"chapinero"`` or ``"laureles
      medellin"``) is split into ``location=parent_city`` plus a synthesized
      ``zone`` if the LLM didn't already supply one.
    - If the LLM extracted only a ``zone`` that maps to a known parent city,
      a hard ``location`` constraint is synthesized so the evaluator's
      hard-constraint gate has something to reject wrong-city listings with.
    - Every location/zone string that survives canonicalization is canonical
      (lowercase, accent-stripped, validated against the DANE list) so every
      downstream consumer can compare them by simple equality.
    """
    constraints = requirements.constraints
    loc_constraint = next((c for c in constraints if c.field == "location"), None)

    # Branch A: no location constraint from the LLM.
    if loc_constraint is None or not isinstance(loc_constraint.exact_value, str):
        # Still canonicalize any pre-existing zone constraint emitted by the LLM.
        for c in constraints:
            if c.field == "zone" and isinstance(c.exact_value, str):
                canon = canonical_zone(c.exact_value)
                if canon:
                    c.exact_value = canon
        # If a zone constraint matches a known parent city, synthesize the
        # missing location so the evaluator can actually gate.
        zone_constraint = next(
            (
                c
                for c in constraints
                if c.field == "zone" and isinstance(c.exact_value, str)
            ),
            None,
        )
        if zone_constraint is not None:
            parent_city = KNOWN_ZONES.get(zone_constraint.exact_value)
            if parent_city:
                constraints.append(Constraint(
                    field="location",
                    exact_value=parent_city,
                    constraint_type="hard",
                    importance="critical",
                ))
                logger.info(
                    "_apply_geo_normalization: synthesized location=%r from zone=%r",
                    parent_city,
                    zone_constraint.exact_value,
                )
        return True

    # Branch B: the LLM gave us a location. Try to canonicalize it.
    result = normalize_geography(loc_constraint.exact_value)
    canon_location = canonical_location(result["location"]) or canonical_location(
        loc_constraint.exact_value
    )
    if canon_location is None:
        logger.warning(
            "_apply_geo_normalization: could not canonicalize location=%r",
            loc_constraint.exact_value,
        )
        return False
    loc_constraint.exact_value = canon_location

    # Canonicalize a pre-existing zone constraint regardless of whether the
    # location string itself triggered a split.
    for c in constraints:
        if c.field == "zone" and isinstance(c.exact_value, str):
            canon = canonical_zone(c.exact_value)
            if canon:
                c.exact_value = canon

    if result["zone"] is None:
        return True

    if any(c.field == "zone" for c in constraints):
        return True
    canon_zone = canonical_zone(result["zone"])
    if not canon_zone:
        return True
    constraints.append(Constraint(
        field="zone",
        exact_value=canon_zone,
        constraint_type="hard",
        importance="critical",
    ))
    return True


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


def _pause_for_clarification(question: str) -> dict:
    """Pause the graph and resume with the user's reply appended to messages.

    LangGraph re-runs the node on resume; the first call raises
    ``GraphInterrupt`` to halt execution, and the second call (after
    ``Command(resume=reply)``) returns ``reply`` from ``interrupt()`` so
    we can append both turns to ``messages`` and let the router self-loop
    back here for a fresh extraction.
    """
    user_reply = interrupt({"clarification_question": question})
    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=user_reply),
        ],
        "requirements_complete": False,
        "is_property_search": True,
        "requirements": None,
    }


def requirements_node(state: PropertyFinderState) -> dict:
    """Parse the conversation into a structured brief (or ask for more info).

    Reads ``state["messages"]`` and returns the state update:
    ``requirements_complete`` gates the loop, ``requirements`` carries the
    parsed brief when complete. When clarification is needed, the node
    pauses via :func:`interrupt`; resuming with the user reply re-runs the
    node with the fuller context.
    """
    messages = list(state.get("messages") or [])

    llm = ChatOpenAI(model=MODEL, temperature=0)
    # function_calling (not strict json_schema) — StructuredRequirements has an
    # open dict[str, float] priority_weights, which strict mode rejects.
    structured_llm = llm.with_structured_output(
        RequirementsExtraction, method="function_calling"
    )

    prompt: list = [("system", SYSTEM_PROMPT), *messages]
    result: RequirementsExtraction = structured_llm.invoke(prompt)
    logger.info(
        "requirements_node: is_property_search=%s is_complete=%s",
        result.is_property_search,
        result.is_complete,
    )

    if not result.is_property_search:
        # Chit-chat / greeting / thanks — route straight to responder. The
        # responder handles the assistant turn so we don't pre-write one here.
        return {"is_property_search": False}

    if result.is_complete and result.extracted_requirements is not None:
        requirements = result.extracted_requirements
        requirements.priority_weights = _normalize_weights(requirements.priority_weights)
        geo_ok = _apply_geo_normalization(requirements)
        if not geo_ok:
            bad_loc = next(
                (
                    c.exact_value
                    for c in requirements.constraints
                    if c.field == "location" and isinstance(c.exact_value, str)
                ),
                None,
            )
            question = (
                f"No reconocí '{bad_loc}' como una ciudad colombiana. "
                "¿Podrías indicarme exactamente en qué ciudad quieres buscar? "
                "Por ejemplo: Medellín, Bogotá, Cali, Barranquilla, Bucaramanga…"
                if bad_loc
                else (
                    "¿En qué ciudad colombiana quieres buscar? Por ejemplo: "
                    "Medellín, Bogotá, Cali, Barranquilla…"
                )
            )
            logger.info(
                "requirements_node: location %r couldn't be canonicalized — clarifying",
                bad_loc,
            )
            return _pause_for_clarification(question)

        logger.info(
            "requirements_node: extracted %d constraint(s), weights=%s",
            len(requirements.constraints), requirements.priority_weights,
        )
        # Start of a new property search: wipe prior-turn results so stale
        # candidates/evaluation from earlier turns in the same thread don't
        # leak into the new run. ``messages`` is intentionally not reset —
        # the add_messages reducer keeps the multi-turn dialogue intact.
        # ``softening_history`` also can't be reset from a node return.
        return {
            "requirements_complete": True,
            "is_property_search": True,
            "requirements": requirements,
            "candidates": [],
            "evaluation": None,
            "softening_attempts": 0,
            "raw_listings": [],
            "verified_listings": [],
            "news_results": {},
            "final_results": [],
            "is_best_effort": False,
            "softening_summary": None,
        }

    # Incomplete — pause the graph and wait for the user's reply.
    question = result.clarification_question or (
        "¿Me cuentas un poco más sobre lo que buscas? Por ejemplo, en qué zona "
        "y cuál es tu presupuesto."
    )
    logger.info("requirements_node: clarification needed — %s", question)
    return _pause_for_clarification(question)
