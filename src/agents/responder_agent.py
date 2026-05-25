"""`responder_node` — the final terminal node.

Runs after either ``whatsapp_agent`` (success path) or directly after the
evaluator gave up (softening budget exhausted). Builds a deterministic text
summary of the run's outcome, hands it to ``gpt-5-nano`` for a friendly
Colombian-Spanish Markdown reply, and appends the reply to ``messages``
as an ``AIMessage``.

The summary is built in pure Python so the LLM never has to invent values:
all property facts come straight from ``Candidate.listing`` / ``match_score``,
and the system prompt explicitly forbids inventing properties.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.state import Candidate, EvaluationResult, PropertyFinderState

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gpt-5-nano"
TOP_N = 3

_SYSTEM_PROMPT = (
    "Eres un asesor inmobiliario experto en Colombia. Tu objetivo es presentar "
    "los resultados finales de una búsqueda de propiedades. Usa Markdown, "
    "viñetas para las propiedades y un tono amable y coloquial (español "
    "colombiano). NUNCA inventes propiedades; usa estrictamente los datos "
    "provistos. Sé conciso."
)

_CHITCHAT_SYSTEM_PROMPT = (
    "Eres un asistente inmobiliario en Colombia. El usuario te está saludando, "
    "agradeciéndote o despidiéndose, o solo conversando. Responde brevemente, "
    "con amabilidad y en español colombiano. No menciones propiedades a menos "
    "que el usuario lo pida explícitamente."
)


def responder_node(state: PropertyFinderState) -> dict:
    """Produce the final user-facing reply and append it to ``messages``.

    Reads ``candidates``, ``evaluation``, and ``softening_attempts`` from the
    state. Returns ``{"messages": [AIMessage(content=...)]}``  — a
    single-element list, which the ``add_messages`` reducer appends to
    whatever conversation is already there.

    When ``is_property_search`` is ``False`` (set by requirements_agent for
    greetings / thanks / goodbyes) the node short-circuits to a brief
    chit-chat reply, bypassing the candidate/evaluation summary entirely so
    stale prior-turn candidates aren't re-served.
    """
    if state.get("is_property_search") is False:
        return _respond_chitchat(state.get("messages") or [])

    candidates: list[Candidate] = state.get("candidates") or []
    evaluation: EvaluationResult | None = state.get("evaluation")
    softening_attempts: int = state.get("softening_attempts", 0)

    success = (
        evaluation is not None and evaluation.passes is True and len(candidates) > 0
    )

    if success:
        summary = _build_success_summary(candidates[:TOP_N])
        logger.info(
            "responder_node: success path with %d candidate(s)", len(candidates)
        )
    else:
        summary = _build_failure_summary(evaluation, softening_attempts)
        logger.info(
            "responder_node: failure path (attempts=%d, evaluation=%s)",
            softening_attempts,
            "present" if evaluation else "missing",
        )

    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=summary),
        ]
    )

    return {"messages": [AIMessage(content=response.content)]}


def _respond_chitchat(history: list) -> dict:
    """Generate a brief friendly reply from the conversation ``messages``."""
    prompt: list = [SystemMessage(content=_CHITCHAT_SYSTEM_PROMPT)]
    for message in history:
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
            prompt.append(message)

    llm = ChatOpenAI(model=MODEL, temperature=0.3)
    response = llm.invoke(prompt)
    logger.info("responder_node: chit-chat reply")
    return {"messages": [AIMessage(content=response.content)]}


def _build_success_summary(top: list[Candidate]) -> str:
    """Render the top candidates as a fact block the LLM can reformat."""
    lines = [
        "RESULTADO DE LA BÚSQUEDA: ÉXITO",
        f"Se encontraron {len(top)} propiedad(es) que cumplen los requisitos.",
        "",
        "PROPIEDADES (en orden de mejor match):",
    ]

    for index, candidate in enumerate(top, start=1):
        listing = candidate.listing
        url = _pick_url(candidate)
        availability_confirmed = getattr(listing, "availability_confirmed", False)
        availability_line = (
            "Disponibilidad: el propietario confirmó disponibilidad por WhatsApp"
            if availability_confirmed
            else "Disponibilidad: no confirmada por WhatsApp"
        )

        lines.extend(
            [
                "",
                f"Propiedad {index}:",
                f"  - URL: {url}",
                f"  - Precio: {_format_price(listing.price)}",
                f"  - Zona: {listing.zone or 'no especificada'}",
                f"  - Área: {_format_area(listing.area_m2)}",
                f"  - Habitaciones: {listing.bedrooms if listing.bedrooms is not None else 'no especificado'}",
                f"  - Match score: {candidate.match_score:.2f}",
                f"  - {availability_line}",
            ]
        )

    lines.extend(
        [
            "",
            "Presenta estas propiedades al usuario en Markdown, con viñetas, "
            "incluyendo todos los datos anteriores. Si alguna propiedad tiene "
            "disponibilidad confirmada por WhatsApp, dilo explícitamente.",
        ]
    )
    return "\n".join(lines)


def _build_failure_summary(
    evaluation: EvaluationResult | None, softening_attempts: int
) -> str:
    """Render a failure context block for the LLM to wrap in friendly prose."""
    lines = [
        "RESULTADO DE LA BÚSQUEDA: SIN RESULTADOS",
        (
            f"Tras {softening_attempts} intento(s) de relajar los requisitos "
            "(presupuesto, zona, etc.), no se encontraron propiedades que "
            "satisfagan los criterios."
        ),
    ]

    if evaluation is not None and evaluation.aggregate_failure_reasons:
        lines.append("")
        lines.append("Principales restricciones incumplidas:")
        for reason in evaluation.aggregate_failure_reasons:
            lines.append(
                f"  - {reason.constraint_field}: esperaba {reason.expected}, "
                f"se encontró {reason.actual}"
            )

    lines.extend(
        [
            "",
            "Explícale al usuario, con amabilidad, que no se encontraron "
            "propiedades que cumplan todos los requisitos a pesar de haber "
            "relajado las restricciones varias veces. Sugiérele revisar el "
            "presupuesto o ampliar la zona de búsqueda. No inventes "
            "propiedades.",
        ]
    )
    return "\n".join(lines)


def _pick_url(candidate: Candidate) -> str:
    """Prefer ``listing.url``, fall back to first ``source_urls``, then placeholder."""
    if candidate.listing.url:
        return candidate.listing.url
    if candidate.source_urls:
        return candidate.source_urls[0]
    return "(URL no disponible)"


def _format_price(price: float | None) -> str:
    if price is None:
        return "no especificado"
    return f"${price:,.0f} COP"


def _format_area(area: float | None) -> str:
    if area is None:
        return "no especificada"
    return f"{area:g} m²"
