"""Standalone smoke test for ``requirements_node`` (vertical slice).

Drives the front-of-graph node through two real LLM calls — an incomplete
request that should trigger a Spanish clarification, and a rich request that
should extract English snake_case constraints with security-weighted
priorities. Unlike the properties test, this is fast and cheap; it only needs
a valid ``OPENAI_API_KEY`` in ``.env``.

    uv run python -m tests.test_requirements_agent

It verifies the node: gates the clarification loop via ``requirements_complete``,
maps Spanish constraints to the canonical English schema, and emits
``priority_weights`` that sum to 1.0 and reflect the user's emphasis.
"""

from __future__ import annotations

import os
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.requirements_agent import requirements_node
from src.graph.graph import make_memory_checkpointer
from src.state import PropertyFinderState, StructuredRequirements


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _find_constraint(requirements: StructuredRequirements, field: str):
    """Return the first constraint with the given field, or None."""
    return next((c for c in requirements.constraints if c.field == field), None)


def _build_solo_graph():
    """Compile a minimal graph (requirements_agent → END) so ``interrupt()``
    has a runtime context. Returns (compiled_graph, config_for_one_thread)."""
    graph = StateGraph(PropertyFinderState)
    graph.add_node("requirements_agent", requirements_node)
    graph.set_entry_point("requirements_agent")
    graph.add_edge("requirements_agent", END)
    compiled = graph.compile(checkpointer=make_memory_checkpointer())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return compiled, config


def test_incomplete_request() -> bool:
    """A vague request must trigger an interrupt with a Spanish question."""
    compiled, config = _build_solo_graph()
    compiled.invoke(
        {"messages": [HumanMessage(content="Hola, busco un lugar para vivir.")]},
        config=config,
    )

    snapshot = compiled.get_state(config)
    interrupts = list(snapshot.interrupts) if snapshot else []
    question = interrupts[0].value.get("clarification_question") if interrupts else None
    print(f"  clarification_question: {question!r}")

    passed = _check(
        "graph paused with a pending interrupt",
        len(interrupts) > 0,
        f"got {len(interrupts)} interrupt(s)",
    )
    passed &= _check(
        "interrupt payload carries clarification_question string",
        isinstance(question, str) and bool(question.strip()),
    )
    passed &= _check(
        "requirements still None while paused",
        snapshot.values.get("requirements") is None if snapshot else False,
    )
    return passed


def test_complete_request_with_weights() -> bool:
    """A rich request must extract English constraints and security-led weights."""
    state: PropertyFinderState = {
        "messages": [
            HumanMessage(
                content=(
                    "Busco un apto en Laureles, mínimo 2 habitaciones, máximo 3 "
                    "millones de pesos. Para mí lo más importante es que el barrio "
                    "sea muy seguro para mi familia, el precio pasa a segundo plano."
                )
            )
        ]
    }
    result = requirements_node(state)

    passed = _check(
        "requirements_complete is True",
        result.get("requirements_complete") is True,
        f"got {result.get('requirements_complete')!r}",
    )

    requirements = result.get("requirements")
    if not isinstance(requirements, StructuredRequirements):
        _check("requirements is a StructuredRequirements", False,
               f"got {type(requirements).__name__}")
        return False

    print(f"  summary: {requirements.summary!r}")
    for c in requirements.constraints:
        print(f"  constraint: {c.model_dump(exclude_none=True)}")
    print(f"  priority_weights: {requirements.priority_weights}")

    passed &= _check("constraints were extracted", len(requirements.constraints) > 0)

    # Constraint fields must be the canonical snake_case English names.
    allowed = {
        "location", "property_type", "transaction_type", "price", "bedrooms",
        "bathrooms", "parking_lots", "estrato", "area_m2", "zone",
    }
    bad = [c.field for c in requirements.constraints if c.field not in allowed]
    passed &= _check("constraint fields are English snake_case", not bad,
                     f"unexpected fields: {bad}" if bad else "")

    # Laureles should land in a location or zone constraint.
    place = _find_constraint(requirements, "location") or _find_constraint(
        requirements, "zone")
    passed &= _check(
        "location/zone constraint mentions Laureles",
        place is not None and "laureles" in str(place.exact_value).lower(),
        f"got {place.model_dump(exclude_none=True) if place else None}",
    )

    bedrooms = _find_constraint(requirements, "bedrooms")
    passed &= _check(
        "bedrooms constraint has min_value 2",
        bedrooms is not None and bedrooms.min_value == 2,
        f"got {bedrooms.model_dump(exclude_none=True) if bedrooms else None}",
    )

    price = _find_constraint(requirements, "price")
    passed &= _check(
        "price constraint has max_value 3_000_000",
        price is not None and price.max_value == 3_000_000,
        f"got {price.model_dump(exclude_none=True) if price else None}",
    )

    # priority_weights: must sum to 1.0 and favour security over price.
    weights = requirements.priority_weights
    total = sum(weights.values())
    passed &= _check("priority_weights sum to 1.0", abs(total - 1.0) < 1e-6,
                     f"sum={total}")
    passed &= _check(
        "security weighted above price",
        weights.get("security", 0.0) > weights.get("price", 0.0),
        f"security={weights.get('security')} price={weights.get('price')}",
    )
    return passed


def _scenario_zone_only_splits_to_city_and_zone() -> bool:
    """A query that only names a neighborhood must yield BOTH a `location`
    constraint (the parent city) AND a separate `zone` constraint."""
    state: PropertyFinderState = {
        "messages": [
            HumanMessage(
                content=(
                    "Busco apartamento en arriendo en Chapinero, 2 habitaciones, "
                    "presupuesto 2 millones, prioridad seguridad."
                )
            )
        ]
    }
    result = requirements_node(state)

    requirements = result.get("requirements")
    if not isinstance(requirements, StructuredRequirements):
        return _check("zone-split: requirements is StructuredRequirements", False,
                      f"got {type(requirements).__name__}")

    loc = _find_constraint(requirements, "location")
    zone = _find_constraint(requirements, "zone")

    passed = _check(
        "zone-split: location constraint resolves to Bogotá",
        loc is not None
        and isinstance(loc.exact_value, str)
        and "bogota" in loc.exact_value.lower(),
        f"got {loc.model_dump(exclude_none=True) if loc else None}",
    )
    passed &= _check(
        "zone-split: zone constraint mentions Chapinero",
        zone is not None
        and isinstance(zone.exact_value, str)
        and "chapinero" in zone.exact_value.lower(),
        f"got {zone.model_dump(exclude_none=True) if zone else None}",
    )
    passed &= _check(
        "zone-split: both constraints are hard",
        loc is not None
        and zone is not None
        and loc.constraint_type == "hard"
        and zone.constraint_type == "hard",
    )
    return passed


def _scenario_unknown_zone_returns_geocheck_clarification() -> bool:
    """An explicit unknown zone must short-circuit with a zone-specific question.

    Pure unit test on ``_apply_geo_normalization`` — no LLM needed. The
    requirements agent's interrupt path is driven by ``_GeoCheck.ok=False``
    and the clarification string that the function returns.
    """
    from src.agents.requirements_agent import _apply_geo_normalization
    from src.state import Constraint

    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location", exact_value="bogota",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="zone", exact_value="Xenovaria",  # not a real Bogotá zone
                constraint_type="hard", importance="critical",
            ),
        ],
        summary="test",
        priority_weights={"price": 0.34, "location": 0.33, "security": 0.33},
    )
    check = _apply_geo_normalization(requirements)
    passed = _check(
        "unknown zone → _GeoCheck.ok is False",
        check.ok is False,
        f"got ok={check.ok}",
    )
    passed &= _check(
        "unknown zone → clarification mentions the offending zone",
        check.clarification is not None
        and "Xenovaria" in (check.clarification or ""),
        f"got clarification={check.clarification!r}",
    )
    return passed


def _scenario_known_zone_canonicalizes_and_passes() -> bool:
    """A real zone (Chapinero, Bogotá) must canonicalize and return ok."""
    from src.agents.requirements_agent import _apply_geo_normalization
    from src.state import Constraint

    requirements = StructuredRequirements(
        constraints=[
            Constraint(
                field="location", exact_value="Bogotá",
                constraint_type="hard", importance="critical",
            ),
            Constraint(
                field="zone", exact_value="Chapinero",
                constraint_type="hard", importance="critical",
            ),
        ],
        summary="test",
        priority_weights={"price": 0.34, "location": 0.33, "security": 0.33},
    )
    check = _apply_geo_normalization(requirements)
    loc = _find_constraint(requirements, "location")
    zone = _find_constraint(requirements, "zone")
    passed = _check("known zone → ok", check.ok, f"got ok={check.ok}")
    passed &= _check(
        "known zone → location canonicalized to lowercase",
        loc is not None and loc.exact_value == "bogota",
        f"got {loc.exact_value if loc else None}",
    )
    passed &= _check(
        "known zone → zone canonicalized to chapinero",
        zone is not None and isinstance(zone.exact_value, str)
        and "chapinero" in zone.exact_value.lower(),
        f"got {zone.exact_value if zone else None}",
    )
    return passed


def main() -> int:
    print("=== requirements_node ===")
    ok = True
    print("\ntest_incomplete_request:")
    ok &= test_incomplete_request()
    print("\ntest_complete_request_with_weights:")
    ok &= test_complete_request_with_weights()
    print("\ntest_zone_only_splits_to_city_and_zone:")
    ok &= _scenario_zone_only_splits_to_city_and_zone()
    print("\ntest_unknown_zone_returns_geocheck_clarification:")
    ok &= _scenario_unknown_zone_returns_geocheck_clarification()
    print("\ntest_known_zone_canonicalizes_and_passes:")
    ok &= _scenario_known_zone_canonicalizes_and_passes()
    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return 0 if ok else 1


def test_zone_only_splits_to_city_and_zone() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        import pytest

        pytest.skip("OPENAI_API_KEY not set")
    assert _scenario_zone_only_splits_to_city_and_zone()


def test_unknown_zone_returns_geocheck_clarification() -> None:
    assert _scenario_unknown_zone_returns_geocheck_clarification()


def test_known_zone_canonicalizes_and_passes() -> None:
    assert _scenario_known_zone_canonicalizes_and_passes()


if __name__ == "__main__":
    raise SystemExit(main())
