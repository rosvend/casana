"""The top-level `PropertyFinderState` shared by every node in the graph.

Eight logical groups, in pipeline order:
    1. User input layer
    2. Structured requirements
    3. Routing decisions
    4. Parallel fetch outputs
    5. Synthesized candidates
    6. Evaluation result
    7. Softening loop state + history
    8. Final output

LangGraph idiom: `TypedDict` with `total=False` so each node returns only
the keys it wrote; missing keys are simply absent and the framework merges
partial updates. The `messages` field uses LangChain's `add_messages`
reducer so resume-after-interrupt composes cleanly; `softening_history`
uses `operator.add` for a plain append-only list.
"""

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.state.evaluation import EvaluationResult
from src.state.listings import Candidate, Listing, VerifiedListing
from src.state.news import NewsResults
from src.state.requirements import StructuredRequirements
from src.state.softening import SofteningAttempt


class PropertyFinderState(TypedDict, total=False):
    """Shared state object for the Estatia LangGraph.

    Every node reads and writes through this TypedDict. Fields are grouped
    by pipeline stage in the comments below. All keys are optional
    (`total=False`) — a node that only writes one key returns a single-key
    dict.
    """

    # 1. User input layer
    messages: Annotated[list[AnyMessage], add_messages]
    """The conversation. Seeded by the API/CLI with a HumanMessage, then
    appended to by every agent that emits a user-facing turn. Uses
    LangChain's `add_messages` reducer (dedupes by message id and supports
    update-by-id) so resume-after-interrupt composes cleanly. The
    clarification loop in requirements_agent pauses via `interrupt()`; the
    question lives in the interrupt payload, not in state."""

    # 2. Structured requirements
    requirements: StructuredRequirements | None
    """Parsed brief produced by requirements_agent; the contract every
    downstream agent reads to know what to look for / score against /
    relax."""

    requirements_complete: bool
    """Gates `requirements_router` in graph.py. False → loop back to
    requirements_agent for more clarification; True → proceed to router_agent."""

    is_property_search: bool
    """Intent flag set by requirements_agent. True when the user is asking to
    search for properties or update filters; False for greetings, thanks,
    goodbyes, or off-topic chit-chat. Read by `route_requirements` to bypass
    the scraper/synthesizer/evaluator chain and route straight to
    responder_agent. Defaults to True (treated as property search) when
    absent."""

    # 3. Routing decisions
    active_branches: list[str]
    """Which fan-out branches router_agent activated this iteration (e.g.
    ['properties_agent', 'news_agent']). Replaced on each router pass."""

    whatsapp_enabled: bool
    """User toggle for the outbound WhatsApp verification step. Set per-run
    (e.g. by the entrypoint or a UI checkbox). When this key is present it
    wins over the ``WHATSAPP_ENABLED`` environment variable; when absent,
    whatsapp_agent falls back to the env var (default False). False ⇒ the
    node promotes candidates to ``VerifiedListing`` with
    ``availability_confirmed=False`` and a note explaining outreach was
    skipped, keeping the downstream schema uniform."""

    # 4. Parallel fetch outputs 
    raw_listings: list[Listing]
    """Listings scraped by properties_agent. Replaced (not appended) on each
    softening retry so stale results don't accumulate."""

    news_results: NewsResults
    """Area news pre-categorized by news_agent. A dict keyed by NewsCategory
    so the synthesizer can attach the right slice to each candidate's zone."""

    verified_listings: list[VerifiedListing]
    """Listings after whatsapp_agent has attempted availability confirmation.
    Replaced on each retry."""

    # 5. Synthesized candidates 
    candidates: list[Candidate]
    """Merged, deduplicated, news-enriched candidates. Written by the
    synthesizer; read by evaluator_agent and `done_node`. Replaced each
    synthesis pass."""

    # 6. Evaluation result 
    evaluation: EvaluationResult | None
    """Latest evaluator verdict. `evaluation_router` in graph.py reads
    `.passes` to decide done / best_effort / soften."""

    # 7. Softening loop
    softening_attempts: int
    """Counter compared against `max_softening_attempts` (currently 3) by
    `evaluation_router`. Softener_agent is responsible for incrementing it;
    if it forgets, the loop will spin forever."""

    softening_history: Annotated[list[SofteningAttempt], add]
    """Append-only record of every relaxation tried, with the evaluator
    outcome that followed. Uses `add` so each softener pass *adds* a record
    instead of overwriting. The softener reads this in full before each new
    decision so it doesn't repeat unproductive moves."""

    # 8. Final output 
    final_results: list[Candidate]
    """Result set returned to the user. Written by `done_node` (clean match)
    or `best_effort_node` (after retries exhausted)."""

    is_best_effort: bool
    """True if the graph exited via `best_effort_node`; False (or absent) on
    a clean pass. Lets the UI flag partial results."""

    softening_summary: str | None
    """Human-readable recap of what was relaxed, for transparency in the UI
    (e.g. 'Raised price ceiling 15%, broadened zone to include Teusaquillo')."""
