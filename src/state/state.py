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
partial updates. `Annotated[list, add]` is applied where the field must
*accumulate across iterations* of a loop, not be overwritten — see the note
on `chat_history` and `softening_history` below.
"""

from operator import add
from typing import Annotated, TypedDict

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
    user_query: str
    """The raw user request. Written by the graph entrypoint; read by
    requirements_agent."""

    chat_history: Annotated[list[dict[str, str]], add]
    """Conversational turns accumulated during the requirements clarification
    loop. Uses `add` so each requirements_agent self-loop iteration appends
    rather than overwrites. Each entry is a {'role': ..., 'content': ...} dict."""

    clarification_question: str | None
    """The next question requirements_agent wants to ask the user. Written
    by requirements_agent when `requirements_complete` is False; consumed by
    the user-facing layer (out of graph scope) and cleared on the next pass."""

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
