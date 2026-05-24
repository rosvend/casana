from langgraph.graph import START, END, StateGraph
from src.state import PropertyFinderState
from src.agents import (
    requirements_agent,
    properties_agent,
    news_agent,
    whatsapp_agent,
    synthesizer_agent,
    evaluator_agent,
    softener_agent,
    responder_agent,
)

"""Roles and responsibilities of each agent:

1. requirements_agent:

This agent is responsible for gathering and understanding the requirements of the user.
It will transform the user's input into a structured format that can be easily processed by other agents.

2. properties_agent:

This agent will scrape real estate listing websites to find properties that match the user's requirements. It will gather information such as price,
location, size, and other relevant details about the properties.

3. news_agent:

This agent is responsible for fetching and providing the latest news related to the real estate market in the user's area of interest such as
security, events and other relevant information.

4. whatsapp_agent: (This agent only runs with the top chosen properties that passed the evaluation)
This agent will handle communication with the real estate agency or landlord that published the listing via WhatsApp with phone numbers listed
to check that the listing is still available. It will send an outbounding message to the contact number provided in the listing and wait for a response for about 60 seconds.
If the agent was able to get valuable information from the contact's answer, it will tell the user about it. Otherwise, it will just inform the user that no response was received.

5. evaluator_agent:

This agent will evaluate the information provided by other agents and determine if it meets the user's requirements.
If the information does not meet the requirements, it will provide feedback so that the softener_agent can relax constraints
until the user's requirements are satisfied or the retry budget is exhausted.

6. softener_agent:

This agent will be responsible for softening the constraints of the user's requirements if the evaluator_agent determines
that the current requirements are too strict and cannot be met with the available information. Each softening attempt will pass the memory of why it failed
to meet the requirements.

7. synthesizer_agent:
This agent will take all the information gathered from the properties_agent and the news agent and synthesize it into a coherent response
that can be evaluated by the evaluator_agent. It will also merge listings, news and run verification results into a single candidate list for the evaluator_agent to assess.

"""

MAX_SOFTENING_ATTEMPTS = 3


def route_requirements(state: PropertyFinderState) -> list[str]:
    """Fan out to the two parallel branches. Deterministic for MVP."""
    return ["properties_agent", "news_agent"]


def route_evaluation(state: PropertyFinderState) -> str:
    """After evaluation: deliver via WhatsApp, soften and retry, or give up.

    Both terminal outcomes (success → whatsapp → responder, give-up → responder)
    converge through ``responder_agent`` before ``END``.
    """
    evaluation = state["evaluation"]
    if evaluation.passes:
        return "whatsapp_agent"
    if state.get("softening_attempts", 0) < MAX_SOFTENING_ATTEMPTS:
        return "softener_agent"
    return "responder_agent"


def build_graph():
    graph = StateGraph(PropertyFinderState)

    graph.add_node("requirements_agent", requirements_agent)
    graph.add_node("properties_agent", properties_agent)
    graph.add_node("news_agent", news_agent)
    graph.add_node("synthesizer_agent", synthesizer_agent)
    graph.add_node("evaluator_agent", evaluator_agent)
    graph.add_node("softener_agent", softener_agent)
    graph.add_node("whatsapp_agent", whatsapp_agent)
    graph.add_node("responder_agent", responder_agent)

    graph.set_entry_point("requirements_agent")

    graph.add_conditional_edges(
        "requirements_agent",
        route_requirements,
        ["properties_agent", "news_agent"],
    )

    graph.add_edge("properties_agent", "synthesizer_agent")
    graph.add_edge("news_agent", "synthesizer_agent")

    graph.add_edge("synthesizer_agent", "evaluator_agent")

    graph.add_conditional_edges(
        "evaluator_agent",
        route_evaluation,
        {
            "whatsapp_agent": "whatsapp_agent",
            "softener_agent": "softener_agent",
            "responder_agent": "responder_agent",
        },
    )

    graph.add_conditional_edges(
        "softener_agent",
        route_requirements,
        ["properties_agent", "news_agent"],
    )

    graph.add_edge("whatsapp_agent", "responder_agent")
    graph.add_edge("responder_agent", END)

    return graph.compile()
