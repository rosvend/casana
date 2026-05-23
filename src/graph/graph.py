from langgraph.graph import START, END, StateGraph
from typing import Literal
from src.state import PropertyFinderState
from src.agents import (
    requirements_agent,
    router_agent,
    properties_agent,
    news_agent,
    whatsapp_agent,
    synthesizer_agent,
    evaluator_agent,
    softener_agent
    )

"""Roles and responsibilities of each agent:

1. requirements_agent: 

This agent is responsible for gathering and understanding the requirements of the user. 
It will transform the user's input into a structured format that can be easily processed by other agents.

2. router_agent: 

This agent will analyze the structured requirements provided by the requirements_agent and determine which other agents need to be involved in fulfilling the user's request. 
It will route the information to the appropriate agents based on the nature of the request.

3. properties_agent: 

This agent will scrape real estate listing websites to find properties that match the user's requirements. It will gather information such as price, 
location, size, and other relevant details about the properties.

4. news_agent: 

This agent is responsible for fetching and providing the latest news related to the real estate market in the user's area of interest such as 
security, events and other relevant information.

5. whatsapp_agent: This agent will handle communication with the user via WhatsApp with phone numbers listed to check that the listing is still available. 

6. evaluator_agent: 

This agent will evaluate the information provided by other agents and determine if it meets the user's requirements. 
If the information does not meet the requirements, it will provide feedback to the router_agent to adjust the routing of information accordingly 
until the user's requirements are satisfied.

7. softener_agent: 

This agent will be responsible for softening the constraints of the user's requirements if the evaluator_agent determines
that the current requirements are too strict and cannot be met with the available information.

8. synthesizer_agent:
This agent will take all the information gathered from the properties_agent and the news agent and synthesize it into a coherent response
that can be evaluated by the evaluator_agent. It will also merge listings, news and run verification results into a single candidate list for the evaluator_agent to assess.

"""

max_softening_attempts = 3

def requirements_router(state: PropertyFinderState) -> Literal["router_agent", "requirements_agent"]:
    """Loop back to requirements_agent if more info is needed from the user."""
    return "router_agent" if state.get("requirements_complete") else "requirements_agent"


def evaluation_router(
    state: PropertyFinderState,
) -> Literal["done", "best_effort", "softener_agent"]:
    """After evaluation: succeed, give up, or soften and retry."""
    evaluation = state.get("evaluation", {}) #TODO: Change to state["evaluation"].passes 
    if evaluation.get("passes"):
        return "done"
    if state.get("softening_attempts", 0) >= max_softening_attempts:
        return "best_effort"
    return "softener_agent"


def done_node(state: PropertyFinderState) -> dict:
    return {"final_results": state["candidates"]}


def best_effort_node(state: PropertyFinderState) -> dict:
    # Return whatever we have, flagged as partial
    return {"final_results": state.get("candidates", [])}

def build_graph():
    graph = StateGraph(PropertyFinderState)

    graph.add_node("requirements_agent", requirements_agent)
    graph.add_node("router_agent", router_agent)
    graph.add_node("properties_agent", properties_agent)
    graph.add_node("news_agent", news_agent)
    graph.add_node("whatsapp_agent", whatsapp_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("evaluator_agent", evaluator_agent)
    graph.add_node("softener_agent", softener_agent)
    graph.add_node("done", done_node)
    graph.add_node("best_effort", best_effort_node)

    graph.add_edge(START, "requirements_agent")
    graph.add_conditional_edges(
        "requirements_agent",
        requirements_router,
        {
            "requirements_agent": "requirements_agent",  # loop for clarification
            "router_agent": "router_agent",
        },
    )

    graph.add_edge("router_agent", "properties_agent")
    graph.add_edge("router_agent", "news_agent")

    # whatsapp_agent depends on listings
    graph.add_edge("properties_agent", "whatsapp_agent")
    graph.add_edge("whatsapp_agent", "synthesizer")
    graph.add_edge("news_agent", "synthesizer")

    graph.add_edge("synthesizer", "evaluator_agent")

    # Loop pattern: evaluator decides done / give up / soften
    graph.add_conditional_edges(
        "evaluator_agent",
        evaluation_router,
        {
            "done": "done",
            "best_effort": "best_effort",
            "softener_agent": "softener_agent",
        },
    )

    graph.add_edge("softener_agent", "router_agent")
    graph.add_edge("done", END)
    graph.add_edge("best_effort", END)

    return graph.compile()