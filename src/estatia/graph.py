from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from estatia.config import Settings
from estatia.models import AgentState, Requirement, Property, NewsItem, Proposal
from estatia.services import Services

logger = logging.getLogger("estatia.graph")


def build_graph(services: Services, settings: Settings):
    graph = StateGraph(AgentState)

    def intake_node(state: AgentState) -> dict:
        logger.info("Node intake:start")
        requirements = services.intake.parse_request(state.user_text)
        logger.info("Node intake:done requirements_found=%s", len(requirements) if requirements else 0)
        return {
            "requirements": requirements,
            "retries": state.retries,
        }

    def coordinator_node(state: AgentState) -> dict:
        logger.info("Node coordinator:start")
        run_news = False
        if state.requirements:
            # Example logic: if the location lacks specific details or is broad
            location = state.requirements[0].location.lower()
            if "bogota" in location or "medellin" in location:
                run_news = True
        logger.info("Node coordinator:done run_news=%s retries=%s", run_news, state.retries)
        return {
            "run_news": run_news,
        }

    def scraping_node(state: AgentState) -> dict:
        logger.info("Node scraper:start")
        properties = []
        feedback = ""
        
        if state.requirements:
            properties = services.listing.search(state.requirements[0])
            
        logger.info("Node scraper:done properties=%s", len(properties) if properties else 0)
        
        if not properties:
            feedback = "No properties matched the current requirements."
            
        return {
            "properties": properties,
            "feedback": feedback,
        }

    def chilling_node(state: AgentState) -> dict:
        logger.info("Node chilling:start")
        retry_count = state.retries + 1
        requirements = None
        if state.requirements:
            requirements = [services.intake.chill_request(state.requirements[0], state.feedback)]
        logger.info("Node chilling:done retry=%s", retry_count)
        return {
            "requirements": requirements,
            "retries": retry_count,
        }

    def news_node(state: AgentState) -> dict:
        logger.info("Node news:start")
        news_items = []
        if state.requirements:
            news_items = services.news.search(state.requirements[0], state.properties or [])
        logger.info("Node news:done insights=%s", len(news_items))
        return {
            "news_items": news_items,
        }

    def skip_news_node(state: AgentState) -> dict:
        logger.info("Node news-skip:done")
        return {
            "news_items": [],
        }

    def whatsapp_node(state: AgentState) -> dict:
        logger.info("Node whatsapp:start")
        validation = services.whatsapp.validate(state.properties or [])
        logger.info("Node whatsapp:done validations=%s", len(validation))
        return {} # Placeholder for validation state if added to AgentState later

    def evaluator_node(state: AgentState) -> dict:
        logger.info("Node evaluator:start")
        evaluation = services.evaluation.evaluate(
            request=state.requirements[0] if state.requirements else None,
            listings=state.properties or [],
            news=state.news_items or [],
            threshold=settings.evaluation_threshold,
        )
        logger.info("Node evaluator:done passed=%s", evaluation.passed)
        return {
            "feedback": "; ".join(evaluation.required_fixes) if not evaluation.passed else "",
            # Needs evaluation state added to AgentState to store full result
        }

    def seller_node(state: AgentState) -> dict:
        logger.info("Node seller:start properties=%s", len(state.properties or []))
        return {} # Placeholder

    def no_results_node(state: AgentState) -> dict:
        logger.warning("Node no-results:triggered retries=%s", state.retries)
        return {} # Placeholder

    def retry_node(state: AgentState) -> dict:
        logger.info("Node retry:triggered current_retries=%s", state.retries)
        return {
            "retries": state.retries + 1,
        }

    def listings_route(state: AgentState) -> str:
        if state.properties:
            logger.info("Route scraper -> after_scrape")
            return "after_scrape"
        if state.retries >= settings.max_retries:
            logger.info("Route scraper -> no_results")
            return "no_results"
        logger.info("Route scraper -> chilling")
        return "chilling"

    def news_route(state: AgentState) -> str:
        route = "news" if state.run_news else "skip_news"
        logger.info("Route after_scrape -> %s", route)
        return route

    def evaluation_route(state: AgentState) -> str:
        # Simplified routing for now
        logger.info("Route evaluator -> seller")
        return "seller"

    graph.add_node("intake", intake_node)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("scraper", scraping_node)
    graph.add_node("after_scrape", lambda state: state)
    graph.add_node("chilling", chilling_node)
    graph.add_node("no_results", no_results_node)
    graph.add_node("retry", retry_node)
    graph.add_node("news", news_node)
    graph.add_node("skip_news", skip_news_node)
    graph.add_node("whatsapp", whatsapp_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("seller", seller_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "coordinator")
    graph.add_edge("coordinator", "scraper")
    graph.add_conditional_edges(
        "scraper",
        listings_route,
        {
            "chilling": "chilling",
            "after_scrape": "after_scrape",
            "no_results": "no_results",
        },
    )
    graph.add_edge("chilling", "coordinator")
    graph.add_edge("no_results", "seller")
    graph.add_edge("retry", "coordinator")
    graph.add_conditional_edges(
        "after_scrape",
        news_route,
        {
            "news": "news",
            "skip_news": "skip_news",
        },
    )
    graph.add_edge("news", "whatsapp")
    graph.add_edge("skip_news", "whatsapp")
    graph.add_edge("whatsapp", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        evaluation_route,
        {
            "seller": "seller",
            "retry": "retry",
        },
    )
    graph.add_edge("seller", END)

    return graph.compile()