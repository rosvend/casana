"""Agent node re-exports.

``src.graph.graph`` imports agents by the bare names below
(``from src.agents import requirements_agent, ...``). Each completed agent is
re-exported here under that name, bound to its node *function* — so
``graph.add_node("news_agent", news_agent)`` registers the callable, not the
module.

Only the completed agents appear here. The remaining names
(``router_agent``, ``whatsapp_agent``) still resolve via submodule import
until those nodes are implemented; add them here as each is finished.
"""

from src.agents.evaluator_agent import evaluator_node as evaluator_agent
from src.agents.news_agent import news_node as news_agent
from src.agents.properties_agent import properties_node as properties_agent
from src.agents.requirements_agent import requirements_node as requirements_agent
from src.agents.softener_agent import softener_node as softener_agent
from src.agents.synthesizer_agent import synthesizer_node as synthesizer_agent

__all__ = [
    "requirements_agent",
    "properties_agent",
    "news_agent",
    "synthesizer_agent",
    "evaluator_agent",
    "softener_agent",
]
