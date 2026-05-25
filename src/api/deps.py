"""FastAPI dependencies: the compiled graph is a process-wide singleton.

The graph carries an in-memory checkpointer so threads survive across
requests but die on process restart. To persist threads, swap
``make_memory_checkpointer()`` for a Postgres-backed saver here.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger("estatia.api.deps")


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Compile the graph once per process and reuse for every request."""
    from src.graph.graph import build_graph, make_memory_checkpointer

    logger.info("Compiling graph with in-memory checkpointer")
    return build_graph(checkpointer=make_memory_checkpointer())
