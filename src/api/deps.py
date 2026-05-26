"""FastAPI dependencies: the compiled graph is a process-wide singleton.

When ``DATABASE_URL`` is set, threads persist across process restarts via
a Postgres-backed checkpointer. Without it, the graph falls back to the
in-memory saver — fine for local dev and unit tests.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger("estatia.api.deps")


@lru_cache(maxsize=1)
def get_graph() -> Any:
    """Compile the graph once per process and reuse for every request."""
    from src.graph.graph import (
        build_graph,
        make_memory_checkpointer,
        make_postgres_checkpointer,
    )

    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            logger.info("Compiling graph with Postgres checkpointer")
            return build_graph(checkpointer=make_postgres_checkpointer(db_url))
        except Exception:
            logger.exception(
                "Postgres checkpointer init failed — falling back to in-memory"
            )

    logger.info("Compiling graph with in-memory checkpointer")
    return build_graph(checkpointer=make_memory_checkpointer())
