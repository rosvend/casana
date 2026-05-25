"""FastAPI app factory for Estatia.

Mounts the chat router and configures CORS from ``ESTATIA_ALLOWED_ORIGINS``
(comma-separated). The special value ``*`` collapses to a single
wildcard origin so a wide-open dev setup works without listing each
frontend port. Default is ``http://localhost:3000``.

Run locally::

    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

load_dotenv()

logging.basicConfig(level=os.getenv("ESTATIA_LOG_LEVEL", "INFO"))


def _allowed_origins() -> list[str]:
    raw = os.getenv("ESTATIA_ALLOWED_ORIGINS", "http://localhost:3000")
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or ["http://localhost:3000"]


def create_app() -> FastAPI:
    app = FastAPI(title="Estatia API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
