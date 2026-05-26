"""FastAPI app factory for Estatia.

Mounts the chat router under ``/api`` so the same FastAPI process can also
serve the built Vite SPA at ``/`` in production (single-origin, no CORS).
CORS is still configured from ``ESTATIA_ALLOWED_ORIGINS`` for split-host
setups where the SPA is served from a different domain.

Run locally::

    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

load_dotenv()

logging.basicConfig(level=os.getenv("ESTATIA_LOG_LEVEL", "INFO"))

# Repo-relative path to the built SPA. Present in container builds (the
# Dockerfile copies it from the Node stage); absent in `uv run uvicorn`
# dev runs, in which case we skip the mount and let Vite serve the SPA.
_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "src" / "frontend" / "dist"


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
    app.include_router(router, prefix="/api")

    if _FRONTEND_DIST.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=_FRONTEND_DIST, html=True),
            name="spa",
        )

    return app


app = create_app()
