from __future__ import annotations

import logging
from urllib.parse import parse_qs

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from estatia.config import settings
from estatia.graph import build_graph
from estatia.logging_utils import configure_logging
from estatia.services import (
    OpenAIWorkflowService,
    PlaywrightListingService,
    Services,
    StandbyWhatsAppService,
    TavilyNewsService,
)


configure_logging(settings.log_level)
logger = logging.getLogger("estatia.app")
templates = Jinja2Templates(directory="src/estatia/templates")
app = FastAPI(title=settings.app_name)


def build_ui_text(language: str) -> dict[str, str]:
    is_spanish = language == "es"
    return {
        "lang": language,
        "page_title": "Estatia",
        "brief": "Resumen inmobiliario" if is_spanish else "Real estate brief",
        "intro": (
            "Escribe la solicitud como la describiría un cliente. El flujo estructura la entrada, "
            "busca inmuebles, relaja restricciones cuando el mercado está apretado y prepara una recomendación final."
            if is_spanish
            else "Write the request the way a client would describe it. The workflow structures the input, "
            "searches for listings, relaxes constraints when the market is too tight, and prepares a final recommendation."
        ),
        "user_input": "Solicitud del usuario" if is_spanish else "User input",
        "supporting": (
            "La versión actual prioriza la obtención de inmuebles y la calidad de la recomendación. "
            "La validación por WhatsApp sigue en espera."
            if is_spanish
            else "The current version prioritizes listing retrieval and recommendation quality. "
            "WhatsApp validation remains on standby."
        ),
        "run": "Ejecutar agente" if is_spanish else "Run workflow",
        "working": "Trabajando..." if is_spanish else "Working...",
        "error": "Error",
        "trace": "Trazabilidad" if is_spanish else "Trace",
        "no_report": "Todavía no hay reporte" if is_spanish else "No report yet",
        "no_report_copy": (
            "El reporte final aparecerá aquí cuando termine el grafo."
            if is_spanish
            else "The final seller HTML will appear here after the graph finishes."
        ),
        "loading_title": "Agente en ejecución" if is_spanish else "Agent running",
        "loading_copy": (
            "Buscando inmuebles, relajando restricciones cuando sea necesario y preparando el reporte final."
            if is_spanish
            else "Searching listings, relaxing constraints when needed, and preparing the final report."
        ),
        "steps": (
            '["Analizando la solicitud...","Buscando inmuebles...","Revisando si hay que relajar restricciones...","Evaluando propiedades candidatas...","Armando el reporte final..."]'
            if is_spanish
            else '["Parsing the request...","Searching for listings...","Checking whether constraints need relaxing...","Evaluating candidate properties...","Composing the final report..."]'
        ),
        "toggle_label": "ES" if language == "en" else "EN",
        "toggle_href": "/" if language == "es" else "/?lang=es",
        "sample_text": (
            "Quiero arrendar un apartamento de 2 habitaciones en Bogotá por hasta 4.500.000 COP. "
            "Prefiero una zona caminable, cerca de transporte público y con buena luz natural."
            if is_spanish
            else "I want to rent a 2-bedroom apartment in Bogota for up to 4,500,000 COP. "
            "I prefer a walkable area, near public transport, with natural light."
        ),
        "empty_error": (
            "Escribe una solicitud inmobiliaria antes de ejecutar el flujo."
            if is_spanish
            else "Please provide a property request before running the workflow."
        ),
    }


def build_services() -> Services:
    logger.info("Building services with listing_mode=%s", settings.listing_mode)
    workflow = OpenAIWorkflowService(settings)
    return Services(
        intake=workflow,
        evaluation=workflow,
        seller=workflow,
        listing=PlaywrightListingService(settings),
        news=TavilyNewsService(settings),
        whatsapp=StandbyWhatsAppService(),
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    logger.info("Rendering home page")
    language = "es" if request.query_params.get("lang") == "es" else "en"
    ui = build_ui_text(language)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": ui["page_title"],
            "result_html": None,
            "trace": [],
            "error": None,
            "raw_text": ui["sample_text"],
            "ui": ui,
        },
    )


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.post("/run", response_class=HTMLResponse)
async def run_workflow(request: Request) -> HTMLResponse:
    body = await request.body()
    form = parse_qs(body.decode("utf-8"))
    raw_text = form.get("raw_text", [""])[0].strip()
    language = "es" if form.get("language", ["en"])[0] == "es" else "en"
    ui = build_ui_text(language)
    logger.info("Received workflow request")
    if not raw_text:
        logger.warning("Workflow request arrived without raw_text")
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": ui["page_title"],
                "result_html": None,
                "trace": [],
                "error": ui["empty_error"],
                "raw_text": raw_text,
                "ui": ui,
            },
        )
    try:
        services = build_services()
        graph = build_graph(services, settings)
        logger.info("Invoking LangGraph workflow")
        state = await run_in_threadpool(
            graph.invoke,
            {"raw_text": raw_text, "language": language, "retries": 0, "trace": []},
        )
        logger.info(
            "Workflow finished with listings=%s evaluation_passed=%s",
            len(state.get("listings", [])),
            state.get("evaluation").passed if state.get("evaluation") else None,
        )
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": ui["page_title"],
                "result_html": state.get("html"),
                "trace": state.get("trace", []),
                "error": None,
                "raw_text": raw_text,
                "ui": ui,
            },
        )
    except Exception as exc:
        logger.exception("Workflow execution failed")
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": ui["page_title"],
                "result_html": None,
                "trace": [],
                "error": str(exc),
                "raw_text": raw_text,
                "ui": ui,
            },
        )
