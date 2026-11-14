"""FastAPI-приложение веб-сервиса черновиков Avito.

Роуты: POST /api/drafts, GET /api/microcategories, GET /api/health.
На старте читаются настройки и создаётся пул БД без подключения;
отсутствие ключей LLM — не ошибка запуска, сбои видны только на запросах.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.clients.decisions import DecisionsClient
from api.clients.openai_compat import OpenAIChatClient
from api.config import Settings
from api.db import close_pool, get_catalog, init_pool
from api.errors import LLMGatewayError
from api.schemas import PipelineInput, PipelineOutput
from api.service import DraftsService

logger = logging.getLogger(__name__)

# Веб-клиент (WP4) в dev-режиме Vite.
DEV_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
# Проверка БД в health-check не должна вешать ответ.
HEALTH_DB_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Собрать сервис при старте и закрыть пул при выключении."""
    settings = Settings()
    app.state.settings = settings
    app.state.pool = await init_pool(settings.database_url)
    classifier = DecisionsClient(
        settings.selectel_decisions_url,
        settings.selectel_api_key,
        settings.selectel_decisions_model,
        timeout=settings.classifier_timeout,
        retries=settings.classifier_retries,
    )
    generator = OpenAIChatClient(
        settings.selectel_api_url,
        settings.selectel_api_key,
        settings.selectel_model,
        timeout=settings.generation_timeout,
        retries=settings.generation_retries,
    )
    app.state.service = DraftsService(
        classifier, generator, app.state.pool, settings.classifier_threshold
    )
    logger.info(
        "Сервис черновиков готов: классификатор %s (Decisions API), генератор %s",
        settings.selectel_decisions_model,
        settings.selectel_model,
    )
    try:
        yield
    finally:
        await close_pool(app.state.pool)


app = FastAPI(
    title="Веб-сервис черновиков Avito",
    description=(
        "По тексту объявления и его микрокатегории подбирает услуги "
        "в других микрокатегориях «Ремонта и отделки» и генерирует черновики."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMGatewayError)
async def llm_gateway_error_handler(request: Request, exc: LLMGatewayError) -> JSONResponse:
    """Сбой LLM-шлюза (классификатор/генератор) — 502 с русским сообщением."""
    logger.error("LLM-шлюз: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


DB_ERROR_EXCEPTIONS: tuple[type[Exception], ...] = (
    asyncpg.PostgresError,
    asyncpg.exceptions.InterfaceError,
    ConnectionRefusedError,
    asyncio.TimeoutError,
)


async def db_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """База данных недоступна — 503 (в health-check ошибка обрабатывается самим роутом)."""
    logger.error("Ошибка БД при запросе %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=503, content={"detail": "База данных недоступна, повторите запрос позже"}
    )


for _db_exc_class in DB_ERROR_EXCEPTIONS:
    app.add_exception_handler(_db_exc_class, db_error_handler)


@app.post("/api/drafts", response_model=PipelineOutput)
async def create_drafts(inp: PipelineInput, request: Request) -> PipelineOutput:
    """Сформировать дополнительные черновики: {"description": ..., "mcId": ...}."""
    return await request.app.state.service.run(inp)


@app.get("/api/microcategories")
async def microcategories(request: Request) -> list[dict[str, object]]:
    """Справочник микрокатегорий из БД: [{"mcId": 101, "title": "..."}]."""
    catalog = await get_catalog(request.app.state.pool)
    return [{"mcId": mc_id, "title": catalog[mc_id]} for mc_id in sorted(catalog)]


@app.get("/api/health")
async def health(request: Request) -> dict[str, object]:
    """Живость API и БД; при недоступной базе — degraded, но не 5xx."""
    settings: Settings = request.app.state.settings
    db_ok = False
    try:
        async with request.app.state.pool.acquire() as conn:
            db_ok = (await conn.fetchval("SELECT 1", timeout=HEALTH_DB_TIMEOUT_S)) == 1
    except Exception as exc:  # noqa: BLE001 — health не должен отдавать 5xx
        logger.warning("Health-check: база недоступна: %s", exc)
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "classifier_model": settings.selectel_decisions_model,
    }
