"""FastAPI-приложение веб-сервиса черновиков Avito.

Роуты: POST /api/drafts, GET /api/microcategories, GET /api/health и датасет
кейса: GET /api/dataset/samples, /api/dataset/random, /api/dataset/stats.
На старте читаются настройки и создаётся пул БД без подключения;
отсутствие ключей LLM — не ошибка запуска, сбои видны только на запросах.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.clients.decisions import DecisionsClient
from api.config import ENV_FILE, Settings
from api.db import (
    close_pool,
    fetch_dataset_samples,
    fetch_dataset_stats,
    fetch_random_dataset_sample,
    get_catalog,
    init_pool,
)
from api.errors import LLMGatewayError
from api.schemas import (
    DatasetSample,
    DatasetSplit,
    DatasetStats,
    PipelineInput,
    PipelineOutput,
)
from api.service import DraftsService

logger = logging.getLogger(__name__)

# Веб-клиент (WP4) в dev-режиме Vite.
DEV_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
# Проверка БД в health-check не должна вешать ответ.
HEALTH_DB_TIMEOUT_S = 2.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Собрать сервис при старте и закрыть пул при выключении."""
    load_dotenv(ENV_FILE)
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
    app.state.service = DraftsService(
        classifier, app.state.pool, settings.classifier_threshold
    )
    logger.info(
        "Сервис черновиков готов: классификатор %s (Decisions API), генератор из config.json",
        settings.selectel_decisions_model,
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


@app.get("/api/dataset/samples", response_model=list[DatasetSample])
async def dataset_samples(
    request: Request,
    split: DatasetSplit = "train",
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[DatasetSample]:
    """Страница объявлений датасета из таблицы case_ads (id по возрастанию)."""
    rows: list[dict[str, Any]] = await fetch_dataset_samples(
        request.app.state.pool, split, limit, offset
    )
    return [DatasetSample.model_validate(row) for row in rows]


@app.get("/api/dataset/random", response_model=DatasetSample)
async def dataset_random(request: Request, split: DatasetSplit = "train") -> DatasetSample:
    """Один случайный сэмпл сплита; пустой сплит — 404 с русским detail."""
    row = await fetch_random_dataset_sample(request.app.state.pool, split)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"В сплите «{split}» нет объявлений — выполните загрузку датасета "
            "(см. README, раздел «Датасет»)",
        )
    return DatasetSample.model_validate(row)


@app.get("/api/dataset/stats", response_model=DatasetStats)
async def dataset_stats(request: Request) -> DatasetStats:
    """Число объявлений по сплитам: {"train": N, "test": M}."""
    return DatasetStats.model_validate(await fetch_dataset_stats(request.app.state.pool))


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
