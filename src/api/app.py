"""FastAPI-приложение веб-сервиса черновиков Avito.

Роуты: POST /api/drafts, POST /api/drafts/batch (SSE-пакет), GET
/api/microcategories, GET /api/health и датасет кейса: GET
/api/dataset/samples, /api/dataset/random, /api/dataset/stats.
Весь пайплайн (извлечение подкатегорий и генерация) делегируется ядру ``main``:
модули выбираются только в ``config.json`` и выполняются через
``main.PipelineRuntime``. На старте читаются ``.env`` и ``config.json``,
выбранные модули проверяются по каталогу, создаётся пул БД без подключения;
отсутствие ключей LLM — не ошибка запуска, сбои видны только на запросах.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.error import URLError

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.bulk import BulkService
from api.config import Settings
from api.db import (
    close_pool,
    fetch_dataset_samples,
    fetch_dataset_stats,
    fetch_random_dataset_sample,
    get_catalog,
    init_pool,
)
from api.schemas import (
    BatchRequest,
    DatasetSample,
    DatasetSplit,
    DatasetStats,
    PipelineInput,
    PipelineOutput,
)
from api.service import DraftsService
from main import (
    DEFAULT_CONFIG_PATH,
    PipelineConfig,
    PipelineOutputError,
    PipelineRuntime,
    PipelineUnavailableError,
    discover_modules,
    load_config,
)

logger = logging.getLogger(__name__)

# Веб-клиент (WP4) в dev-режиме Vite.
DEV_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
# Проверка БД в health-check не должна вешать ответ.
HEALTH_DB_TIMEOUT_S = 2.0

# Ошибки провайдера и нарушения контракта модели — 502 с безопасным detail.
PIPELINE_ERROR_DETAIL = "Ошибка провайдера или некорректный результат модели"
PIPELINE_ERROR_EXCEPTIONS: tuple[type[Exception], ...] = (
    PipelineOutputError,
    ValueError,
    URLError,
    TimeoutError,
    ConnectionError,
)
# Недоступные настройки/зависимости/ключ выбранного модуля — 503.
PIPELINE_UNAVAILABLE_DETAIL = (
    "Модуль пайплайна недоступен: проверьте настройки и ресурсы сервера"
)


def _validate_selected_modules(config: PipelineConfig) -> None:
    """Проверить, что модули из config.json существуют; иначе — ошибка старта.

    Запасная реализация молча не подставляется: неизвестный extractor или
    generator валит запуск приложения с явным сообщением.
    """
    extractors = discover_modules("subcategory_extraction")
    generators = discover_modules("draft_generation")
    if config.subcategory_extraction not in extractors:
        raise ValueError(
            "В config.json выбран несуществующий extractor: "
            f"{config.subcategory_extraction}"
        )
    if config.draft_generation not in generators:
        raise ValueError(
            "В config.json выбран несуществующий generator: "
            f"{config.draft_generation}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Собрать сервис при старте и закрыть пул при выключении."""
    load_dotenv(DEFAULT_CONFIG_PATH.parent / ".env")
    config = load_config()
    _validate_selected_modules(config)
    settings = Settings()
    runtime = PipelineRuntime()
    app.state.settings = settings
    app.state.config = config
    app.state.runtime = runtime
    app.state.pool = await init_pool(settings.database_url)
    app.state.service = DraftsService(runtime, config, app.state.pool)
    # Пакетный сервис не переиспользует runtime: у каждой строки свой пайплайн.
    app.state.bulk = BulkService()
    logger.info(
        "Сервис черновиков готов: extractor %s, generator %s (config.json)",
        config.subcategory_extraction,
        config.draft_generation,
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


@app.exception_handler(PipelineUnavailableError)
async def pipeline_unavailable_error_handler(
    request: Request, exc: PipelineUnavailableError
) -> JSONResponse:
    """Выбранный модуль недоступен — 503 без деталей причины и ключей."""
    logger.warning(
        "Модуль недоступен: %s/%s (%s)",
        exc.service,
        exc.module,
        type(exc.__cause__).__name__,
    )
    return JSONResponse(status_code=503, content={"detail": PIPELINE_UNAVAILABLE_DETAIL})


async def pipeline_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Ошибка провайдера или некорректный результат модели — 502, без traceback."""
    logger.warning(
        "Ошибка пайплайна при запросе %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(status_code=502, content={"detail": PIPELINE_ERROR_DETAIL})


for _pipeline_exc_class in PIPELINE_ERROR_EXCEPTIONS:
    app.add_exception_handler(_pipeline_exc_class, pipeline_error_handler)


DB_ERROR_EXCEPTIONS: tuple[type[Exception], ...] = (
    asyncpg.PostgresError,
    asyncpg.exceptions.InterfaceError,
    ConnectionRefusedError,
)
# asyncio.TimeoutError — алиас встроенного TimeoutError (Python 3.11+):
# таймауты относим к ошибкам пайплайна (502), а не к ошибкам БД.


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


@app.post("/api/drafts/batch")
async def create_drafts_batch(inp: BatchRequest, request: Request) -> StreamingResponse:
    """Пакетная обработка: SSE-поток start/result/error/done по мере готовности.

    Тело валидируется до старта потока (лимиты и mcId — 422); дальше каждая
    строка обрабатывается независимым пайплайном в общем пуле потоков под
    семафором, результат уходит кадром по мере завершения задачи.
    """
    stream = request.app.state.bulk.stream(
        inp.ads,
        inp.concurrency,
        request.app.state.config,
        request.is_disconnected,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    """Живость API, выбранная пара модулей и БД; недоступная база — degraded, не 5xx."""
    config: PipelineConfig = request.app.state.config
    db_ok = False
    try:
        async with request.app.state.pool.acquire() as conn:
            db_ok = (await conn.fetchval("SELECT 1", timeout=HEALTH_DB_TIMEOUT_S)) == 1
    except Exception as exc:  # noqa: BLE001 — health не должен отдавать 5xx
        logger.warning("Health-check: база недоступна: %s", exc)
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "extractor": config.subcategory_extraction,
        "generator": config.draft_generation,
    }
