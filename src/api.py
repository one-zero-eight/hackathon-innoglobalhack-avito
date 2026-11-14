"""HTTP-адаптер существующего пайплайна без файлового ввода и вывода."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status

from main import (
    DEFAULT_CONFIG_PATH,
    PIPELINE_CLASSES,
    LLMError,
    PipelineInput,
    PipelineOutput,
    load_config,
    load_pipeline,
    run_pipeline,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Прочитать конфигурацию проекта при старте, не вызывая внешние модели."""
    load_dotenv(DEFAULT_CONFIG_PATH.parent / ".env")
    config = load_config()
    for service in PIPELINE_CLASSES:
        load_pipeline(service, config)
    application.state.pipeline_config = config
    yield


app = FastAPI(
    title="Avito draft generation",
    description="Выделение услуг и создание черновиков. Допустимые mcId: 101–106.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Проверить доступность сервера без обращений к API моделей."""
    return {"status": "ok"}


@app.post(
    "/drafts", response_model=PipelineOutput, response_model_by_alias=True,
    responses={502: {"description": "Ошибка API модели или некорректный результат пайплайна"}},
)
def create_drafts(inp: PipelineInput, request: Request) -> PipelineOutput:
    """Обработать JSON выбранными модулями синхронно в thread pool."""
    try:
        return run_pipeline(inp, request.app.state.pipeline_config)
    except LLMError as exc:
        logger.warning("Ошибка API модели: %s", exc)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from None
    except ValueError:
        logger.warning("Пайплайн вернул некорректный результат")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Пайплайн вернул некорректный результат",
        ) from None
