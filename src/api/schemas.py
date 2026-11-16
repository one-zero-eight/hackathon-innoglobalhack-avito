"""Схемы веб-сервиса: контракт кейса из main.py, датасет и пакетный запрос.

Контракт входа/выхода (`PipelineInput` / `Draft` / `PipelineOutput`) и таблица
микрокатегорий не дублируются — реэкспортируются из CLI-ядро `src/main.py`.
Схемы `Batch*` описывают тело и SSE-события `POST /api/drafts/batch`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from main import MICROCATEGORIES, Draft, PipelineInput, PipelineOutput

DatasetSplit = Literal["train", "test"]

# Лимиты пакетной обработки (POST /api/drafts/batch).
MAX_BATCH_ADS = 1000
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 8


class DatasetSample(BaseModel):
    """Объявление датасета кейса из таблицы case_ads (эндпоинты /api/dataset/*)."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    split: DatasetSplit
    item_id: int | None = Field(default=None, alias="itemId")
    title: str
    description: str
    source_mc_id: int = Field(alias="sourceMcId")
    found_mc_ids: list[int] | None = Field(default=None, alias="foundMcIds")
    categories: list[str] | None = None


class DatasetStats(BaseModel):
    """Количество объявлений датасета по сплитам: {"train": N, "test": M}."""

    train: int
    test: int


class BatchRequest(BaseModel):
    """Тело POST /api/drafts/batch: массив объявлений и параллелизм.

    Каждое объявление валидируется как `PipelineInput` (mcId 101–106),
    всего не более `MAX_BATCH_ADS` строк; `concurrency` ограничивается
    при обработке, не здесь (clamp к [1, MAX_CONCURRENCY]).
    """

    ads: list[PipelineInput] = Field(max_length=MAX_BATCH_ADS)
    concurrency: int | None = None


class BatchStartEvent(BaseModel):
    """SSE-событие `start`: размер пакета и фактический параллелизм."""

    total: int
    concurrency: int


class BatchResultEvent(BaseModel):
    """SSE-событие `result`: черновики строки по её позиции во входе."""

    model_config = ConfigDict(populate_by_name=True)

    index: int
    mc_id: int = Field(alias="mcId")
    drafts: list[Draft]


class BatchErrorEvent(BaseModel):
    """SSE-событие `error`: сбой одной строки с безопасным detail."""

    index: int
    detail: str


class BatchDoneEvent(BaseModel):
    """SSE-событие `done`: итоги пакета по обработанным строкам."""

    processed: int
    succeeded: int
    failed: int


class StreamCategoriesEvent(BaseModel):
    """SSE-событие `categories` потока /api/drafts/stream: цели классификации."""

    model_config = ConfigDict(populate_by_name=True)

    mc_ids: list[int] = Field(alias="mcIds")


class StreamErrorEvent(BaseModel):
    """SSE-событие `error` потока /api/drafts/stream: безопасный detail."""

    detail: str


__all__ = [
    "MICROCATEGORIES",
    "BatchDoneEvent",
    "BatchErrorEvent",
    "BatchRequest",
    "BatchResultEvent",
    "BatchStartEvent",
    "DEFAULT_CONCURRENCY",
    "MAX_BATCH_ADS",
    "MAX_CONCURRENCY",
    "DatasetSample",
    "DatasetSplit",
    "DatasetStats",
    "Draft",
    "PipelineInput",
    "PipelineOutput",
    "StreamCategoriesEvent",
    "StreamErrorEvent",
]
