"""Схемы веб-сервиса: контракт кейса из main.py и схемы датасета.

Контракт входа/выхода (`PipelineInput` / `Draft` / `PipelineOutput`) и таблица
микрокатегорий не дублируются — реэкспортируются из CLI-ядро `src/main.py`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from main import MICROCATEGORIES, Draft, PipelineInput, PipelineOutput

DatasetSplit = Literal["train", "test"]


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


__all__ = [
    "MICROCATEGORIES",
    "DatasetSample",
    "DatasetSplit",
    "DatasetStats",
    "Draft",
    "PipelineInput",
    "PipelineOutput",
]
