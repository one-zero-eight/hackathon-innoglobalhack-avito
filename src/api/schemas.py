"""Схемы веб-сервиса: контракт кейса из main.py, валидация ответов LLM и датасет.

Контракт входа/выхода (`PipelineInput` / `Draft` / `PipelineOutput`) и таблица
микрокатегорий не дублируются — реэкспортируются из CLI-ядро `src/main.py`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from main import MICROCATEGORIES, Draft, McId, PipelineInput, PipelineOutput


class ClassifierItem(BaseModel):
    """Оценка одной микрокатегории в ответе классификатора: {"mcId": 101, "p": 0.93}."""

    model_config = ConfigDict(populate_by_name=True)

    mc_id: McId = Field(alias="mcId")
    p: float

    @field_validator("p", mode="before")
    @classmethod
    def _clamp_p(cls, value: object) -> float:
        """Число вне [0, 1] (или строка "0.95") приводим к p в диапазоне [0, 1]."""
        try:
            p = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"p не является числом: {value!r}") from exc
        return min(max(p, 0.0), 1.0)



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
    "ClassifierItem",
    "DatasetSample",
    "DatasetSplit",
    "DatasetStats",
    "Draft",
    "PipelineInput",
    "PipelineOutput",
]
