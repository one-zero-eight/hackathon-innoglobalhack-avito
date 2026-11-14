"""Схемы веб-сервиса: контракт кейса из main.py и валидация ответов LLM.

Контракт входа/выхода (`PipelineInput` / `Draft` / `PipelineOutput`) и таблица
микрокатегорий не дублируются — реэкспортируются из CLI-ядро `src/main.py`.
"""

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


class GeneratedText(BaseModel):
    """Ответ генератора черновика: {"title": "...", "text": "..."}, обе строки непустые."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


__all__ = [
    "MICROCATEGORIES",
    "ClassifierItem",
    "Draft",
    "GeneratedText",
    "PipelineInput",
    "PipelineOutput",
]
