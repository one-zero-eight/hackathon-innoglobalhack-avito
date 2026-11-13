"""CLI-ядро проекта.

Здесь живёт всё общее для сервисов: микрокатегории, контракт входа/выхода,
база пайплайнов и клиент LLM. Логика отдельных шагов — в папках сервисов,
которые импортируют общее из этого модуля.

Импорты сервисов намеренно внутри run_pipeline(): сервисы импортируют main,
и модульный импорт на верхнем уровне дал бы циклическую зависимость.

Пример:
    python main.py --input data/input.json --output out/output.json
"""

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# --- Микрокатегории кейса «Ремонт и отделка» ---
# Таблица зафиксирована условиями кейса (docs/Innoglobalhack-Avito_2026.pdf):
# новые микрокатегории не создаются, выходить за таблицу нельзя.

MICROCATEGORIES: dict[int, str] = {
    101: "Ремонт под ключ",
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}

VALID_MCIDS: frozenset[int] = frozenset(MICROCATEGORIES)


def is_valid_mcid(mcid: int) -> bool:
    """Проверить, что mcId есть в таблице микрокатегорий кейса."""
    return mcid in VALID_MCIDS


def mcid_name(mcid: int) -> str | None:
    """Название микрокатегории по mcId или None, если такой категории нет."""
    return MICROCATEGORIES.get(mcid)


# --- Контракт входа/выхода ---
# Контракт зафиксирован условиями кейса и меняется только с согласования команды:
# вход — {"description": str, "mcId": int},
# выход — {"drafts": [{"mcId": int, "title": str|null, "text": str|null}]}.


def _validate_mcid(value: int) -> int:
    """Отбросить mcId, которого нет в таблице микрокатегорий кейса."""
    if not is_valid_mcid(value):
        raise ValueError(f"Неизвестная микрокатегория: {value}")
    return value


McId = Annotated[int, AfterValidator(_validate_mcid)]


class PipelineInput(BaseModel):
    """Входные данные пайплайна: текст объявления и его микрокатегория."""

    model_config = ConfigDict(populate_by_name=True)

    description: str
    mc_id: McId = Field(alias="mcId")


class Draft(BaseModel):
    """Черновик объявления для целевой микрокатегории."""

    model_config = ConfigDict(populate_by_name=True)

    mc_id: McId = Field(alias="mcId")
    title: str | None = None
    text: str | None = None


class PipelineOutput(BaseModel):
    """Выходные данные пайплайна: список дополнительных черновиков."""

    model_config = ConfigDict(populate_by_name=True)

    drafts: list[Draft] = Field(default_factory=list)


# --- База пайплайнов ---
# Один сервис — один класс с методом run(). Класс не делает ничего при импорте.


class BasePipeline(ABC):
    """Оркестрируемый шаг пайплайна."""

    name: str = "base"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Выполнить шаг пайплайна и вернуть результат."""


# --- Клиент LLM ---
# Заглушка jev-модели. Точный model id и ключ берём из окружения (.env):
# LLM_MODEL_ID — идентификатор модели (по умолчанию "jev"),
# LLM_API_KEY — ключ доступа (в репозиторий не коммитим).


class LLMClient:
    """Минимальный клиент LLM. TODO: реализовать вызов jev-модели."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "jev")
        self.api_key = os.getenv("LLM_API_KEY")

    def complete(self, prompt: str, **kwargs) -> str:
        """TODO: заменить заглушку на реальный вызов jev-модели."""
        raise NotImplementedError("LLMClient.complete: вызов jev-модели не реализован")


# --- Оркестрация ---


def run_pipeline(inp: PipelineInput) -> PipelineOutput:
    """Прогнать цепочку сервисов: выделение -> проверка -> генерация.

    Импорты внутри функции: сервисы импортируют общее из main, поэтому на
    верхнем уровне модуля была бы циклическая зависимость.
    """
    from category_validation import CategoryValidationPipeline
    from draft_generation import DraftGenerationPipeline
    from subcategory_extraction import SubcategoryExtractionPipeline

    extraction = SubcategoryExtractionPipeline()
    validation = CategoryValidationPipeline()
    generation = DraftGenerationPipeline()

    candidates = extraction.run(inp)
    target_mc_ids = validation.run(candidates, inp.mc_id)
    drafts = generation.run(inp, target_mc_ids)
    return PipelineOutput(drafts=drafts)


# --- CLI ---


def build_parser() -> argparse.ArgumentParser:
    """CLI: пути к файлу ввода и файлу вывода."""
    parser = argparse.ArgumentParser(description="Выделение услуг и черновики объявлений")
    parser.add_argument("--input", required=True, type=Path, help="путь к JSON на вход")
    parser.add_argument("--output", required=True, type=Path, help="путь к JSON на выход")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Прочитать входной файл, прогнать пайплайн, записать выходной файл."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()  # переменные из .env (LLM_MODEL_ID, LLM_API_KEY)
    args = build_parser().parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    inp = PipelineInput.model_validate(payload)

    output = run_pipeline(inp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output.model_dump(by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Готово: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
