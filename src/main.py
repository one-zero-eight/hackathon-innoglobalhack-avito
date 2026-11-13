"""CLI-ядро проекта.

Здесь живёт всё общее для сервисов: микрокатегории, контракт входа/выхода,
база пайплайнов и клиент LLM. Логика отдельных шагов — в папках сервисов,
которые импортируют общее из этого модуля.

Импорты модулей выполняются при загрузке пайплайна: сервисы импортируют main,
и модульный импорт на верхнем уровне дал бы циклическую зависимость.

Пример:
    python src/main.py --input data/input.json --output out/output.json
"""

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Annotated, Literal

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
# Один вариант сервиса — один класс с методом run().


class BasePipeline(ABC):
    """Оркестрируемый шаг пайплайна."""

    name: str = "base"

    @abstractmethod
    def run(self, *args, **kwargs):
        """Выполнить шаг пайплайна и вернуть результат."""


# --- Клиент LLM ---
# Общий интерфейс клиента. Точный model id и ключ берём из окружения (.env):
# LLM_MODEL_ID — идентификатор модели (по умолчанию "jev"),
# LLM_API_KEY — ключ доступа (в репозиторий не коммитим).


class LLMClient(ABC):
    """Интерфейс клиента LLM для конкретных реализаций."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("LLM_MODEL_ID", "jev")
        self.api_key = os.getenv("LLM_API_KEY")

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Получить ответ модели на промпт."""


# --- Оркестрация ---


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
ServiceName = Literal["subcategory_extraction", "draft_generation"]
PIPELINE_CLASSES: dict[ServiceName, str] = {
    "subcategory_extraction": "SubcategoryExtractionPipeline",
    "draft_generation": "DraftGenerationPipeline",
}
ModuleName = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_]*$")]


class PipelineConfig(BaseModel):
    """Выбранный модуль для каждого раздела проекта."""

    model_config = ConfigDict(extra="forbid")

    subcategory_extraction: ModuleName | None
    draft_generation: ModuleName | None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    """Прочитать и проверить конфигурацию выбора модулей."""
    return PipelineConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_pipeline(
    service: ServiceName, config: PipelineConfig | None = None
) -> BasePipeline:
    """Загрузить выбранный класс из <service>/<module>/pipeline.py."""
    if service not in PIPELINE_CLASSES:
        raise ValueError(f"Неизвестный раздел: {service}")
    config = config if config is not None else load_config()
    module_name = getattr(config, service)
    if module_name is None:
        raise ValueError(f"Для раздела {service} не выбран модуль в config.json")
    module_path = f"{service}.{module_name}.pipeline"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name in {service, f"{service}.{module_name}", module_path}:
            raise ValueError(f"Модуль раздела не найден: {module_path}") from exc
        raise
    class_name = PIPELINE_CLASSES[service]
    pipeline_class = getattr(module, class_name, None)
    if not isinstance(pipeline_class, type) or not issubclass(pipeline_class, BasePipeline):
        raise TypeError(f"{module_path}.{class_name} должен наследовать BasePipeline")
    return pipeline_class()


def run_pipeline(
    inp: PipelineInput, config: PipelineConfig | None = None
) -> PipelineOutput:
    """Прогнать выбранные модули: выделение -> генерация."""
    config = config if config is not None else load_config()
    extraction = load_pipeline("subcategory_extraction", config)
    generation = load_pipeline("draft_generation", config)

    target_mc_ids = extraction.run(inp)
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
    # Модули импортируют main: CLI должен использовать тот же BasePipeline.
    sys.modules["main"] = sys.modules[__name__]
    sys.exit(main())
