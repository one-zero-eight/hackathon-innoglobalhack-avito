"""HTTP-адаптер полного пайплайна с одинаковым выбором модулей обоих шагов."""

from enum import Enum
import logging
from pathlib import Path
from urllib.error import URLError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from main import (
    DEFAULT_CONFIG_PATH,
    LLMCostLedger,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineRuntime,
    PipelineUnavailableError,
    discover_modules,
    load_config,
    track_llm_costs,
)

logger = logging.getLogger(__name__)


class ModuleDefaults(BaseModel):
    """Имена модулей, используемые при отсутствии query-параметров."""

    extractor: str
    generator: str


class ModuleCatalog(BaseModel):
    """Обнаруженные реализации; наличие в списке не гарантирует готовность ресурсов."""

    extractors: list[str]
    generators: list[str]
    defaults: ModuleDefaults


def create_app(config_path: Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    """Подготовить сервер без импорта моделей, загрузки весов и вызовов провайдеров."""
    load_dotenv(DEFAULT_CONFIG_PATH.parent / ".env")
    config = load_config(config_path)
    extractors = discover_modules("subcategory_extraction")
    generators = discover_modules("draft_generation")
    if config.subcategory_extraction not in extractors or config.draft_generation not in generators:
        raise ValueError("В config.json должны быть выбраны существующие extractor и generator")
    catalog = ModuleCatalog(
        extractors=list(extractors), generators=list(generators),
        defaults=ModuleDefaults(extractor=config.subcategory_extraction, generator=config.draft_generation),
    )
    # Enum обеспечивает и проверку допустимых значений, и dropdown в Swagger.
    extractor_choice = Enum("Extractor", {name.replace(".", "_"): name for name in extractors}, type=str)
    generator_choice = Enum("Generator", {name: name for name in generators}, type=str)
    runtime = PipelineRuntime()
    app = FastAPI(
        title="Avito draft generation", version="1.0.0",
        description="Полный пайплайн: выбранный extractor → выбранный generator. Допустимые mcId: 101–106.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Проверить HTTP-сервер без проверки внешних моделей и ресурсов."""
        return {"status": "ok"}

    @app.get("/modules", response_model=ModuleCatalog)
    def modules() -> ModuleCatalog:
        """Вернуть имена модулей и defaults; секреты и параметры провайдеров не публикуются."""
        return catalog

    @app.post(
        "/drafts", response_model=PipelineOutput, response_model_by_alias=True,
        responses={
            502: {"description": "Ошибка провайдера или некорректный результат модели"},
            503: {"description": "Не настроены ресурсы, зависимости или ключ доступа модуля"},
        },
    )
    def create_drafts(
        inp: PipelineInput,
        extractor: extractor_choice | None = Query(default=None, description="Модуль extractor; без значения — default из config.json"),
        generator: generator_choice | None = Query(default=None, description="Модуль generator; без значения — default из config.json"),
    ) -> PipelineOutput:
        """Выполнить запрос в thread pool; runtime сериализует загрузку и оба шага."""
        selected = PipelineConfig(
            subcategory_extraction=extractor.value if extractor is not None else config.subcategory_extraction,
            draft_generation=generator.value if generator is not None else config.draft_generation,
        )
        try:
            ledger = LLMCostLedger()
        except (OSError, ValueError) as exc:
            logger.warning("Не настроен учёт API: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Не настроен учёт API; проверьте конфигурацию сервера") from None
        try:
            # Контекст отдельного запроса не накапливает usage в переиспользуемых клиентах.
            with track_llm_costs(ledger):
                return runtime.run(inp, selected)
        except PipelineUnavailableError as exc:
            logger.warning("Модуль недоступен: %s/%s (%s)", exc.service, exc.module, type(exc.__cause__).__name__)
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except (ValueError, RuntimeError, URLError, TimeoutError, ConnectionError) as exc:
            logger.warning("Ошибка выполнения пайплайна: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Ошибка провайдера или некорректный результат модели") from None

    return app
