"""Проверки конфигурации и загрузки вариантов без изменения бенчмарков."""

import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from main import (
    PIPELINE_CLASSES,
    BasePipeline,
    PipelineConfig,
    PipelineInput,
    load_config,
    load_pipeline,
    run_pipeline,
)


def test_default_modules() -> None:
    """До появления реализаций выбор явно пуст, загрузка завершается ошибкой."""
    config = load_config()
    for service in PIPELINE_CLASSES:
        assert getattr(config, service) is None
        with pytest.raises(ValueError, match=f"Для раздела {service} не выбран модуль"):
            load_pipeline(service, config)


def test_config_path_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Корневой конфиг находится и при запуске из другого каталога."""
    monkeypatch.chdir(tmp_path)
    assert load_config().subcategory_extraction is None


def test_load_custom_config(tmp_path: Path) -> None:
    """Конфиг можно прочитать по явно переданному пути."""
    path = tmp_path / "config.json"
    payload = load_config().model_dump()
    payload["subcategory_extraction"] = "embeddings"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_config(path).subcategory_extraction == "embeddings"


@pytest.mark.parametrize(
    "changes",
    [
        {"subcategory_extraction": "../jev"},
        {"draft_generation": ""},
        {"category_validation": 123},
        {"unexpected": "jev"},
    ],
)
def test_invalid_config(changes: dict) -> None:
    """Ошибки конфигурации не маскируются значениями по умолчанию."""
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(load_config().model_dump() | changes)


def test_missing_service() -> None:
    """Все разделы должны быть заданы явно."""
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"subcategory_extraction": "jev"})


def test_missing_module() -> None:
    """Отсутствующий модуль даёт понятную ошибку."""
    config = load_config().model_copy(update={"subcategory_extraction": "missing"})
    with pytest.raises(ValueError, match="subcategory_extraction.missing.pipeline"):
        load_pipeline("subcategory_extraction", config)


def test_alternative_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Переключение реализации не требует изменения оркестрации."""
    import sys

    class AlternativePipeline(BasePipeline):
        """Тестовая реализация выделения."""

        def run(self, inp: PipelineInput) -> list[int]:
            """Вернуть категорию для проверки выбранного варианта."""
            return [104]

    module = ModuleType("subcategory_extraction.embeddings.pipeline")
    module.SubcategoryExtractionPipeline = AlternativePipeline
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class ValidationPipeline(BasePipeline):
        """Тестовая проверка передачи результатов между шагами."""

        def run(self, candidates: list[int], current_mc_id: int) -> list[int]:
            """Проверить результат выделения и передать его дальше."""
            assert candidates == [104]
            assert current_mc_id == 101
            return candidates

    class GenerationPipeline(BasePipeline):
        """Тестовая генерация для проверки оркестрации."""

        def run(self, inp: PipelineInput, target_mc_ids: list[int]) -> list[dict]:
            """Вернуть черновик для полученной целевой категории."""
            assert target_mc_ids == [104]
            return [{"mcId": 104, "title": "Сантехника", "text": inp.description}]

    for service, pipeline_class in (
        ("category_validation", ValidationPipeline),
        ("draft_generation", GenerationPipeline),
    ):
        test_module = ModuleType(f"{service}.test_variant.pipeline")
        setattr(test_module, PIPELINE_CLASSES[service], pipeline_class)
        monkeypatch.setitem(sys.modules, test_module.__name__, test_module)
    config = PipelineConfig(
        subcategory_extraction="embeddings",
        category_validation="test_variant",
        draft_generation="test_variant",
    )
    inp = PipelineInput(description="Сантехника", mcId=101)
    assert load_pipeline("subcategory_extraction", config).run(inp) == [104]
    assert run_pipeline(inp, config).model_dump(by_alias=True) == {
        "drafts": [{"mcId": 104, "title": "Сантехника", "text": "Сантехника"}]
    }
