"""Бенчмарк-прогон сервиса subcategory_extraction.

Единственный тест сервиса: берёт входные данные из .md-бенчмарков, прогоняет
пайплайн и проверяет соответствие контракту кейса. Метрики и сравнение версий
кода фиксируются в самом .md бенчмарка (см. AGENTS.md, раздел 4).
"""

import json
import re
from pathlib import Path

import pytest

from main import PipelineInput, is_valid_mcid, load_config, load_pipeline

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"
JSON_BLOCK = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def load_benchmark_cases() -> list[tuple[str, dict]]:
    """Входные данные всех .md-бенчмарков сервиса: (файл, payload)."""
    cases: list[tuple[str, dict]] = []
    for md_path in sorted(BENCHMARKS_DIR.glob("*.md")):
        for block in JSON_BLOCK.findall(md_path.read_text(encoding="utf-8")):
            cases.append((md_path.name, json.loads(block)))
    return cases


@pytest.mark.parametrize(("benchmark", "payload"), load_benchmark_cases())
def test_benchmark(benchmark: str, payload: dict) -> None:
    config = load_config()
    if config.subcategory_extraction is None:
        pytest.skip("Модуль subcategory_extraction не выбран")
    inp = PipelineInput.model_validate(payload)

    result = load_pipeline("subcategory_extraction", config).run(inp)

    assert isinstance(result, list)
    assert all(isinstance(mcid, int) and is_valid_mcid(mcid) for mcid in result)
