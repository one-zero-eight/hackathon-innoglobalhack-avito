"""Бенчмарк-прогон сервиса category_validation.

Единственный тест сервиса: берёт входные данные из .md-бенчмарков, прогоняет
пайплайн и проверяет соответствие контракту кейса. Метрики и сравнение версий
кода фиксируются в самом .md бенчмарка (см. AGENTS.md, раздел 4).
"""

import json
import re
from pathlib import Path

import pytest

from main import is_valid_mcid, load_config, load_pipeline

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
    if config.category_validation is None:
        pytest.skip("Модуль category_validation не выбран")
    current_mc_id = payload["current_mc_id"]

    result = load_pipeline("category_validation", config).run(payload["candidates"], current_mc_id)

    assert isinstance(result, list)
    assert all(isinstance(mcid, int) and is_valid_mcid(mcid) for mcid in result)
    assert len(result) == len(set(result)), "дубли микрокатегорий"
    assert all(mcid != current_mc_id for mcid in result), "исходная категория не исключается"
