"""Бенчмарк-прогон сервиса draft_generation.

Единственный тест сервиса: берёт входные данные из .md-бенчмарков, прогоняет
пайплайн и проверяет соответствие контракту кейса. Метрики и сравнение версий
кода фиксируются в самом .md бенчмарка (см. AGENTS.md, раздел 4).
"""

import json
import re
from pathlib import Path

import pytest

from main import PipelineInput, PipelineOutput, is_valid_mcid, load_config, load_pipeline

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
    if config.draft_generation is None:
        pytest.skip("Модуль draft_generation не выбран")
    inp = PipelineInput.model_validate(payload)

    output = PipelineOutput(
        drafts=load_pipeline("draft_generation", config).run(inp, payload["target_mc_ids"])
    )

    contract = output.model_dump(by_alias=True)  # сериализация в контракт кейса
    assert set(contract) == {"drafts"}
    for draft in output.drafts:
        assert is_valid_mcid(draft.mc_id)
        assert draft.mc_id != inp.mc_id
