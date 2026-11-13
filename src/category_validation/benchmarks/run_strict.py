"""Проверить контракт strict на данных общего Markdown-бенчмарка."""

import json
import logging
from pathlib import Path

from main import PipelineConfig, load_pipeline

SOURCE = Path(__file__).with_name("strict.md")


def main() -> None:
    """Загрузить выбранный вариант и проверить случаи из Markdown."""
    cases = json.loads(SOURCE.read_text(encoding="utf-8").split("```json\n")[1].split("```")[0])
    pipeline = load_pipeline("category_validation", PipelineConfig(
        subcategory_extraction=None, category_validation="strict", draft_generation=None,
    ))
    for case in cases:
        output = pipeline.run(case["candidates"], case["current"])
        if output != case["expected"]:
            raise ValueError(f"Неверный результат для {case}: {output}")
    for candidates, current in (([], 999), ([], True), (None, 101)):
        try:
            pipeline.run(candidates, current)
        except ValueError:
            continue
        raise ValueError("Некорректный вход не отклонён")
    logging.info("strict: %d/%d случаев, 3/3 некорректных входа отклонены",
                 len(cases), len(cases))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
