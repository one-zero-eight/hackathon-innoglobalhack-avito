"""Нативный Jev с фиксированными границами официальных микрокатегорий."""

from pathlib import Path

from subcategory_extraction.decisions.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/boundaries.md"


class SubcategoryExtractionPipeline(JevPipeline):
    """Использовать статический справочник без эвристических правок ответа."""

    name = "jev_boundaries"

    def __init__(self) -> None:
        super().__init__()
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
