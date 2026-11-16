"""Адаптер remont_mc v0.4 к интерфейсу выделения микрокатегорий."""

from pathlib import Path

from main import BasePipeline, PipelineInput

from .src import mc_pipeline
from .src.student import Student

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


class SubcategoryExtractionPipeline(BasePipeline):
    """Выделить дополнительные услуги правилами и RuModernBERT v0.4."""

    name = "remont_mc_v04"

    def __init__(self) -> None:
        self.student = Student(ARTIFACTS)

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть найденные категории, исключив исходную."""
        result = mc_pipeline.predict([inp.description], student=self.student)[0]
        return [cid for cid in result["mcId"] if cid != inp.mc_id]
