"""Взвешивание вероятностей Jev и обученных локальных голов TF-IDF."""

import math
import os

from main import BasePipeline, PipelineInput
from subcategory_extraction.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline
from subcategory_extraction.tfidf.pipeline import SubcategoryExtractionPipeline as TfidfPipeline


class SubcategoryExtractionPipeline(BasePipeline):
    """Объединить вероятности, сохранив Jev для необученных локально категорий."""

    name = "jev_tfidf"

    def __init__(self) -> None:
        self.weight = float(os.getenv("JEV_TFIDF_WEIGHT", "0.5"))
        self.threshold = float(os.getenv("JEV_TFIDF_THRESHOLD", "0.5"))
        if not math.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("JEV_TFIDF_WEIGHT должен быть конечным числом от 0 до 1")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("JEV_TFIDF_THRESHOLD должен быть конечным числом от 0 до 1")
        self.jev = JevPipeline()
        self.tfidf = TfidfPipeline()

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Взвесить одинаковые целевые категории без исходной микрокатегории."""
        jev_probabilities = self.jev.probabilities(inp)
        local_probabilities = self.tfidf.probabilities(inp)
        return {
            cid: (
                self.weight * probability + (1 - self.weight) * local_probabilities[cid]
                if cid in self.tfidf.trained_categories else probability
            )
            for cid, probability in jev_probabilities.items()
        }

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории с объединённой вероятностью не ниже порога."""
        return [cid for cid, value in self.probabilities(inp).items() if value >= self.threshold]
