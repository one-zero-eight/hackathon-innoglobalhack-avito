"""Разметка всех категорий внешнего описания без исходного mcId."""

import json

from main import MICROCATEGORIES
from subcategory_extraction.decisions.google_decision.pipeline import (
    CATEGORY_EXAMPLES,
    SubcategoryExtractionPipeline,
)


class CollectionAnnotationPipeline(SubcategoryExtractionPipeline):
    """Переиспользовать клиент и промпт классификатора для разметки коллекций."""

    def score_description(self, description: str) -> dict[int, float]:
        """Запросить и проверить оценки всех шести категорий."""
        questions = {
            f"mc{cid}": {
                "type": "noul",
                "instructions": self.prompt.format(category=name, examples=CATEGORY_EXAMPLES[cid]),
            }
            for cid, name in MICROCATEGORIES.items()
        }
        result = json.loads(self.client.complete(description, questions=questions))
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ValueError("Google Decision: отсутствует словарь answers")
        if set(result["answers"]) != set(questions):
            raise ValueError("Набор ответов не совпадает с вопросами")
        probabilities: dict[int, float] = {}
        for name, answer in result["answers"].items():
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                raise ValueError(f"Некорректная вероятность для {name}")
            probabilities[int(name[2:])] = float(probability)
        return probabilities
