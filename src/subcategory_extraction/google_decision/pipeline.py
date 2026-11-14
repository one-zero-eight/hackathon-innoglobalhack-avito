"""Адаптер промптов эксперимента к интерфейсу выделения категорий."""

import json
import os
from pathlib import Path

from main import BasePipeline, MICROCATEGORIES, PipelineInput
from subcategory_extraction.google_decision.client import GoogleDecisionClient

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
CATEGORY_EXAMPLES = {
    101: "комплексный ремонт помещения в несколько этапов",
    102: "штукатурка, выравнивание штукатуркой, штукатурка откосов",
    103: "укладка плитки, керамогранита или мозаики, запил и затирка швов",
    104: "водоснабжение, канализация, сантехприборы, водяное отопление",
    105: "проводка, розетки, щиты, светильники, электрический тёплый пол",
    106: "обои, окраска, подготовка под обои или краску",
}


class SubcategoryExtractionPipeline(BasePipeline):
    """Определить дополнительные категории по вероятностям Google Decision."""

    name = "google_decision"

    def __init__(self) -> None:
        self.client = GoogleDecisionClient()
        variant = os.getenv("GOOGLE_DECISION_PROMPT", "guided")
        if variant not in ("baseline", "guided"):
            raise ValueError("GOOGLE_DECISION_PROMPT: допустимы baseline и guided")
        self.prompt = (PROMPT_DIR / f"google_decision_{variant}.md").read_text(encoding="utf-8")
        self.threshold = float(os.getenv("GOOGLE_DECISION_THRESHOLD", "0.5"))
        if not 0 <= self.threshold <= 1:
            raise ValueError("GOOGLE_DECISION_THRESHOLD должен быть от 0 до 1")

    def probabilities(
        self, inp: PipelineInput, categories: set[int] | None = None,
    ) -> dict[int, float]:
        """Оценить все дополнительные категории или заданное подмножество."""
        return self.score_description(inp.description, categories, current_mc_id=inp.mc_id)

    def score_description(
        self, description: str, categories: set[int] | None = None,
        *, current_mc_id: int | None = None,
    ) -> dict[int, float]:
        """Оценить текст, в том числе без исходной микрокатегории."""
        if categories is not None and any(type(cid) is not int or cid not in MICROCATEGORIES for cid in categories):
            raise ValueError("Неизвестные категории в запросе")
        questions = {
            f"mc{cid}": {
                "type": "noul",
                "instructions": self.prompt.format(category=name, examples=CATEGORY_EXAMPLES[cid]),
            }
            for cid, name in MICROCATEGORIES.items()
            if cid != current_mc_id and (categories is None or cid in categories)
        }
        if not questions:
            return {}
        result = json.loads(self.client.complete(description, questions=questions))
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ValueError("Google Decision: отсутствует словарь answers")
        answers = result["answers"]
        if set(answers) != set(questions):
            raise ValueError("Набор ответов Gemini не совпадает с вопросами")
        probabilities: dict[int, float] = {}
        for name, answer in answers.items():
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if type(probability) not in (float, int) or not 0 <= probability <= 1:
                raise ValueError(f"Некорректная вероятность для {name}")
            probabilities[int(name[2:])] = float(probability)
        return probabilities

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории с вероятностью не ниже порога."""
        return [cid for cid, value in self.probabilities(inp).items() if value >= self.threshold]
