"""Выделение дополнительных услуг через Google Decisions API."""

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from main import BasePipeline, GoogleDecisionClient, MICROCATEGORIES, PipelineInput

RESOURCE_DIR = Path(__file__).resolve().parent / "prompts"


class ExtractionSettings(BaseModel):
    """Промпт и порог выделения; параметры API находятся в общем клиенте."""

    model_config = ConfigDict(extra="forbid")

    prompt: Literal["baseline", "guided"] = "guided"
    threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class NoulAnswer(BaseModel):
    """Ответ на один бинарный вопрос Google без приведения строк и bool."""

    model_config = ConfigDict(extra="ignore", strict=True)

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class DecisionResponse(BaseModel):
    """Оболочка typed answers; метаданные API не влияют на выделение."""

    model_config = ConfigDict(extra="ignore", strict=True)

    answers: dict[str, NoulAnswer]


def load_guidance() -> dict[int, str]:
    """Прочитать границы категорий из ресурса и сверить с общим контрактом."""
    resource = json.loads((RESOURCE_DIR / "category_guidance.json").read_text(encoding="utf-8"))
    if not isinstance(resource, dict) or set(resource) != {str(cid) for cid in MICROCATEGORIES}:
        raise ValueError("Справочник выделения должен описывать все категории контракта")
    if any(not isinstance(value, str) or not value.strip() for value in resource.values()):
        raise ValueError("Описания границ категорий должны быть непустыми строками")
    return {cid: resource[str(cid)] for cid in MICROCATEGORIES}


class SubcategoryExtractionPipeline(BasePipeline):
    """Оценить дополнительные категории независимо в одном запросе Google."""

    name = "google_decision"

    def __init__(self) -> None:
        self.settings = ExtractionSettings(
            prompt=os.getenv("EXTRACTION_PROMPT") or "guided",
            threshold=os.getenv("EXTRACTION_THRESHOLD") or 0.5,
        )
        self.threshold = self.settings.threshold
        self.prompt_path = RESOURCE_DIR / f"google_decision_{self.settings.prompt}.md"
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        self.guidance = load_guidance()
        self.client = GoogleDecisionClient()

    def probabilities(
        self, inp: PipelineInput, categories: set[int] | None = None,
    ) -> dict[int, float]:
        """Вернуть проверенные вероятности только запрошенных других категорий."""
        if categories is not None and any(
            type(cid) is not int or cid not in MICROCATEGORIES for cid in categories
        ):
            raise ValueError("Запрошены неизвестные категории выделения")
        candidates = [
            cid for cid in sorted(MICROCATEGORIES)
            if cid != inp.mc_id and (categories is None or cid in categories)
        ]
        if not candidates or not inp.description.strip():
            return {}
        questions = {
            f"mc{cid}": {
                "type": "noul",
                "instructions": self.prompt.format(
                    category=MICROCATEGORIES[cid], guidance=self.guidance[cid],
                ),
            }
            for cid in candidates
        }
        raw = self.client.complete(inp.description, questions=questions)
        try:
            response = DecisionResponse.model_validate_json(raw)
        except ValidationError:
            raise ValueError("Google Decision вернул некорректные typed answers") from None
        if set(response.answers) != set(questions):
            raise ValueError("Google Decision вернул другой набор вопросов")
        return {cid: response.answers[f"mc{cid}"].noul for cid in candidates}

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть уникальные mcId по порогу, исключив исходную категорию."""
        return [cid for cid, probability in self.probabilities(inp).items() if probability >= self.threshold]
