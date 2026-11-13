"""Фиксированная маршрутизация Jev/TF-IDF к сильному JSON Schema арбитру."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from main import BasePipeline, McId, MICROCATEGORIES, PipelineInput, StructuredLLMClient
from subcategory_extraction.google_decision.pipeline import CATEGORY_EXAMPLES
from subcategory_extraction.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline
from subcategory_extraction.tfidf.pipeline import SubcategoryExtractionPipeline as TfidfPipeline

PROMPT_PATH = Path(__file__).resolve().parent / "prompts/arbitrate.md"
THRESHOLD = 0.5
UNCERTAINTY_LOW = 0.30
UNCERTAINTY_HIGH = 0.70


class Decision(BaseModel):
    """Бинарный вердикт арбитра, не калиброванная вероятность."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mc_id: McId = Field(alias="mcId")
    offered: bool
    quote: str


class ArbitrationResponse(BaseModel):
    """Ровно по одному решению на переданную арбитру категорию."""

    model_config = ConfigDict(extra="forbid", strict=True)

    decisions: list[Decision]


def _response_model(
    inp: PipelineInput, routed: set[int],
) -> type[ArbitrationResponse]:
    """Проверить точный набор категорий и цитаты ещё до кэширования ответа."""

    class SourceArbitrationResponse(ArbitrationResponse):
        @model_validator(mode="after")
        def validate_decisions(self) -> Self:
            """Не принимать пропущенные категории, дубликаты и ложные цитаты."""
            ids = [item.mc_id for item in self.decisions]
            if len(ids) != len(routed) or set(ids) != routed:
                raise ValueError("Набор решений арбитра должен точно совпадать с routed")
            for item in self.decisions:
                if item.offered and not item.quote.strip():
                    raise ValueError("Положительный вердикт требует непустой цитаты")
                if item.quote and (
                    not item.quote.strip() or item.quote not in inp.description
                ):
                    raise ValueError("Цитата арбитра должна быть точной подстрокой description")
            return self

    return SourceArbitrationResponse


class SubcategoryExtractionPipeline(BasePipeline):
    """Заменить только спорные решения Jev независимыми решениями сильной LLM."""

    name = "jev_arbiter"

    def __init__(self) -> None:
        self.threshold = THRESHOLD
        self.uncertainty_low = UNCERTAINTY_LOW
        self.uncertainty_high = UNCERTAINTY_HIGH
        self.baseline = JevPipeline()
        self.baseline.threshold = self.threshold
        self.local = TfidfPipeline()
        self.arbiter = StructuredLLMClient()
        self.arbitration_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.last_routed_categories: list[int] = []
        self.last_arbiter_used = False
        self.last_baseline_probabilities: dict[int, float] = {}
        self.last_local_probabilities: dict[int, float] = {}
        self.last_decisions: list[dict[str, int | bool | str]] = []

    def run(self, inp: PipelineInput) -> list[int]:
        """Маршрутизировать спор либо Jev p∈[0.30,0.70], затем объединить решения."""
        self.last_routed_categories = []
        self.last_arbiter_used = False
        self.last_baseline_probabilities = {}
        self.last_local_probabilities = {}
        self.last_decisions = []
        jev = self.baseline.probabilities(inp)
        self.last_baseline_probabilities = jev
        self.last_local_probabilities = {
            cid: value for cid, value in self.local.probabilities(inp).items()
            if cid in self.local.trained_categories
        }
        self.last_routed_categories = [
            cid for cid, value in jev.items()
            if self.uncertainty_low <= value <= self.uncertainty_high or (
                cid in self.last_local_probabilities
                and (value >= self.threshold)
                != (self.last_local_probabilities[cid] >= self.threshold)
            )
        ]
        selected = {cid for cid, value in jev.items() if value >= self.threshold}
        if not self.last_routed_categories:
            return [cid for cid in MICROCATEGORIES if cid in selected]
        categories = [
            {"mcId": cid, "name": MICROCATEGORIES[cid], "examples": CATEGORY_EXAMPLES[cid]}
            for cid in self.last_routed_categories
        ]
        arbitration_input = json.dumps(
            {"description": inp.description, "mcId": inp.mc_id, "categories": categories},
            ensure_ascii=False,
        )
        routed = set(self.last_routed_categories)
        response_model = _response_model(inp, routed)
        self.last_arbiter_used = True
        response = response_model.model_validate_json(self.arbiter.complete(
            arbitration_input,
            response_model=response_model,
            system_prompt=self.arbitration_prompt,
        ))
        self.last_decisions = [item.model_dump(by_alias=True) for item in response.decisions]
        selected.difference_update(routed)
        selected.update(item.mc_id for item in response.decisions if item.offered)
        return [cid for cid in MICROCATEGORIES if cid in selected]
