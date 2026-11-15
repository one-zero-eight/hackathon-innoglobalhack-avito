"""Полный смысловой арбитраж по общим категориям поверх нативного Jev."""

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from main import BasePipeline, McId, MICROCATEGORIES, PipelineInput, StructuredLLMClient
from subcategory_extraction.decisions.jev_semantic.pipeline import DEFINITIONS, SubcategoryExtractionPipeline as JevPipeline

PROMPT = Path(__file__).with_name("prompts") / "adjudicate.md"


class Decision(BaseModel):
    """Проверенный бинарный ответ для одной категории."""

    model_config = ConfigDict(extra="forbid", strict=True)
    mcId: McId
    offered: bool


def response_model(categories: set[int]) -> type[BaseModel]:
    """Потребовать ровно один ответ на каждую eligible-категорию до кэширования."""
    class Response(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        decisions: list[Decision]

        @model_validator(mode="after")
        def complete_categories(self):
            ids = [item.mcId for item in self.decisions]
            if len(ids) != len(categories) or set(ids) != categories:
                raise ValueError("Ответ арбитра должен содержать ровно все переданные категории")
            return self
    return Response


class SubcategoryExtractionPipeline(BasePipeline):
    """Перепроверить все категории по полному тексту, не только вероятные ошибки Jev."""

    name = "jev_semantic_arbiter"

    def __init__(self) -> None:
        self.baseline = JevPipeline()
        self.arbiter = StructuredLLMClient()
        self.prompt = PROMPT.read_text()
        self.last_probabilities: dict[int, float] = {}

    def run(self, inp: PipelineInput) -> list[int]:
        """Передать модели общие критерии, полный текст и необязательные по смыслу prior-оценки."""
        categories = set(MICROCATEGORIES) - {inp.mc_id}
        prior = self.baseline.probabilities(inp)
        schema = response_model(categories)
        payload = json.dumps({"description": inp.description, "source_mcId": inp.mc_id,
                              "definitions": [{"mcId": cid, "name": MICROCATEGORIES[cid], "definition": DEFINITIONS[cid]}
                                              for cid in sorted(categories)],
                              "auxiliary_probabilities": prior}, ensure_ascii=False)
        answer = schema.model_validate_json(self.arbiter.complete(payload, response_model=schema, system_prompt=self.prompt))
        self.last_probabilities = {item.mcId: float(item.offered) for item in answer.decisions}
        return sorted(item.mcId for item in answer.decisions if item.offered)

    def benchmark_metadata(self) -> dict:
        """Зафиксировать обе модели и смысл бинарного вывода без секретов."""
        return {"native": self.baseline.benchmark_metadata(), "arbiter_model": self.arbiter.model_id,
                "schema_in_system_prompt": os.getenv("STRUCTURED_LLM_SCHEMA_IN_PROMPT", "0") == "1",
                "arbiter_endpoint": self.arbiter.endpoint, "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
                "decision_type": "structured binary decisions; not calibrated probabilities", "full_context": True}
