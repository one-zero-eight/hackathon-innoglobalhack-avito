"""Выделение дополнительных категорий нативными вероятностями Jev."""

import hashlib
import json
import math
import os
from pathlib import Path

from main import BasePipeline, MICROCATEGORIES, PipelineInput
from subcategory_extraction.decisions.google_decision.pipeline import CATEGORY_EXAMPLES
from subcategory_extraction.decisions.jev.client import JevClient

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts/guided.md"


class SubcategoryExtractionPipeline(BasePipeline):
    """Оценить услуги общим guided-промптом через нативный API Jev."""

    name = "jev"

    def __init__(self) -> None:
        self.client = JevClient()
        path = Path(os.getenv("JEV_PROMPT_PATH", str(DEFAULT_PROMPT_PATH))).expanduser()
        self.prompt = path.read_text(encoding="utf-8")
        self.threshold = float(os.getenv("JEV_THRESHOLD", "0.5"))
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("JEV_THRESHOLD должен быть конечным числом от 0 до 1")
        self.last_probabilities: dict[int, float] = {}

    def benchmark_metadata(self) -> dict:
        """Зафиксировать провайдера, модель и промпт без API-ключа."""
        return {
            "provider": self.client.provider,
            "endpoint": self.client.endpoint,
            "model": self.client.model_id,
            "threshold": self.threshold,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "cache_enabled": True,
        }

    def probabilities(
        self, inp: PipelineInput, categories: set[int] | None = None,
    ) -> dict[int, float]:
        """Оценить все дополнительные категории или указанное подмножество."""
        if categories is not None and any(
            type(cid) is not int or cid not in MICROCATEGORIES for cid in categories
        ):
            raise ValueError("Неизвестные категории в запросе Jev")
        questions = {
            f"mc{cid}": {
                "type": "noul",
                "instructions": self.prompt.format(category=name, examples=CATEGORY_EXAMPLES[cid]),
            }
            for cid, name in MICROCATEGORIES.items()
            if cid != inp.mc_id and (categories is None or cid in categories)
        }
        if not questions:
            self.last_probabilities = {}
            return {}
        result = json.loads(self.client.complete(inp.description, questions=questions))
        self.last_probabilities = {
            int(name[2:]): float(answer["noul"])
            for name, answer in result["answers"].items()
        }
        return self.last_probabilities.copy()

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории с вероятностью не ниже фиксированного порога."""
        return [cid for cid, value in self.probabilities(inp).items() if value >= self.threshold]
