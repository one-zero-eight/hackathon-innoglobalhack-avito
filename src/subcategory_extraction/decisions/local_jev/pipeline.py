"""Выделение дополнительных микрокатегорий локальной моделью jevos."""

import hashlib
import json
import math
import os
import platform
from pathlib import Path

from main import BasePipeline, MICROCATEGORIES, PipelineInput
from subcategory_extraction.decisions.google_decision.pipeline import CATEGORY_EXAMPLES
from subcategory_extraction.decisions.local_jev.client import LocalJevClient

DEFAULT_PROMPT = Path(__file__).resolve().parent / "prompts/guided_ru.md"
LOCAL_PROMPTS = Path(__file__).resolve().parent / "prompts"


class SubcategoryExtractionPipeline(BasePipeline):
    """Оценить услуги локальным jevos с выбранным до test языком вопросов."""

    name = "local_jev"

    def __init__(self) -> None:
        self.variant = os.getenv("LOCAL_JEV_PROMPT", "english")
        if self.variant not in ("english", "guided_ru"):
            raise ValueError("LOCAL_JEV_PROMPT: допустимы english и guided_ru")
        default_path = LOCAL_PROMPTS / "english.md" if self.variant == "english" else DEFAULT_PROMPT
        self.prompt_path = Path(os.getenv("LOCAL_JEV_PROMPT_PATH", str(default_path))).expanduser()
        self.prompt = self.prompt_path.read_text(encoding="utf-8")
        if self.variant == "english":
            self.prompt = self.prompt.strip()
        self.examples = (
            {int(cid): value for cid, value in json.loads(
                (LOCAL_PROMPTS / "english-examples.json").read_text(encoding="utf-8"),
            ).items()}
            if self.variant == "english" else CATEGORY_EXAMPLES
        )
        if set(self.examples) != set(MICROCATEGORIES):
            raise ValueError("Примеры локального Jev должны покрывать категории контракта")
        self.threshold = float(os.getenv("LOCAL_JEV_THRESHOLD", "0.5"))
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("LOCAL_JEV_THRESHOLD должен быть конечным числом от 0 до 1")
        self.client = LocalJevClient()
        self.last_probabilities: dict[int, float] = {}

    def run(self, inp: PipelineInput) -> list[int]:
        """Оценить все категории кроме исходной одним настоящим запросом."""
        self.last_probabilities = {}
        questions = {
            f"mc{cid}": {"type": "noul", "instructions": self.prompt.format(
                category=name, examples=self.examples[cid],
            )}
            for cid, name in MICROCATEGORIES.items() if cid != inp.mc_id
        }
        result = json.loads(self.client.complete(inp.description, questions=questions))
        self.last_probabilities = {
            int(name[2:]): float(answer["noul"]) for name, answer in result["answers"].items()
        }
        return [cid for cid, value in self.last_probabilities.items() if value >= self.threshold]

    def benchmark_metadata(self) -> dict:
        """Зафиксировать сервер, веса, промпт и порог до оценки на test."""
        return {
            "source": "https://github.com/feder-cr/jev/releases/tag/jevos-v5",
            "model": self.client.model_id,
            "server": self.client.health,
            "endpoint": self.client.endpoint,
            "prompt_path": str(self.prompt_path.relative_to(Path(__file__).resolve().parents[4]))
            if self.prompt_path.is_relative_to(Path(__file__).resolve().parents[4])
            else str(self.prompt_path),
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "prompt_variant": self.variant,
            "category_examples": self.examples,
            "threshold": self.threshold,
            "task_training": "none; upstream pretrained weights, fixed prompt and threshold",
            "upstream_training_overlap": "unknown; upstream training data not available for audit",
            "client_response_cache": False,
            "client_retries": False,
            "platform": platform.platform(),
        }
