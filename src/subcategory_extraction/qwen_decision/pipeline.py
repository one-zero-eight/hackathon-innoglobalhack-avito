"""Qwen Decision: независимые вопросы по услугам и train-калиброванные пороги."""

import hashlib
import json
import math
import os
from pathlib import Path

from main import BasePipeline, MICROCATEGORIES, PipelineInput
from subcategory_extraction.qwen_decision.client import QwenDecisionClient

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = MODULE_DIR / "artifacts/calibration.json"
PROMPTS = ("guided", "boundaries")


class SubcategoryExtractionPipeline(BasePipeline):
    """Оценить все дополнительные услуги, исключив исходную микрокатегорию."""

    name = "qwen_decision"

    def __init__(self) -> None:
        self.client = QwenDecisionClient()
        variant = os.getenv("QWEN_DECISION_PROMPT", "auto")
        self.thresholds = dict.fromkeys(MICROCATEGORIES, 0.5)
        self.calibration_sha256: str | None = None
        self.last_probabilities: dict[int, float] = {}
        if variant == "auto":
            path = Path(os.getenv("QWEN_DECISION_CALIBRATION", str(DEFAULT_CALIBRATION))).expanduser()
            if not path.is_file():
                raise ValueError("Нет train-калибровки Qwen; запустите train.py или задайте QWEN_DECISION_PROMPT=guided")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["model"] != self.client.model_id or data["endpoint"] != self.client.endpoint:
                raise ValueError("Калибровка относится к другой модели/endpoint Qwen")
            variant = data["prompt"]
            thresholds = data["thresholds"]
            if set(thresholds) != {str(cid) for cid in MICROCATEGORIES} or any(
                type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1
                for t in thresholds.values()
            ):
                raise ValueError("Некорректные пороги Qwen")
            self.thresholds = {int(cid): float(t) for cid, t in thresholds.items()}
            self.calibration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if variant not in PROMPTS:
            raise ValueError("QWEN_DECISION_PROMPT: допустимы auto, guided, boundaries")
        self.prompt_variant = variant
        prompt_path = MODULE_DIR / "prompts" / f"{variant}.md"
        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if self.calibration_sha256 and self.prompt_sha256 != data["prompt_sha256"]:
            raise ValueError("Промпт изменён после калибровки Qwen")

    def probabilities(self, inp: PipelineInput, categories: set[int] | None = None) -> dict[int, float]:
        """Получить P(yes) независимо для каждой запрошенной дополнительной категории."""
        if categories is not None and any(type(cid) is not int or cid not in MICROCATEGORIES for cid in categories):
            raise ValueError("Неизвестные микрокатегории Qwen")
        questions = {
            f"mc{cid}": {"type": "noul", "instructions": self.prompt.format(category=name)}
            for cid, name in MICROCATEGORIES.items()
            if cid != inp.mc_id and (categories is None or cid in categories)
        }
        if not questions:
            return {}
        result = json.loads(self.client.complete(inp.description, questions=questions))
        return {int(name[2:]): float(answer["noul"]) for name, answer in result["answers"].items()}

    def run(self, inp: PipelineInput) -> list[int]:
        """Применить зафиксированные до test пороги; не создавать категорий вне контракта."""
        self.last_probabilities = self.probabilities(inp)
        return [cid for cid, probability in self.last_probabilities.items() if probability >= self.thresholds[cid]]

    def benchmark_metadata(self) -> dict:
        """Зафиксировать фактическую модель, промпт и пороги для общего runner."""
        return {
            "model": self.client.model_id, "endpoint": self.client.endpoint,
            "prompt": self.prompt_variant, "prompt_sha256": self.prompt_sha256,
            "thresholds": self.thresholds, "calibration_sha256": self.calibration_sha256,
            "training_dataset": "docs/improved_dataset/train.jsonl" if self.calibration_sha256 else None,
        }
