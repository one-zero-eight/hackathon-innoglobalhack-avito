"""Jev с зафиксированными промптом и train-калиброванными порогами."""

import hashlib
import json
import math
import os
from pathlib import Path

from main import MICROCATEGORIES, PipelineInput
from subcategory_extraction.jev.pipeline import SubcategoryExtractionPipeline as JevPipeline

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = MODULE_DIR / "artifacts/calibration.json"
PROMPTS = ("guided", "baseline")


class SubcategoryExtractionPipeline(JevPipeline):
    """Использовать тот же Jev API и проверенную калибровку выбранной модели."""

    name = "jev_calibrated"

    def __init__(self) -> None:
        super().__init__()
        variant = os.getenv("JEV_CALIBRATED_PROMPT", "auto")
        self.thresholds = dict.fromkeys(MICROCATEGORIES, self.threshold)
        self.calibration_sha256: str | None = None
        self.calibration_data: dict = {}
        if variant == "auto":
            path = Path(os.getenv("JEV_CALIBRATION_PATH", str(DEFAULT_CALIBRATION))).expanduser()
            if not path.is_file():
                raise ValueError("Нет калибровки Jev: запустите jev_calibrated/train.py")
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("version") != 1 or data.get("control_confirmed") is not True
                    or data.get("model") != self.client.model_id
                    or data.get("endpoint") != self.client.endpoint
                    or data.get("provider") != self.client.provider):
                raise ValueError("Калибровка Jev относится к другой модели или провайдеру")
            variant = data.get("prompt")
            thresholds = data.get("thresholds")
            if (not isinstance(thresholds, dict)
                    or set(thresholds) != {str(cid) for cid in MICROCATEGORIES}
                    or any(type(t) not in (int, float) or not math.isfinite(t)
                           or not 0 <= t <= 1 for t in thresholds.values())):
                raise ValueError("Некорректные пороги калибровки Jev")
            self.thresholds = {int(cid): float(value) for cid, value in thresholds.items()}
            self.calibration_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.calibration_data = data
        if variant not in PROMPTS:
            raise ValueError("JEV_CALIBRATED_PROMPT: допустимы auto, guided, baseline")
        self.prompt_variant = variant
        path = MODULE_DIR / "prompts" / f"{variant}.md"
        self.prompt = path.read_text(encoding="utf-8")
        self.prompt_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.calibration_sha256 and self.prompt_sha256 != self.calibration_data.get("prompt_sha256"):
            raise ValueError("Промпт Jev изменился после калибровки")

    def run(self, inp: PipelineInput) -> list[int]:
        """Применить порог каждой категории без изменения исходных вероятностей."""
        values = self.probabilities(inp)
        return [cid for cid, value in values.items() if value >= self.thresholds[cid]]

    def benchmark_metadata(self) -> dict:
        """Записать фактические модель, промпт и калибровку без секретов."""
        source = self.calibration_data.get("training_dataset")
        training_table = (
            "case_train" if source == "docs/innoglobalhack_avito/public-train-annotated.jsonl"
            else source
        )
        return {
            "provider": self.client.provider, "endpoint": self.client.endpoint,
            "model": self.client.model_id, "prompt": self.prompt_variant,
            "prompt_sha256": self.prompt_sha256, "thresholds": self.thresholds,
            "calibration_sha256": self.calibration_sha256,
            "training_dataset": training_table,
            "calibration_training_source": source,
            "fit_description_sha256": self.calibration_data.get("fit_description_sha256"),
            "control_confirmed": self.calibration_data.get("control_confirmed"),
            "cache_enabled": True,
        }
