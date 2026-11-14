"""Локальный inference общего encoder с зафиксированными порогами категорий."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from main import PipelineInput
from subcategory_extraction.semantic_encoder.pipeline import (
    SubcategoryExtractionPipeline as SemanticEncoderPipeline,
)

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "out/rubert-thresholds/encoder"


class SubcategoryExtractionPipeline(SemanticEncoderPipeline):
    """Применить пороги validation, не меняя веса и расчёт вероятностей encoder."""

    name = "semantic_encoder_calibrated"

    def __init__(self, artifact_path: Path | None = None) -> None:
        if artifact_path is None:
            artifact_path = Path(os.getenv(
                "SEMANTIC_ENCODER_ARTIFACT_PATH", str(DEFAULT_ARTIFACT),
            ))
        super().__init__(artifact_path=artifact_path)
        calibration = json.loads(
            (self.artifact_path / "calibration.json").read_text(encoding="utf-8"),
        )
        if not isinstance(calibration, dict):
            raise ValueError("calibration.json должен содержать JSON-объект")
        if type(calibration.get("version")) is not int or calibration["version"] != 1:
            raise ValueError("Неподдерживаемая версия calibration.json")
        metadata_sha256 = hashlib.sha256(
            (self.artifact_path / "metadata.json").read_bytes(),
        ).hexdigest()
        if calibration.get("artifact_metadata_sha256") != metadata_sha256:
            raise ValueError("calibration.json не соответствует metadata.json encoder")
        if calibration.get("selected_method") not in ("fixed", "global", "per_category"):
            raise ValueError("Неизвестный метод подбора порогов calibration.json")
        thresholds = calibration.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != {
            str(category) for category in self.categories
        }:
            raise ValueError("Пороги calibration.json должны охватывать все классы encoder")
        for category, threshold in thresholds.items():
            if (
                type(threshold) not in (int, float)
                or not 0 <= threshold <= 1
                or not math.isfinite(threshold)
            ):
                raise ValueError(
                    f"Порог категории {category} должен быть конечным числом от 0 до 1",
                )
        self.thresholds: dict[int, float] = {
            int(category): float(threshold) for category, threshold in thresholds.items()
        }
        self._calibration = calibration

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Вернуть копию метаданных encoder вместе с происхождением порогов."""
        metadata = super().model_metadata
        metadata["calibration"] = deepcopy(self._calibration)
        return metadata

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть дополнительные категории с вероятностью не ниже своего порога."""
        return [
            category for category, score in self.probabilities(inp).items()
            if score >= self.thresholds[category]
        ]
