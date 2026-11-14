"""CPU-инференс студента с validation-калибровкой, привязанной к артефакту."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from main import BasePipeline, PipelineInput
from subcategory_extraction.semantic_encoder_f1.model import (
    collate_documents,
    load_artifact,
    tokenize_document,
)

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "out/rubert-f1-plan/selected"


class SubcategoryExtractionPipeline(BasePipeline):
    """Загрузить студента и пороги категорий, не требуя учителя."""

    name = "semantic_encoder_f1"

    def __init__(self, artifact_path: Path | None = None) -> None:
        if artifact_path is None:
            artifact_path = Path(os.getenv("RUBERT_F1_ARTIFACT_PATH", str(DEFAULT_ARTIFACT)))
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        requested_device = os.getenv("RUBERT_F1_DEVICE", "cpu")
        if requested_device != "cpu":
            raise ValueError("RUBERT_F1_DEVICE must be cpu: production inference is CPU-only")
        self._device = torch.device("cpu")
        self._model, self._tokenizer, self._metadata = load_artifact(
            self.artifact_path, self._device,
        )
        self._model.requires_grad_(False)
        self.categories = tuple(self._metadata["classes"])
        trained = self._metadata.get("trained_classes")
        if trained is None:
            raise ValueError("Production artifact must record trained_classes")
        self._trained_categories = frozenset(trained)
        calibration = json.loads(
            (self.artifact_path / "calibration.json").read_text(encoding="utf-8"),
        )
        if not isinstance(calibration, dict):
            raise ValueError("calibration.json must contain an object")
        if type(calibration.get("version")) is not int or calibration["version"] != 1:
            raise ValueError("Unsupported calibration version")
        metadata_sha256 = hashlib.sha256(
            (self.artifact_path / "metadata.json").read_bytes(),
        ).hexdigest()
        if calibration.get("artifact_metadata_sha256") != metadata_sha256:
            raise ValueError("calibration.json does not match artifact metadata SHA256")
        selected_method = calibration.get("selected_method")
        if not isinstance(selected_method, str) or not selected_method.strip():
            raise ValueError("Calibration selected_method must be a nonempty provenance string")
        thresholds = calibration.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != {
            str(category) for category in self.categories
        }:
            raise ValueError("Calibration thresholds must cover every artifact class")
        for category, threshold in thresholds.items():
            if (
                type(threshold) not in (int, float)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(f"Category {category} threshold must be finite and in [0, 1]")
        self.thresholds = {
            int(category): float(threshold) for category, threshold in thresholds.items()
        }
        self._calibration = calibration

    @property
    def trained_categories(self) -> frozenset[int]:
        """Сообщить классы с положительными обучающими примерами."""
        return self._trained_categories

    @property
    def unsupported_categories(self) -> frozenset[int]:
        """Показать отсутствие положительных примеров, включая101, не обнуляя score."""
        return frozenset(self.categories) - self.trained_categories

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Вернуть независимую копию происхождения весов и порогов."""
        metadata = deepcopy(self._metadata)
        metadata["calibration"] = deepcopy(self._calibration)
        return metadata

    @property
    def device(self) -> str:
        """Сообщить фактическое устройство production inference."""
        return str(self._device)

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Оценить дополнительные категории без обнуления неподдержанных выходов."""
        windows = tokenize_document(
            self._tokenizer, inp.description, self._metadata["architecture"],
        )
        batch = collate_documents(self._tokenizer, [windows]).to(self._device)
        with torch.inference_mode():
            scores = self._model(batch).sigmoid()[0].cpu().tolist()
        return {
            category: float(score)
            for category, score in zip(self.categories, scores, strict=True)
            if category != inp.mc_id
        }

    def run(self, inp: PipelineInput) -> list[int]:
        """Выбрать дополнительные категории с их фиксированными порогами."""
        return [
            category for category, score in self.probabilities(inp).items()
            if score >= self.thresholds[category]
        ]
