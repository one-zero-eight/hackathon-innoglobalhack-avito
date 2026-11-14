"""Локальный inference дообученного семантического encoder-классификатора."""

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import torch

from main import BasePipeline, PipelineInput
from subcategory_extraction.semantic_encoder.model import (
    DEFAULT_ARTIFACT,
    collate_documents,
    load_resource,
    tokenize_document,
)


class SubcategoryExtractionPipeline(BasePipeline):
    """Загрузить общий локальный encoder и определить дополнительные категории."""

    name = "semantic_encoder"

    def __init__(self) -> None:
        self.artifact_path = Path(os.getenv(
            "SEMANTIC_ENCODER_ARTIFACT_PATH", str(DEFAULT_ARTIFACT),
        )).expanduser().resolve()
        self._resource = load_resource(
            self.artifact_path, os.getenv("SEMANTIC_ENCODER_DEVICE", "auto"),
        )
        self.categories = tuple(self._resource.metadata["classes"])
        self.threshold = float(self._resource.metadata["threshold"])
        self._trained_categories = frozenset(self._resource.metadata["trained_classes"])

    @property
    def trained_categories(self) -> frozenset[int]:
        """Сообщить категории с положительными примерами официального train."""
        return self._trained_categories

    @property
    def unsupported_categories(self) -> frozenset[int]:
        """Сообщить категории без обучающих положительных примеров, включая 101."""
        return frozenset(self.categories) - self.trained_categories

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Вернуть независимую копию метаданных для общего отчёта."""
        return deepcopy(self._resource.metadata)

    @property
    def device(self) -> str:
        """Сообщить фактическое устройство inference."""
        return str(self._resource.device)

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Оценить все выходы головы без искусственного подавления категории 101."""
        windows = tokenize_document(self._resource.tokenizer, inp.description)
        batch = collate_documents(self._resource.tokenizer, [windows]).to(self._resource.device)
        with torch.inference_mode():
            scores = self._resource.model(batch).sigmoid()[0].cpu().tolist()
        return {
            category: float(score)
            for category, score in zip(self.categories, scores, strict=True)
            if category != inp.mc_id
        }

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории с вероятностью ≥ 0.5, кроме исходной категории."""
        return [
            category for category, score in self.probabilities(inp).items()
            if score >= self.threshold
        ]
