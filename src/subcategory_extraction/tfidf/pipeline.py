"""Выделение категорий обученными локальными классификаторами."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from main import BasePipeline, MICROCATEGORIES, PipelineInput

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "out/tfidf.json"


class SubcategoryExtractionPipeline(BasePipeline):
    """Применить TF-IDF и веса логистических регрессий из JSON, без pickle."""

    name = "tfidf"

    def __init__(self) -> None:
        path = Path(os.getenv("TFIDF_ARTIFACT_PATH", str(DEFAULT_ARTIFACT))).expanduser()
        if not path.is_file():
            raise ValueError(
                f"Нет артефакта TF-IDF: {path}. "
                "Запустите PYTHONPATH=src uv run python -m subcategory_extraction.jev_tfidf.train"
            )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        self.categories = artifact["categories"]
        if artifact["version"] != 1 or self.categories != list(MICROCATEGORIES):
            raise ValueError("Несовместимый формат артефакта TF-IDF")
        trained_categories = artifact.get("trained_categories", self.categories)
        if (
            not isinstance(trained_categories, list)
            or any(type(cid) is not int or cid not in self.categories for cid in trained_categories)
            or len(set(trained_categories)) != len(trained_categories)
        ):
            raise ValueError("Некорректные обученные категории TF-IDF")
        self._trained_categories = frozenset(trained_categories)
        self.threshold = float(artifact["threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("Некорректный порог TF-IDF")
        vocabulary = artifact["vocabulary"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocabulary,
        )
        self.vectorizer.idf_ = np.asarray(artifact["idf"], dtype=float)
        self.weights = np.asarray(artifact["weights"], dtype=float)
        self.intercepts = np.asarray(artifact["intercepts"], dtype=float)
        if (
            self.weights.shape != (len(self.categories), len(vocabulary))
            or self.intercepts.shape != (len(self.categories),)
            or self.vectorizer.idf_.shape != (len(vocabulary),)
            or not all(np.isfinite(values).all() for values in (
                self.weights, self.intercepts, self.vectorizer.idf_,
            ))
        ):
            raise ValueError("Некорректные размеры или значения весов TF-IDF")

    @property
    def trained_categories(self) -> frozenset[int]:
        """Вернуть неизменяемый набор категорий с обученными головами."""
        return self._trained_categories

    def probabilities(self, inp: PipelineInput) -> dict[int, float]:
        """Оценить вероятности всех категорий, кроме исходной."""
        features = self.vectorizer.transform([inp.description])
        logits = np.asarray(features @ self.weights.T).ravel() + self.intercepts
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        return {
            cid: float(value)
            for cid, value in zip(self.categories, probabilities, strict=True)
            if cid != inp.mc_id
        }

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории, кроме исходной, с учётом сохранённого порога."""
        return [cid for cid, value in self.probabilities(inp).items() if value >= self.threshold]
