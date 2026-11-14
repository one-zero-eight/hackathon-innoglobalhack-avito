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
                "Запустите uv run python src/subcategory_extraction/benchmarks/tfidf_quality/run.py"
            )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        self.categories = artifact["categories"]
        if artifact["version"] != 1 or self.categories != list(MICROCATEGORIES):
            raise ValueError("Несовместимый формат артефакта TF-IDF")
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

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть категории, кроме исходной, с учётом сохранённого порога."""
        features = self.vectorizer.transform([inp.description])
        logits = np.asarray(features @ self.weights.T).ravel() + self.intercepts
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
        return [
            cid for cid, value in zip(self.categories, probabilities, strict=True)
            if cid != inp.mc_id and value >= self.threshold
        ]
