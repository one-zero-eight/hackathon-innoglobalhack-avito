"""Фильтр отдельных заказов поверх категорий snippet-логрега."""

import gzip
import json
import re
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer

from main import PipelineInput
from ..snippet_logistic_regression.pipeline import (
    WORD,
    SubcategoryExtractionPipeline as CategoryPipeline,
    features,
    identity,
    windows,
)

ARTIFACT = Path(__file__).with_name("artifacts") / "model.json.gz"
THRESHOLD = 0.5
BOUNDARY = re.compile(r"[.!?;\n]+")
ANCHOR = re.compile(
    r"не$|только$|исключительно$|целиком$|ключ$|"
    r"отдельн|частич|мелк|разов|локальн|комплекс|додел|минимальн"
)
HOMOGLYPHS = str.maketrans("aceopxykmtё", "асеорхуктме")


def order_features(description: str) -> list[str]:
    """Собрать бинарные признаки окон по всему объявлению."""
    normalized = description.casefold().translate(HOMOGLYPHS)
    snippets = []
    for clause in BOUNDARY.split(normalized):
        anchors = {token for token in WORD.findall(clause) if ANCHOR.match(token)}
        snippets.extend(windows(clause, anchors, width=11))
    return features(snippets, "pairs")


class OrderFormatClassifier:
    """Оценить общий отказ от частичных заказов."""

    def __init__(self) -> None:
        with gzip.open(ARTIFACT, "rt", encoding="utf-8") as stream:
            model = json.load(stream)
        self.vectorizer = CountVectorizer(
            analyzer=identity, binary=True, vocabulary=model["vocabulary"],
        )
        self.weights = np.asarray(model["weights"])
        self.intercept = float(model["intercept"])
        if (self.weights.shape != (len(model["vocabulary"]),)
                or not np.isfinite(self.weights).all() or not np.isfinite(self.intercept)):
            raise ValueError("Некорректные веса классификатора формата заказа")

    def refusal_probability(self, description: str) -> float:
        """Вернуть вероятность отказа, обученную на разметке формата заказа."""
        row = order_features(description)
        if not row:
            return 0.0
        score = (self.vectorizer.transform([row]) @ self.weights)[0] + self.intercept
        return float(expit(score))


class SubcategoryExtractionPipeline(CategoryPipeline):
    """Снять все категории при общем отказе от частичных заказов."""

    name = "order_format_logistic_regression"

    def __init__(self) -> None:
        super().__init__()
        self.order_format = OrderFormatClassifier()

    def run(self, inp: PipelineInput) -> list[int]:
        categories = super().run(inp)
        if categories and self.order_format.refusal_probability(inp.description) >= THRESHOLD:
            return []
        return categories
